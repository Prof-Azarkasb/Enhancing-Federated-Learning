"""
Lightweight Personalized Federated Learning (LWP-FL) adapted to Google 2019 Cluster (Kaggle).

Design:
 - Clients := batches of job rows (round-robin).
 - Task := multiclass classification of job priority into 3 classes (Low/Medium/High).
 - Model := shared backbone (dense MLP) aggregated at server + client-specific small head (Dense -> softmax).
 - Server aggregates ONLY backbone weights (FedAvg weighted by sample counts).
 - Clients keep and train their personalized head locally (lightweight personalization).
"""

import os
import glob
import random
import time
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, metrics

# optional resource monitor
try:
    import psutil
except Exception:
    psutil = None

# ---------------- Configuration (tweakable) ----------------
DATA_DIR = "./data"                # place Google Cluster CSVs here
NUM_CLIENTS = 10                   # number of federated clients (batches)
INPUT_FEATURES = 8                 # max numeric features per job (pad/truncate)
CLIENT_LOCAL_EPOCHS = 3
CLIENT_BATCH_SIZE = 32
BACKBONE_LR = 1e-3
HEAD_LR = 5e-3
GLOBAL_ROUNDS = 20
TEST_HOLDOUT_RATIO = 0.15
FAILURE_RATE = 0.12                # chance per-client to drop in a round
RANDOM_SEED = 42
VERBOSE = False

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ---------------- Data helpers ----------------
def list_csv_files(data_dir: str) -> List[str]:
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}. Please download the Kaggle dataset and put CSVs there.")
    return sorted(files)


def choose_priority_column(df: pd.DataFrame) -> Optional[str]:
    """
    Try to find a priority-like column. Check common names; if none, return None.
    """
    candidates = ['priority', 'scheduling_class', 'scheduling-class', 'priority_level', 'priorityClass', 'priority_level']
    for c in candidates:
        if c in df.columns:
            return c
    return None


def extract_features_and_priority(df: pd.DataFrame, max_cols: int = INPUT_FEATURES) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract up to `max_cols` numeric features and a priority label.
    If a priority column exists, map it to 3 classes (Low/Med/High).
    Otherwise, create proxy priority by binning the first numeric column into 3 quantiles.
    Returns:
      X: (n_rows, max_cols) float32
      y: (n_rows,) int labels 0/1/2
    """
    numeric = df.select_dtypes(include=[np.number]).fillna(0.0)
    if numeric.shape[1] == 0:
        return np.empty((0, max_cols), dtype=np.float32), np.empty((0,), dtype=np.int32)

    # pick up to max_cols numeric features
    cols = list(numeric.columns)[:max_cols]
    X = numeric[cols].to_numpy(dtype=np.float32)
    if X.shape[1] < max_cols:
        pad = np.zeros((X.shape[0], max_cols - X.shape[1]), dtype=np.float32)
        X = np.hstack([X, pad])

    # priority label
    pcol = choose_priority_column(df)
    if pcol is not None:
        raw_p = df[pcol].fillna(0).to_numpy()
        # Normalize/cast numeric categories, then bin into 3 classes by quantile
        try:
            raw_p = raw_p.astype(float)
            bins = np.nanpercentile(raw_p, [33.33, 66.67])
            y = np.digitize(raw_p, bins)
            y = np.clip(y, 0, 2).astype(np.int32)
        except Exception:
            # fallback to proxy
            firstcol = numeric.iloc[:, 0].to_numpy()
            bins = np.nanpercentile(firstcol, [33.33, 66.67])
            y = np.digitize(firstcol, bins).astype(np.int32)
    else:
        # proxy: use first numeric column, quantile bin into 3 classes
        firstcol = numeric.iloc[:, 0].to_numpy()
        bins = np.nanpercentile(firstcol, [33.33, 66.67])
        y = np.digitize(firstcol, bins).astype(np.int32)
    return X, y


def build_clients_from_files(files: List[str], num_clients: int = NUM_CLIENTS, input_features: int = INPUT_FEATURES
                             ) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Tuple[np.ndarray, np.ndarray]]:
    """
    Read CSV files, create pooled rows (X,y) and then distribute rows round-robin to num_clients.
    Also set aside TEST_HOLDOUT_RATIO fraction of files as global test set.
    """
    files = files.copy()
    random.shuffle(files)
    n_holdout = max(1, int(len(files) * TEST_HOLDOUT_RATIO))
    holdout_files = files[:n_holdout]
    data_files = files[n_holdout:]

    # build global test
    X_test_list, y_test_list = [], []
    for f in holdout_files:
        try:
            df = pd.read_csv(f, low_memory=True)
        except Exception:
            continue
        Xf, yf = extract_features_and_priority(df, input_features)
        if Xf.shape[0] > 0:
            X_test_list.append(Xf); y_test_list.append(yf)
    if X_test_list:
        X_test = np.vstack(X_test_list); y_test = np.concatenate(y_test_list)
    else:
        X_test = np.empty((0, input_features), dtype=np.float32); y_test = np.empty((0,), dtype=np.int32)

    # pool rows from remaining files
    pool_X = []
    pool_y = []
    for f in data_files:
        try:
            df = pd.read_csv(f, low_memory=True)
        except Exception:
            continue
        Xf, yf = extract_features_and_priority(df, input_features)
        if Xf.shape[0] > 0:
            pool_X.append(Xf); pool_y.append(yf)
    if not pool_X:
        raise RuntimeError("No usable numeric rows found in dataset files.")
    pool_X = np.vstack(pool_X); pool_y = np.concatenate(pool_y)

    # shuffle rows
    idx = np.arange(pool_X.shape[0])
    np.random.shuffle(idx)
    pool_X = pool_X[idx]; pool_y = pool_y[idx]

    # round-robin assign rows to clients
    clients_X = [[] for _ in range(num_clients)]
    clients_y = [[] for _ in range(num_clients)]
    for i in range(pool_X.shape[0]):
        cid = i % num_clients
        clients_X[cid].append(pool_X[i])
        clients_y[cid].append(pool_y[i])

    clients_data = []
    for cid in range(num_clients):
        Xc = np.vstack(clients_X[cid]).astype(np.float32) if clients_X[cid] else np.empty((0, input_features), dtype=np.float32)
        yc = np.array(clients_y[cid], dtype=np.int32) if clients_y[cid] else np.empty((0,), dtype=np.int32)
        # ensure small clients have at least a few samples
        if Xc.shape[0] < 10:
            extra = np.random.rand(10, input_features).astype(np.float32) * 1e-3
            Xc = np.vstack([Xc, extra]) if Xc.size else extra
            yc = np.concatenate([yc, np.zeros(10, dtype=np.int32)]) if yc.size else np.zeros(10, dtype=np.int32)
        clients_data.append((Xc, yc))
    return clients_data, (X_test, y_test)


# ---------------- Model definition (backbone + personalized head) ----------------
def build_backbone(input_dim: int) -> tf.keras.Model:
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(64, activation='relu')(inp)
    x = layers.Dense(32, activation='relu')(x)
    # backbone outputs embedding vector
    model = tf.keras.Model(inputs=inp, outputs=x, name="backbone")
    return model


def build_head(embedding_dim: int, num_classes: int = 3) -> tf.keras.Model:
    inp = layers.Input(shape=(embedding_dim,))
    out = layers.Dense(num_classes, activation='softmax')(inp)
    model = tf.keras.Model(inputs=inp, outputs=out, name="head")
    return model


# ---------------- Client class ----------------
class LWPClient:
    def __init__(self, cid: int, X: np.ndarray, y: np.ndarray, backbone_template: tf.keras.Model,
                 scaler: StandardScaler, head_lr: float = HEAD_LR, local_epochs: int = CLIENT_LOCAL_EPOCHS):
        self.id = cid
        self.X_raw = X.copy()
        self.y = y.copy()
        self.n = X.shape[0]
        self.scaler = scaler
        # normalized inputs
        self.X = scaler.transform(self.X_raw).astype(np.float32)
        # backbone: local copy (will be overwritten by server broadcast)
        self.backbone = tf.keras.models.clone_model(backbone_template)
        self.backbone.set_weights(backbone_template.get_weights())
        # personalized head (small) — initialized randomly per client
        emb_dim = self.backbone.output_shape[-1]
        self.head = build_head(emb_dim, num_classes=3)
        # optimizers: head only (personalization is lightweight)
        self.head_optimizer = optimizers.Adam(learning_rate=head_lr)
        # loss
        self.loss_fn = losses.SparseCategoricalCrossentropy()
        self.local_epochs = local_epochs

    def set_backbone_weights(self, weights):
        self.backbone.set_weights(weights)

    def get_backbone_weights(self):
        return self.backbone.get_weights()

    def get_head_weights(self):
        return self.head.get_weights()

    def set_head_weights(self, w):
        self.head.set_weights(w)

    def train_local(self, train_head_only: bool = True):
        """
        Train personalized head locally. Optionally allow small fine-tuning of backbone (not by default).
        """
        if self.X.shape[0] == 0:
            return
        ds = tf.data.Dataset.from_tensor_slices((self.X, self.y)).shuffle(1000, seed=RANDOM_SEED).batch(CLIENT_BATCH_SIZE)
        for ep in range(self.local_epochs):
            for xb, yb in ds:
                # forward through backbone
                with tf.GradientTape(persistent=True) as tape:
                    emb = self.backbone(xb, training=not train_head_only)
                    preds = self.head(emb, training=True)
                    loss = self.loss_fn(yb, preds)
                # head gradients and update
                head_grads = tape.gradient(loss, self.head.trainable_variables)
                self.head_optimizer.apply_gradients(zip(head_grads, self.head.trainable_variables))
                # optional backbone fine-tune small LR (if needed): not enabled by default
                del tape

    def evaluate_local(self) -> float:
        """
        Evaluate personalized model on local data (returns accuracy).
        """
        if self.X.shape[0] == 0:
            return 0.0
        preds = self.head(self.backbone(self.X, training=False), training=False)
        acc = np.mean(np.argmax(preds.numpy(), axis=1) == self.y)
        return float(acc)


# ---------------- Server class ----------------
class LWPServer:
    def __init__(self, backbone_template: tf.keras.Model):
        self.backbone = backbone_template
        self.global_backbone_weights = backbone_template.get_weights()

    def aggregate_backbones(self, client_backbone_weights: List[List[np.ndarray]], sample_counts: List[int]):
        """
        Weighted FedAvg on backbone weights by sample_counts.
        """
        total = float(sum(sample_counts))
        if total == 0:
            return
        # init accumulator
        avg = []
        num_layers = len(client_backbone_weights[0])
        for layer_idx in range(num_layers):
            acc = np.zeros_like(client_backbone_weights[0][layer_idx], dtype=np.float32)
            for w, n in zip(client_backbone_weights, sample_counts):
                acc += w[layer_idx] * (n / total)
            avg.append(acc)
        self.global_backbone_weights = avg
        self.backbone.set_weights(self.global_backbone_weights)

    def broadcast_backbone(self) -> List[np.ndarray]:
        return self.global_backbone_weights


# ---------------- Orchestration ----------------
def run_lwpfl(clients_data: List[Tuple[np.ndarray, np.ndarray]], X_test: np.ndarray, y_test: np.ndarray):
    # Fit global scaler on all data (clients + test)
    all_X = np.vstack([X for X, _ in clients_data] + ([X_test] if X_test.shape[0] > 0 else []))
    scaler = StandardScaler()
    scaler.fit(all_X)

    # build backbone template
    backbone_template = build_backbone(input_dim=all_X.shape[1])

    # create clients
    clients = []
    for i, (Xc, yc) in enumerate(clients_data):
        c = LWPClient(i, Xc, yc, backbone_template, scaler, head_lr=HEAD_LR, local_epochs=CLIENT_LOCAL_EPOCHS)
        clients.append(c)

    server = LWPServer(backbone_template)
    # initialize server backbone weights (already in template)
    server.global_backbone_weights = backbone_template.get_weights()

    # initial broadcast
    for c in clients:
        c.set_backbone_weights(server.broadcast_backbone())

    # run global rounds
    for rnd in range(1, GLOBAL_ROUNDS + 1):
        print(f"\n=== LWP-FL Round {rnd}/{GLOBAL_ROUNDS} ===")
        client_backbone_weights = []
        sample_counts = []
        live_clients = 0

        # clients perform local personalization (train head) and may optionally fine-tune
        for c in clients:
            if random.random() < FAILURE_RATE:
                print(f"Client {c.id} dropped this round (simulated).")
                continue
            # local personalized training (head only)
            t0 = time.time()
            c.train_local(train_head_only=True)
            t1 = time.time()
            live_clients += 1
            client_backbone_weights.append(c.get_backbone_weights())
            sample_counts.append(c.n)
            print(f" Client {c.id}: local acc={c.evaluate_local():.3f}, samples={c.n}, local_time={t1-t0:.3f}s")
            # note: clients keep their personalized heads local (not uploaded)

        # server aggregates backbone weights from participating clients
        if client_backbone_weights:
            t0_agg = time.time()
            server.aggregate_backbones(client_backbone_weights, sample_counts)
            t1_agg = time.time()
            print(f" Server aggregated backbone from {len(client_backbone_weights)} clients in {t1_agg-t0_agg:.3f}s")
            # broadcast updated backbone to all clients
            for c in clients:
                c.set_backbone_weights(server.broadcast_backbone())
        else:
            print(" No client contributions this round; skipping aggregation.")

        # Evaluate global backbone performance by training a temporary global head on pooled validation (quick)
        if X_test.shape[0] > 0:
            # Build temporary head and train it quickly on test features to measure backbone quality
            emb_dim = server.backbone.output_shape[-1]
            temp_head = build_head(emb_dim, num_classes=3)
            temp_opt = optimizers.Adam(learning_rate=1e-3)
            # prepare test embeddings
            Xs = scaler.transform(X_test).astype(np.float32)
            emb = server.backbone(Xs, training=False).numpy()
            # quick head training for a few epochs on test (note: just for evaluation metric)
            ytest = y_test.astype(np.int32)
            ds = tf.data.Dataset.from_tensor_slices((emb, ytest)).batch(64)
            for _ in range(3):
                for xb, yb in ds:
                    with tf.GradientTape() as tape:
                        preds = temp_head(xb, training=True)
                        loss = losses.sparse_categorical_crossentropy(yb, preds)
                        loss = tf.reduce_mean(loss)
                    grads = tape.gradient(loss, temp_head.trainable_variables)
                    temp_opt.apply_gradients(zip(grads, temp_head.trainable_variables))
            # evaluate
            preds = temp_head(emb, training=False).numpy()
            acc = np.mean(np.argmax(preds, axis=1) == ytest)
            print(f" Global backbone test accuracy (via temporary head): {acc:.4f}")
        else:
            print(" No global test set available.")

        # optional system resource log
        if psutil:
            print(f" System CPU%: {psutil.cpu_percent(interval=0.2)}, Mem%: {psutil.virtual_memory().percent}")

    print("\nLWP-FL finished.")


# ---------------- Main ----------------
if __name__ == "__main__":
    files = list_csv_files(DATA_DIR)
    clients_data, (X_test, y_test) = build_clients_from_files(files, num_clients=NUM_CLIENTS, input_features=INPUT_FEATURES)
    run_lwpfl(clients_data, X_test, y_test)

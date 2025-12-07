"""
FL-CN: Federated Learning with Critical Nodes (Google 2019 Cluster sample)
"""

import os
import glob
import random
import time
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, metrics

# optional monitoring
try:
    import psutil
except Exception:
    psutil = None

# ---------------- Configuration ----------------
DATA_DIR = "./data"                    # Put Google 2019 Cluster CSVs here
NUM_CLIENTS = 10                       # number of pseudo-clients (batches)
CLIENT_FRACTION = 0.6                  # fraction of clients to participate per round
GLOBAL_ROUNDS = 12
LOCAL_EPOCHS = 3
LOCAL_BATCH = 32
BACKBONE_LR = 1e-3
INPUT_FEATURES = 8                     # first N numeric features per job
TEST_HOLDOUT_RATIO = 0.15              # fraction of files held out as test
CRITICAL_NODE_THRESHOLD = 0.7          # threshold for marking node critical (priority normalized)
MAX_CRITICAL_NODES = 5
FAILURE_RATE = 0.12                    # random dropout per client per round
RANDOM_SEED = 42
VERBOSE = False

# seeds
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ---------------- Data utilities ----------------
def list_csv_files(data_dir: str) -> List[str]:
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files in {data_dir}. Download Kaggle dataset and put CSVs there.")
    return sorted(files)


def choose_priority_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ['priority', 'scheduling_class', 'scheduling-class', 'priority_level', 'priorityClass']
    for c in candidates:
        if c in df.columns:
            return c
    return None


def extract_features_and_priority(df: pd.DataFrame, max_cols: int = INPUT_FEATURES) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract up to max_cols numeric features and construct priority label (3 classes).
    If explicit priority column exists it is used and binned into 3 classes; otherwise
    use the first numeric column as proxy and bin into 3 quantiles.
    """
    numeric = df.select_dtypes(include=[np.number]).fillna(0.0)
    if numeric.shape[1] == 0:
        return np.empty((0, max_cols), dtype=np.float32), np.empty((0,), dtype=np.int32)

    cols = list(numeric.columns)[:max_cols]
    X = numeric[cols].to_numpy(dtype=np.float32)
    if X.shape[1] < max_cols:
        pad = np.zeros((X.shape[0], max_cols - X.shape[1]), dtype=np.float32)
        X = np.hstack([X, pad])

    pcol = choose_priority_column(df)
    if pcol is not None:
        raw_p = df[pcol].fillna(0).to_numpy()
        try:
            raw_p = raw_p.astype(float)
            bins = np.nanpercentile(raw_p, [33.33, 66.67])
            y = np.digitize(raw_p, bins)
            y = np.clip(y, 0, 2).astype(np.int32)
        except Exception:
            firstcol = numeric.iloc[:, 0].to_numpy()
            bins = np.nanpercentile(firstcol, [33.33, 66.67])
            y = np.digitize(firstcol, bins).astype(np.int32)
    else:
        firstcol = numeric.iloc[:, 0].to_numpy()
        bins = np.nanpercentile(firstcol, [33.33, 66.67])
        y = np.digitize(firstcol, bins).astype(np.int32)

    return X, y


def build_clients_from_files(files: List[str], num_clients: int = NUM_CLIENTS, input_features: int = INPUT_FEATURES
                             ) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Tuple[np.ndarray, np.ndarray]]:
    """
    Read CSVs, use a fraction of files as global test set, pool rows and assign round-robin to clients.
    Returns clients_data list and global test (X_test, y_test).
    """
    files = files.copy()
    random.shuffle(files)
    n_holdout = max(1, int(len(files) * TEST_HOLDOUT_RATIO))
    holdout_files = files[:n_holdout]
    pool_files = files[n_holdout:]

    # build test
    X_test_list, y_test_list = [], []
    for f in holdout_files:
        try:
            df = pd.read_csv(f, low_memory=True)
        except Exception:
            continue
        Xf, yf = extract_features_and_priority(df, input_features)
        if Xf.shape[0] > 0:
            X_test_list.append(Xf); y_test_list.append(yf)
    X_test = np.vstack(X_test_list) if X_test_list else np.empty((0, input_features), dtype=np.float32)
    y_test = np.concatenate(y_test_list) if y_test_list else np.empty((0,), dtype=np.int32)

    # pool rows from remaining files
    pool_X, pool_y = [], []
    for f in pool_files:
        try:
            df = pd.read_csv(f, low_memory=True)
        except Exception:
            continue
        Xf, yf = extract_features_and_priority(df, input_features)
        if Xf.shape[0] > 0:
            pool_X.append(Xf); pool_y.append(yf)
    if not pool_X:
        raise RuntimeError("No usable numeric data found in dataset files.")
    pool_X = np.vstack(pool_X); pool_y = np.concatenate(pool_y)

    # shuffle and round-robin assign
    idx = np.arange(pool_X.shape[0])
    np.random.shuffle(idx)
    pool_X = pool_X[idx]; pool_y = pool_y[idx]

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
        # ensure at least small dataset
        if Xc.shape[0] < 8:
            extra = np.random.rand(8, input_features).astype(np.float32) * 1e-3
            Xc = np.vstack([Xc, extra]) if Xc.size else extra
            yc = np.concatenate([yc, np.zeros(8, dtype=np.int32)]) if yc.size else np.zeros(8, dtype=np.int32)
        clients_data.append((Xc, yc))
    return clients_data, (X_test, y_test)


# ---------------- Model factory ----------------
def create_classifier(input_dim: int, num_classes: int = 3) -> tf.keras.Model:
    model = models.Sequential([
        layers.InputLayer(input_shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=BACKBONE_LR),
                  loss=losses.SparseCategoricalCrossentropy(),
                  metrics=[metrics.SparseCategoricalAccuracy()])
    return model


# ---------------- CriticalNode (client) ----------------
class CriticalNode:
    def __init__(self, cid: int, X: np.ndarray, y: np.ndarray, model_template: tf.keras.Model):
        self.id = cid
        self.X_raw = X.copy()
        self.y = y.copy()
        self.n = X.shape[0]
        # normalized features will be set by server/global scaler
        self.X = None
        # node model - separate copy per node
        self.model = tf.keras.models.clone_model(model_template)
        self.model.set_weights(model_template.get_weights())
        # compute proxy computational power & latency from data statistics
        # comp_power: normalized mean magnitude of features (higher -> more compute capability proxy)
        if self.n > 0:
            feat_mean = float(np.mean(np.abs(self.X_raw)))
            feat_std = float(np.std(self.X_raw))
            # normalize into [0,1] by a simple sigmoid-like mapping
            self.comp_power = 1.0 / (1.0 + np.exp(- (feat_mean / (1.0 + feat_std))))
        else:
            self.comp_power = random.uniform(0.1, 1.0)
        # latency proxy: (1 - comp_power) + small noise, in [0,1]
        self.latency = float(np.clip(1.0 - self.comp_power + random.uniform(-0.05, 0.05), 0.0, 1.0))

    def set_inputs(self, X_scaled: np.ndarray):
        self.X = X_scaled.astype(np.float32)

    def set_weights(self, weights: List[np.ndarray]):
        self.model.set_weights(weights)

    def get_weights(self) -> List[np.ndarray]:
        return self.model.get_weights()

    def train_local(self, epochs: int = LOCAL_EPOCHS, batch_size: int = LOCAL_BATCH) -> Tuple[float, float]:
        """
        Local train; returns (loss, accuracy).
        """
        if self.X is None or self.X.shape[0] == 0:
            return float('nan'), float('nan')
        start_cpu = psutil.cpu_percent(interval=None) if psutil else 0.0
        start_mem = psutil.virtual_memory().percent if psutil else 0.0

        history = self.model.fit(self.X, self.y, epochs=epochs, batch_size=batch_size, verbose=0)
        loss = float(history.history['loss'][-1]) if 'loss' in history.history else float('nan')
        acc = float(history.history['sparse_categorical_accuracy'][-1]) if 'sparse_categorical_accuracy' in history.history else float('nan')

        end_cpu = psutil.cpu_percent(interval=None) if psutil else 0.0
        end_mem = psutil.virtual_memory().percent if psutil else 0.0
        if VERBOSE:
            print(f"Client {self.id}: CPU Δ {end_cpu-start_cpu:.2f}%, Mem Δ {end_mem-start_mem:.2f}%")
        return loss, acc

    def get_node_priority(self) -> float:
        """
        Compute priority score used by CTIS-like selection: higher -> more 'critical'.
        Use comp_power / (latency + eps), normalize later at server.
        """
        return self.comp_power / (self.latency + 1e-9)


# ---------------- Server ----------------
class FLServerCN:
    def __init__(self, model_template: tf.keras.Model, scaler: StandardScaler):
        self.global_model = tf.keras.models.clone_model(model_template)
        self.global_model.set_weights(model_template.get_weights())
        self.scaler = scaler
        self.global_weights = self.global_model.get_weights()

    def broadcast(self) -> List[np.ndarray]:
        return self.global_weights

    def aggregate_weights(self, client_weights: List[List[np.ndarray]], sample_counts: List[int]):
        """
        Weighted average aggregation by sample_counts (FedAvg-style).
        """
        total = float(sum(sample_counts)) if sample_counts else 0.0
        if total == 0 or not client_weights:
            return
        num_layers = len(client_weights[0])
        new_weights = []
        for li in range(num_layers):
            acc = np.zeros_like(client_weights[0][li], dtype=np.float32)
            for w, n in zip(client_weights, sample_counts):
                acc += w[li] * (n / total)
            new_weights.append(acc)
        self.global_weights = new_weights
        self.global_model.set_weights(self.global_weights)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
        if X_test.shape[0] == 0:
            return float('nan'), float('nan')
        loss, acc = self.global_model.evaluate(X_test, y_test, verbose=0)
        return float(loss), float(acc)


# ---------------- Orchestration ----------------
def run_fl_cn(clients_data: List[Tuple[np.ndarray, np.ndarray]], X_test: np.ndarray, y_test: np.ndarray):
    # build scaler on whole training+test
    all_X = np.vstack([X for X, _ in clients_data] + ([X_test] if X_test.shape[0] > 0 else []))
    scaler = StandardScaler()
    scaler.fit(all_X)

    # build model template
    model_template = create_classifier(input_dim=all_X.shape[1], num_classes=3)

    # create nodes
    nodes: List[CriticalNode] = []
    for cid, (Xc, yc) in enumerate(clients_data):
        node = CriticalNode(cid, Xc, yc, model_template)
        # set scaled inputs
        Xc_s = scaler.transform(Xc).astype(np.float32) if Xc.shape[0] > 0 else np.empty((0, all_X.shape[1]), dtype=np.float32)
        node.set_inputs(Xc_s)
        nodes.append(node)

    server = FLServerCN(model_template, scaler)

    # main federated rounds
    for rnd in range(1, GLOBAL_ROUNDS + 1):
        print(f"\n--- FL-CN Round {rnd}/{GLOBAL_ROUNDS} ---")

        # compute priority scores and normalize to [0,1]
        scores = np.array([node.get_node_priority() for node in nodes], dtype=np.float32)
        if np.all(np.isfinite(scores)):
            min_s, max_s = float(np.min(scores)), float(np.max(scores))
            if max_s - min_s > 1e-9:
                norm_scores = (scores - min_s) / (max_s - min_s)
            else:
                norm_scores = np.ones_like(scores)
        else:
            norm_scores = np.zeros_like(scores)

        # select critical nodes: those with normalized score >= threshold
        critical_idx = [i for i, s in enumerate(norm_scores) if s >= CRITICAL_NODE_THRESHOLD]
        critical_nodes = [nodes[i] for i in critical_idx]
        # limit to MAX_CRITICAL_NODES
        if len(critical_nodes) > MAX_CRITICAL_NODES:
            # choose top by score
            order = np.argsort(-norm_scores[critical_idx])
            sel_idx = [critical_idx[i] for i in order[:MAX_CRITICAL_NODES]]
            critical_nodes = [nodes[i] for i in sel_idx]

        # ensure at least CLIENT_FRACTION are selected (fill with highest remaining)
        required = max(1, int(np.ceil(CLIENT_FRACTION * len(nodes))))
        if len(critical_nodes) < required:
            # pick highest scoring nodes not already selected
            remaining = [i for i in range(len(nodes)) if i not in critical_idx]
            remaining_sorted = sorted(remaining, key=lambda i: -norm_scores[i])
            to_add = remaining_sorted[:max(0, required - len(critical_nodes))]
            critical_nodes.extend([nodes[i] for i in to_add])

        print(f" Selected critical nodes: {[n.id for n in critical_nodes]} (threshold {CRITICAL_NODE_THRESHOLD}, max {MAX_CRITICAL_NODES})")

        # also consider possible additional participants (optional): here we use only selected nodes
        participants = critical_nodes.copy()

        # step 1: broadcast global weights
        global_w = server.broadcast()
        for n in participants:
            n.set_weights(global_w)

        # step 2: each participant trains locally (with failure simulation)
        client_weights = []
        sample_counts = []
        for n in participants:
            if random.random() < FAILURE_RATE:
                print(f" Node {n.id} failed to communicate this round.")
                continue
            t0 = time.time()
            loss, acc = n.train_local(epochs=LOCAL_EPOCHS, batch_size=LOCAL_BATCH)
            t1 = time.time()
            client_weights.append(n.get_weights())
            sample_counts.append(n.n)
            print(f" Node {n.id} trained: loss={loss:.4f}, acc={acc:.4f}, time={t1-t0:.2f}s, comp_power={n.comp_power:.3f}, latency={n.latency:.3f}")

        # step 3: aggregate if we have contributions
        if client_weights:
            t0 = time.time()
            server.aggregate_weights(client_weights, sample_counts)
            t1 = time.time()
            print(f" Server aggregated {len(client_weights)} clients in {t1-t0:.3f}s")
            # broadcast aggregated weights to all nodes (so next round they start from same global)
            new_global = server.broadcast()
            for n in nodes:
                n.set_weights(new_global)
        else:
            print(" No client contributions this round; global model unchanged.")

        # evaluation on held-out test set
        if X_test.shape[0] > 0:
            Xs = scaler.transform(X_test).astype(np.float32)
            loss, acc = server.evaluate(Xs, y_test)
            print(f" Global test -- loss: {loss:.4f}, acc: {acc:.4f}")
        else:
            print(" No test set available for evaluation.")

        if psutil:
            print(f" System CPU%: {psutil.cpu_percent(interval=0.2)}, Mem%: {psutil.virtual_memory().percent}")

    print("\nFL-CN finished.")


# ---------------- Main ----------------
if __name__ == "__main__":
    files = list_csv_files(DATA_DIR)
    clients_data, (X_test, y_test) = build_clients_from_files(files, num_clients=NUM_CLIENTS, input_features=INPUT_FEATURES)
    run_fl_cn(clients_data, X_test, y_test)

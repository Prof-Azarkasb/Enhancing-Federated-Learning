"""
grafl_signsgd_google_cluster.py
Gradient Federated Learning (SignSGD-style with optional momentum) adapted for
Google 2019 Cluster sample (Kaggle). Clients are batches of job rows (round-robin).
"""

import os
import glob
import random
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, metrics

# optional system monitor
try:
    import psutil
except Exception:
    psutil = None

# ------------------ Configuration ------------------
DATA_DIR = "./data"               # folder with Google Cluster CSVs
NUM_CLIENTS = 10                  # number of federated clients (batches)
INPUT_FEATURES = 8                # number of numeric columns per job (pad/truncate)
LOCAL_EPOCHS = 2                  # local epochs used to compute gradients
LOCAL_BATCH = 32
GLOBAL_ROUNDS = 30
SIGNSGD_LR = 0.01                 # server step size
MOMENTUM = 0.9                    # momentum coefficient (use 0.0 to disable)
FAILURE_RATE = 0.12               # probability a client fails to send grads
TEST_HOLDOUT_RATIO = 0.15
RANDOM_SEED = 42

# reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ------------------ Data utilities ------------------
def list_csv_files(data_dir: str) -> List[str]:
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}. Place the Kaggle CSV files there.")
    return sorted(files)


def extract_numeric_features_from_df(df: pd.DataFrame, max_cols: int = INPUT_FEATURES
                                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select up to max_cols numeric columns from df, pad/truncate to shape (n_rows, max_cols).
    Target y is row-wise mean of selected numeric columns (proxy regression target).
    """
    numeric = df.select_dtypes(include=[np.number]).fillna(0.0)
    if numeric.shape[1] == 0:
        return np.empty((0, max_cols), dtype=np.float32), np.empty((0,), dtype=np.float32)
    cols = list(numeric.columns)[:max_cols]
    X = numeric[cols].to_numpy(dtype=np.float32)
    if X.shape[1] < max_cols:
        pad = np.zeros((X.shape[0], max_cols - X.shape[1]), dtype=np.float32)
        X = np.hstack([X, pad])
    y = np.mean(X[:, :len(cols)], axis=1).astype(np.float32)
    return X, y


def build_clients_from_files(files: List[str], num_clients: int = NUM_CLIENTS, input_features: int = INPUT_FEATURES
                             ) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Tuple[np.ndarray, np.ndarray]]:
    """
    Read CSVs, convert each file to (Xf, yf), pool all rows from non-holdout files,
    then assign rows round-robin into num_clients batches.
    Returns clients_data list of (X_client, y_client) and (X_test, y_test).
    """
    files = files.copy()
    random.shuffle(files)
    n_holdout = max(1, int(len(files) * TEST_HOLDOUT_RATIO))
    holdout_files = files[:n_holdout]
    pool_files = files[n_holdout:]

    # build test set from holdout files
    X_test_list, y_test_list = [], []
    for f in holdout_files:
        try:
            df = pd.read_csv(f, low_memory=True)
        except Exception:
            continue
        Xf, yf = extract_numeric_features_from_df(df, input_features)
        if Xf.shape[0] > 0:
            X_test_list.append(Xf); y_test_list.append(yf)
    if X_test_list:
        X_test = np.vstack(X_test_list); y_test = np.concatenate(y_test_list)
    else:
        X_test = np.empty((0, input_features), dtype=np.float32); y_test = np.empty((0,), dtype=np.float32)

    # pool rows from pool_files
    pool_X = []
    pool_y = []
    for f in pool_files:
        try:
            df = pd.read_csv(f, low_memory=True)
        except Exception:
            continue
        Xf, yf = extract_numeric_features_from_df(df, input_features)
        if Xf.shape[0] > 0:
            pool_X.append(Xf); pool_y.append(yf)
    if not pool_X:
        raise RuntimeError("No usable numeric rows found in dataset files.")
    pool_X = np.vstack(pool_X); pool_y = np.concatenate(pool_y)

    # shuffle rows
    idx = np.arange(pool_X.shape[0])
    np.random.shuffle(idx)
    pool_X = pool_X[idx]; pool_y = pool_y[idx]

    # round-robin assignment to clients
    clients_chunks_X = [[] for _ in range(num_clients)]
    clients_chunks_y = [[] for _ in range(num_clients)]
    for i in range(pool_X.shape[0]):
        cid = i % num_clients
        clients_chunks_X[cid].append(pool_X[i])
        clients_chunks_y[cid].append(pool_y[i])

    clients_data = []
    for cid in range(num_clients):
        Xc = np.vstack(clients_chunks_X[cid]).astype(np.float32) if clients_chunks_X[cid] else np.empty((0, input_features), dtype=np.float32)
        yc = np.array(clients_chunks_y[cid], dtype=np.float32) if clients_chunks_y[cid] else np.empty((0,), dtype=np.float32)
        # ensure clients not empty
        if Xc.shape[0] < 8:
            extra = np.random.rand(8, input_features).astype(np.float32) * 1e-3
            Xc = np.vstack([Xc, extra]) if Xc.size else extra
            yc = np.concatenate([yc, np.zeros(8, dtype=np.float32)]) if yc.size else np.zeros(8, dtype=np.float32)
        clients_data.append((Xc, yc))
    return clients_data, (X_test, y_test)


# ------------------ Model factory ------------------
def create_regressor(input_dim: int):
    model = models.Sequential([
        layers.InputLayer(input_shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='linear')
    ])
    # compile not necessary for manual gradient computations but useful for evaluation
    model.compile(optimizer=optimizers.SGD(0.01), loss=losses.MeanSquaredError(), metrics=[metrics.MeanSquaredError()])
    return model


# ------------------ Client (computes grads signs) ------------------
class GraClient:
    def __init__(self, cid: int, X: np.ndarray, y: np.ndarray):
        self.id = cid
        self.X = X
        self.y = y
        self.n = X.shape[0]
        self.model = create_regressor(self.X.shape[1])
        self.scaler = None

    def set_weights(self, weights: List[np.ndarray]):
        self.model.set_weights(weights)

    def get_weights(self) -> List[np.ndarray]:
        return self.model.get_weights()

    def compute_avg_gradients(self, epochs: int = LOCAL_EPOCHS, batch_size: int = LOCAL_BATCH) -> List[np.ndarray]:
        """
        Compute average gradients for local data over `epochs` and return list of numpy arrays (float)
        """
        if self.X.shape[0] == 0:
            return None
        ds = tf.data.Dataset.from_tensor_slices((self.X, self.y)).shuffle(1000, seed=RANDOM_SEED).batch(batch_size)
        accum = None
        steps = 0
        for ep in range(epochs):
            for xb, yb in ds:
                with tf.GradientTape() as tape:
                    preds = self.model(xb, training=True)
                    loss = tf.reduce_mean(tf.keras.losses.MSE(yb, tf.squeeze(preds)))
                grads = tape.gradient(loss, self.model.trainable_variables)
                grads_np = [g.numpy().astype(np.float32) if g is not None else np.zeros_like(w.numpy()) for g, w in zip(grads, self.model.trainable_variables)]
                if accum is None:
                    accum = [np.array(g, dtype=np.float32) for g in grads_np]
                else:
                    for i in range(len(accum)):
                        accum[i] += grads_np[i]
                steps += 1
        if steps == 0 or accum is None:
            return None
        avg = [a / float(steps) for a in accum]
        return avg

    def compute_signs(self, avg_grads: List[np.ndarray]) -> List[np.int8]:
        """
        Convert averaged gradients to sign arrays (-1,0,1) per weight array
        """
        if avg_grads is None:
            return None
        signs = [np.sign(g).astype(np.int8) for g in avg_grads]
        return signs


# ------------------ Server (aggregates signs & updates global) ------------------
class GraServer:
    def __init__(self, clients: List[GraClient], lr: float = SIGNSGD_LR, momentum: float = MOMENTUM):
        self.clients = clients
        self.lr = lr
        self.momentum_coeff = momentum
        # initialize global weights from first client
        init_weights = [c.get_weights() for c in clients if c.get_weights()]
        self.global_weights = init_weights[0] if init_weights else None
        # initialize momentum buffers matching weight shapes (float)
        if self.global_weights is not None:
            self.momentum_buffers = [np.zeros_like(w, dtype=np.float32) for w in self.global_weights]
        else:
            self.momentum_buffers = None

    def broadcast(self):
        if self.global_weights is None:
            return
        for c in self.clients:
            c.set_weights(self.global_weights)

    def aggregate_and_update(self, signs_from_clients: List[List[np.ndarray]]):
        """
        signs_from_clients: list of clients, each is list of sign arrays per-layer
        Aggregation: sum signs per element => aggregated_sum array
        If momentum_coeff > 0: momentum = momentum_coeff * momentum + aggregated_sum
        update: global_weights -= lr * sign(momentum_or_aggregated_sum)
        """
        if not signs_from_clients:
            return
        layer_count = len(signs_from_clients[0])
        # accumulator
        acc = [np.zeros_like(signs_from_clients[0][i], dtype=np.int32) for i in range(layer_count)]
        for s in signs_from_clients:
            for i in range(layer_count):
                acc[i] += s[i].astype(np.int32)
        # convert acc to float for momentum update
        acc_f = [a.astype(np.float32) for a in acc]
        # initialize momentum if needed
        if self.momentum_buffers is None:
            self.momentum_buffers = [np.zeros_like(w, dtype=np.float32) for w in self.global_weights]
        # update momentum buffers
        for i in range(layer_count):
            self.momentum_buffers[i] = self.momentum_coeff * self.momentum_buffers[i] + acc_f[i]
        # compute sign of momentum buffer (where zero -> 0)
        agg_sign = [np.sign(mb).astype(np.int8) for mb in self.momentum_buffers]
        # apply update: w <- w - lr * sign(mb)
        new_weights = []
        for w, s in zip(self.global_weights, agg_sign):
            step = self.lr * s.astype(np.float32)
            new_w = w - step
            new_weights.append(new_w)
        self.global_weights = new_weights
        # broadcast new global to clients
        self.broadcast()


# ------------------ Orchestration ------------------
def run_gra_fl(clients_data: List[Tuple[np.ndarray, np.ndarray]], X_test: np.ndarray, y_test: np.ndarray):
    # global scaling: fit StandardScaler on all features
    all_X = np.vstack([X for X, _ in clients_data] + ([X_test] if X_test.shape[0] > 0 else []))
    scaler = StandardScaler()
    scaler.fit(all_X)
    clients_objs = []
    for i, (Xc, yc) in enumerate(clients_data):
        Xc_s = scaler.transform(Xc).astype(np.float32)
        yc_s = yc.astype(np.float32)
        c = GraClient(i, Xc_s, yc_s)
        clients_objs.append(c)

    server = GraServer(clients_objs, lr=SIGNSGD_LR, momentum=MOMENTUM)
    server.broadcast()

    for r in range(1, GLOBAL_ROUNDS + 1):
        print(f"\n=== Gra-FL Round {r}/{GLOBAL_ROUNDS} ===")
        collected_signs = []
        total_comm_latency = 0.0
        for c in clients_objs:
            if random.random() < FAILURE_RATE:
                print(f"Client {c.id} simulated failure (no contribution).")
                continue
            # client computes avg gradients
            t0 = time.time()
            avg_grads = c.compute_avg_gradients(epochs=LOCAL_EPOCHS, batch_size=LOCAL_BATCH)
            signs = c.compute_signs(avg_grads)
            t1 = time.time()
            if signs is not None:
                collected_signs.append(signs)
                latency = t1 - t0
                total_comm_latency += latency
                print(f" Client {c.id} contributed signs; n_samples={c.n}; latency={latency:.4f}s")
        if collected_signs:
            t0 = time.time()
            server.aggregate_and_update(collected_signs)
            t1 = time.time()
            print(f" Aggregation + update time: {t1 - t0:.4f}s")
        else:
            print(" No contributions this round — skipping update.")

        # evaluate global model
        if X_test.shape[0] > 0 and server.global_weights is not None:
            eval_model = create_regressor(X_test.shape[1])
            eval_model.set_weights(server.global_weights)
            loss = eval_model.evaluate(X_test, y_test, verbose=0)
            if isinstance(loss, list):
                mse = float(loss[0])
            else:
                mse = float(loss)
            print(f" Global test MSE after round {r}: {mse:.6f}")
        else:
            print(" No test evaluation available.")

        if psutil:
            print(f" System CPU%: {psutil.cpu_percent(interval=0.2)} Mem%: {psutil.virtual_memory().percent}")
        print(f" Total communication latency this round: {total_comm_latency:.4f}s")

    print("\nGra-FL (SignSGD) finished.")


# ------------------ Main ------------------
if __name__ == "__main__":
    files = list_csv_files(DATA_DIR)
    clients_data, (X_test, y_test) = build_clients_from_files(files, num_clients=NUM_CLIENTS, input_features=INPUT_FEATURES)
    run_gra_fl(clients_data, X_test, y_test)

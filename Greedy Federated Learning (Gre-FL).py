"""
Greedy Federated Learning (Gre-FL) adapted to Google 2019 Cluster sample.

Design decisions (per user instruction):
 - Clients are built from BATCHES OF JOBS (round-robin assignment of job rows to clients).
 - Task (supervised, per-job): predict a proxy target derived from job numeric fields
   (we use the per-row mean of selected numeric columns as regression target).
 - Each client trains a small MLP locally; server aggregates via FedAvg (weighted by samples).
 - Greedy task scheduler allocates a fixed resource pool each round to clients with
   highest priority (priority = computation_power * avg_local_target).
 - Simulate client failures; measure aggregation latency; optional psutil monitoring.
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

# optional resource monitor
try:
    import psutil
except Exception:
    psutil = None

# ---------------- CONFIG ----------------
DATA_DIR = "./data"              # folder containing Google 2019 Cluster CSV files
NUM_CLIENTS = 10                 # number of federated clients (batches of jobs)
INPUT_FEATURES = 6               # number of numeric columns picked as features per job (capped)
LOCAL_EPOCHS = 3                 # local training epochs per round
BATCH_SIZE = 32
GLOBAL_ROUNDS = 20
LEARNING_RATE = 1e-3
RESOURCE_POOL = 100.0            # total resource units (arbitrary units) per round to allocate
RESOURCE_SCALING = 0.7           # factor used in greedy allocation
FAILURE_RATE = 0.12              # per-client failure/dropout probability per round
TEST_HOLDOUT_RATIO = 0.15        # fraction of files held out as global test set
RANDOM_SEED = 42

# reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ---------------- Data utilities ----------------
def list_csv_files(data_dir: str) -> List[str]:
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}. Please download the Kaggle dataset and put CSVs there.")
    return sorted(files)


def extract_numeric_features_from_df(df: pd.DataFrame, max_cols: int = INPUT_FEATURES) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a dataframe for a job log file, select up to `max_cols` numeric columns and produce:
      X: (n_rows, max_cols) numeric features (missing columns padded with zeros)
      y: (n_rows,) target computed as the row-wise mean of selected numeric features (proxy target)
    Returns arrays; rows with all-zero features (if any) are still included.
    """
    numeric = df.select_dtypes(include=[np.number]).fillna(0.0)
    if numeric.shape[1] == 0:
        # nothing numeric -> return empty
        return np.empty((0, max_cols), dtype=np.float32), np.empty((0,), dtype=np.float32)
    # take first up to max_cols columns
    cols = list(numeric.columns)[:max_cols]
    X = numeric[cols].to_numpy(dtype=np.float32)
    # if fewer columns than max_cols, pad to the right
    if X.shape[1] < max_cols:
        pad = np.zeros((X.shape[0], max_cols - X.shape[1]), dtype=np.float32)
        X = np.hstack([X, pad])
    # proxy target: mean of selected features per row (normalized later globally)
    y = np.mean(X[:, :len(cols)], axis=1).astype(np.float32)
    return X, y


def build_clients_from_files(files: List[str], num_clients: int = NUM_CLIENTS, input_features: int = INPUT_FEATURES
                             ) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Tuple[np.ndarray, np.ndarray]]:
    """
    Read files, convert each file's rows to (X, y), then distribute rows round-robin into `num_clients` batches.
    Also build a global test set from held-out files (TEST_HOLDOUT_RATIO fraction of files).
    Returns: clients_data list of (X_client, y_client) and (X_test, y_test)
    """
    files = files.copy()
    random.shuffle(files)
    n_holdout = max(1, int(len(files) * TEST_HOLDOUT_RATIO))
    holdout_files = files[:n_holdout]
    data_files = files[n_holdout:]

    # build global test set from holdout files
    X_test_list, y_test_list = [], []
    for f in holdout_files:
        try:
            df = pd.read_csv(f, low_memory=True)
        except Exception:
            continue
        Xf, yf = extract_numeric_features_from_df(df, max_cols=input_features)
        if Xf.shape[0] > 0:
            X_test_list.append(Xf); y_test_list.append(yf)
    if X_test_list:
        X_test = np.vstack(X_test_list); y_test = np.concatenate(y_test_list)
    else:
        X_test = np.empty((0, input_features), dtype=np.float32); y_test = np.empty((0,), dtype=np.float32)

    # pool all rows from data_files
    pool_X = []
    pool_y = []
    for f in data_files:
        try:
            df = pd.read_csv(f, low_memory=True)
        except Exception:
            continue
        Xf, yf = extract_numeric_features_from_df(df, max_cols=input_features)
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
    clients_chunks = [[] for _ in range(num_clients)]
    clients_chunks_y = [[] for _ in range(num_clients)]
    for i in range(pool_X.shape[0]):
        cid = i % num_clients
        clients_chunks[cid].append(pool_X[i])
        clients_chunks_y[cid].append(pool_y[i])

    clients_data = []
    for cid in range(num_clients):
        Xc = np.vstack(clients_chunks[cid]).astype(np.float32) if clients_chunks[cid] else np.empty((0, input_features), dtype=np.float32)
        yc = np.array(clients_chunks_y[cid], dtype=np.float32) if clients_chunks_y[cid] else np.empty((0,), dtype=np.float32)
        # ensure small clients are padded with tiny random data to avoid empty datasets
        if Xc.shape[0] < 10:
            # create small synthetic rows (rare)
            extra = np.random.rand(10, input_features).astype(np.float32) * 1e-3
            Xc = np.vstack([Xc, extra]) if Xc.size else extra
            yc = np.concatenate([yc, np.zeros(10, dtype=np.float32)]) if yc.size else np.zeros(10, dtype=np.float32)
        clients_data.append((Xc, yc))
    return clients_data, (X_test, y_test)


# ---------------- Model (local client) ----------------
def create_regressor(input_dim: int):
    model = models.Sequential([
        layers.InputLayer(input_shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss=losses.MeanSquaredError(),
                  metrics=[metrics.MeanSquaredError()])
    return model


# ---------------- Client & Scheduler & Server ----------------
class GreClient:
    def __init__(self, cid: int, X: np.ndarray, y: np.ndarray):
        self.id = cid
        self.X = X
        self.y = y
        self.n = X.shape[0]
        self.model = create_regressor(X.shape[1])
        # derive computation_power from local data: lower mean target -> lower load -> higher compute power
        # computation_power in (0, 1]; we invert mean(y) to produce it
        mean_y = float(np.mean(y)) if y.size else 0.0
        self.comp_power = 1.0 / (1.0 + mean_y)  # deterministic and reproducible
        # local scaler for stability (we will also apply global scaler before training)
        self.scaler = None

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def local_train(self, epochs=LOCAL_EPOCHS, batch_size=BATCH_SIZE, verbose=0):
        if self.X.shape[0] == 0:
            return self.get_weights()
        # optional resource monitoring
        if psutil:
            cpu0 = psutil.cpu_percent(interval=None)
            mem0 = psutil.virtual_memory().percent
        # perform local training
        self.model.fit(self.X, self.y, epochs=epochs, batch_size=batch_size, verbose=verbose)
        if psutil:
            cpu1 = psutil.cpu_percent(interval=None)
            mem1 = psutil.virtual_memory().percent
            print(f"Client {self.id} - CPU Δ: {cpu1 - cpu0:.2f}%, MEM Δ: {mem1 - mem0:.2f}%")
        return self.get_weights()


class GreedyTaskScheduler:
    def __init__(self, clients: List[GreClient], resource_pool: float = RESOURCE_POOL, scaling: float = RESOURCE_SCALING):
        self.clients = clients
        self.resource_pool = resource_pool
        self.scaling = scaling

    def allocate(self) -> dict:
        """
        Greedy allocation by priority = comp_power * avg_local_target.
        Higher priority clients receive resource first until pool exhausted.
        Returns mapping client_id -> allocated_resource.
        """
        pool = self.resource_pool
        # compute avg local target per client
        priorities = []
        for c in self.clients:
            avg_t = float(np.mean(c.y)) if c.y.size else 0.0
            pri = c.comp_power * avg_t
            priorities.append((c.id, pri))
        # sort descending by priority
        priorities.sort(key=lambda x: x[1], reverse=True)
        alloc = {c.id: 0.0 for c in self.clients}
        for cid, pri in priorities:
            if pool <= 0:
                break
            c = next(filter(lambda x: x.id == cid, self.clients))
            request = c.comp_power * self.scaling * 10.0  # scale factor to map comp_power->resource units
            assigned = min(pool, request)
            alloc[cid] = assigned
            pool -= assigned
        return alloc


class GreServer:
    def __init__(self, clients: List[GreClient]):
        self.clients = clients
        # initialize global weights as the average of client initial weights
        init_weights = [c.get_weights() for c in clients if c.get_weights()]
        self.global_weights = init_weights[0] if init_weights else None
        if init_weights and len(init_weights) > 1:
            self.global_weights = self.fedavg(init_weights)

    def fedavg(self, weight_lists: List[List[np.ndarray]], sample_counts: List[int] = None) -> List[np.ndarray]:
        """
        Weighted FedAvg by sample_counts if provided, else simple average.
        weight_lists: list of model.get_weights()
        sample_counts: list of ints (same length), optional
        """
        if not weight_lists:
            return None
        if sample_counts is None:
            # simple average
            avg = []
            n_models = len(weight_lists)
            for layer_idx in range(len(weight_lists[0])):
                layer_stack = np.stack([w[layer_idx] for w in weight_lists], axis=0)
                avg.append(np.mean(layer_stack, axis=0))
            return avg
        else:
            total = float(sum(sample_counts))
            avg = []
            for layer_idx in range(len(weight_lists[0])):
                acc = np.zeros_like(weight_lists[0][layer_idx])
                for w, n in zip(weight_lists, sample_counts):
                    acc += w[layer_idx] * (n / total)
                avg.append(acc)
            return avg

    def update_global(self, collected_weights: List[List[np.ndarray]], sample_counts: List[int]):
        start = time.time()
        self.global_weights = self.fedavg(collected_weights, sample_counts)
        # push to clients
        for c in self.clients:
            c.set_weights(self.global_weights)
        end = time.time()
        latency = end - start
        print(f"Aggregation latency: {latency:.4f} s")


# ---------------- Orchestration ----------------
def run_greedy_fl(clients_data: List[Tuple[np.ndarray, np.ndarray]], X_test: np.ndarray, y_test: np.ndarray):
    # global standardization: fit scaler on all client data + test
    all_X = np.vstack([X for X, _ in clients_data] + ([X_test] if X_test.shape[0] > 0 else []))
    scaler = StandardScaler()
    scaler.fit(all_X)
    clients_objs = []
    for i, (Xc, yc) in enumerate(clients_data):
        Xc_s = scaler.transform(Xc).astype(np.float32)
        yc_s = yc.astype(np.float32)
        cl = GreClient(i, Xc_s, yc_s)
        clients_objs.append(cl)

    server = GreServer(clients_objs)
    scheduler = GreedyTaskScheduler(clients_objs)

    # initial broadcast
    if server.global_weights:
        for c in clients_objs:
            c.set_weights(server.global_weights)

    # main FL rounds
    for rnd in range(1, GLOBAL_ROUNDS + 1):
        print(f"\n=== Gre-FL Round {rnd}/{GLOBAL_ROUNDS} ===")
        # Greedy allocation
        allocation = scheduler.allocate()
        print("Resource allocation (top entries):", dict(list(allocation.items())[:6]))

        # clients train (simulate failures)
        collected_weights = []
        sample_counts = []
        for c in clients_objs:
            if random.random() < FAILURE_RATE:
                print(f"Client {c.id} failed to send update (simulated).")
                continue
            # optionally log allocated resource (not used directly in training here)
            allocated = allocation.get(c.id, 0.0)
            # local training
            w = c.local_train()
            collected_weights.append(w)
            sample_counts.append(c.n)
            print(f" Client {c.id} trained on {c.n} samples; comp_power={c.comp_power:.3f}; allocated={allocated:.2f}")

        # aggregate if any updates received
        if collected_weights:
            server.update_global(collected_weights, sample_counts)
        else:
            print("No client updates this round — skipping aggregation.")

        # evaluate global model on test (if available)
        if X_test.shape[0] > 0 and server.global_weights is not None:
            # set a temporary model to evaluate
            eval_model = create_regressor(X_test.shape[1])
            eval_model.set_weights(server.global_weights)
            loss = eval_model.evaluate(X_test, y_test, verbose=0)
            if isinstance(loss, list): mse = float(loss[0])
            else: mse = float(loss)
            print(f" Global test MSE after round {rnd}: {mse:.6f}")
        else:
            print("No global test evaluation (no test samples or no global weights).")

    print("\nGreedy Federated Learning finished.")


# ---------------- Main ----------------
if __name__ == "__main__":
    files = list_csv_files(DATA_DIR)
    clients_data, (X_test, y_test) = build_clients_from_files(files, num_clients=NUM_CLIENTS, input_features=INPUT_FEATURES)
    # optionally global standardization for y as well: y already 0..1 per-file mean; if desired, keep as-is
    run_greedy_fl(clients_data, X_test, y_test)

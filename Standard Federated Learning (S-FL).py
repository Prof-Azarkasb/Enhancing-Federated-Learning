"""
Standard Federated Learning (S-FL) on Google 2019 Cluster samples.
- Task: predict next-step resource usage (regression) using sliding-window samples
- Local models: small MLP (no pretrained libs)
- Server: FedAvg weighted by client sample counts
Run:
  python sfl_google_cluster.py
"""

import os
import glob
import random
import math
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, metrics

# ---------------- Config ----------------
DATA_DIR = "./data"            # place Google cluster CSV files here
NUM_CLIENTS = 10               # number of federated clients to simulate
INPUT_LEN = 32                 # sliding-window length (features per sample)
SAMPLE_STEP = 1                # sliding step
LOCAL_EPOCHS = 3               # local epochs per round
BATCH_SIZE = 32
GLOBAL_ROUNDS = 15
LEARNING_RATE = 1e-3
FAILURE_RATE = 0.10            # probability a client fails to send update in a round
TEST_FILE_HOLDOUT = 0.15       # fraction of files held out as global test set
RANDOM_SEED = 42

# reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ---------------- Data utilities ----------------
def find_csv_files(data_dir: str) -> List[str]:
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {data_dir}. Download dataset and place CSVs there.")
    return sorted(files)


def file_series_to_windows(path: str, input_len: int = INPUT_LEN, step: int = SAMPLE_STEP
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read CSV, choose first numeric column, normalize, build sliding-window samples.
    Returns X (N_samples x input_len), y (N_samples,) where y is next-step value (regression).
    """
    df = pd.read_csv(path, low_memory=True)
    # select numeric columns
    numeric = df.select_dtypes(include=[np.number]).fillna(0.0)
    if numeric.shape[1] == 0:
        return np.empty((0, input_len), dtype=np.float32), np.empty((0,), dtype=np.float32)
    col = numeric.columns[0]
    series = numeric[col].values.astype(np.float32)
    if series.size <= input_len:
        return np.empty((0, input_len), dtype=np.float32), np.empty((0,), dtype=np.float32)

    # normalize series (per-file)
    if series.max() > series.min():
        series = (series - series.min()) / (series.max() - series.min())
    else:
        series = np.zeros_like(series)

    X_list = []
    y_list = []
    for i in range(0, len(series) - input_len, step):
        window = series[i:i + input_len]
        target = series[i + input_len]  # next step
        X_list.append(window)
        y_list.append(target)
    if not X_list:
        return np.empty((0, input_len), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return np.stack(X_list, axis=0).astype(np.float32), np.array(y_list, dtype=np.float32)


def build_clients_from_files(files: List[str], num_clients: int = NUM_CLIENTS,
                             input_len: int = INPUT_LEN) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Tuple[np.ndarray, np.ndarray]]:
    """
    Convert files to per-client datasets using sliding-window.
    Splits some files as global test holdout.
    Returns:
      clients_data: list of (X_client, y_client)
      (X_test, y_test): global test set (concatenated windows from holdout files)
    """
    # shuffle files and select test holdout files
    files = files.copy()
    random.shuffle(files)
    num_holdout = max(1, int(len(files) * TEST_FILE_HOLDOUT))
    holdout_files = files[:num_holdout]
    pool_files = files[num_holdout:]

    # build test set
    X_test_list, y_test_list = [], []
    for f in holdout_files:
        Xf, yf = file_series_to_windows(f, input_len)
        if Xf.shape[0] > 0:
            X_test_list.append(Xf)
            y_test_list.append(yf)
    if X_test_list:
        X_test = np.vstack(X_test_list)
        y_test = np.concatenate(y_test_list)
    else:
        X_test = np.empty((0, input_len), dtype=np.float32)
        y_test = np.empty((0,), dtype=np.float32)

    # for pool files create per-file windows and group by clients
    per_file_windows = []
    for f in pool_files:
        Xf, yf = file_series_to_windows(f, input_len)
        if Xf.shape[0] > 0:
            per_file_windows.append((Xf, yf))

    if not per_file_windows:
        raise RuntimeError("No usable windows extracted from data files. Check CSV contents.")

    # distribute files (and their windows) to clients round-robin
    clients_data = [ ([],[]) for _ in range(num_clients) ]  # (list_X_chunks, list_y_chunks)
    for idx, (Xf, yf) in enumerate(per_file_windows):
        client_idx = idx % num_clients
        clients_data[client_idx][0].append(Xf)
        clients_data[client_idx][1].append(yf)

    # concatenate chunks to produce final arrays
    final_clients = []
    for chunks_x, chunks_y in clients_data:
        if chunks_x:
            Xc = np.vstack(chunks_x)
            yc = np.concatenate(chunks_y)
        else:
            # if a client got no data, create tiny random placeholder (should not happen if enough files)
            Xc = np.random.rand(20, input_len).astype(np.float32)
            yc = np.random.rand(20).astype(np.float32)
        final_clients.append((Xc, yc))
    return final_clients, (X_test, y_test)


# ---------------- Model and FL classes ----------------
def create_regressor(input_len: int = INPUT_LEN) -> tf.keras.Model:
    model = models.Sequential([
        layers.InputLayer(input_shape=(input_len,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer=optimizers.Adam(LEARNING_RATE),
                  loss=losses.MeanSquaredError(),
                  metrics=[metrics.MeanSquaredError()])
    return model


class FLClient:
    def __init__(self, cid: int, X: np.ndarray, y: np.ndarray):
        self.id = cid
        self.X = X
        self.y = y
        self.n = X.shape[0]
        self.model = create_regressor(INPUT_LEN)

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def train(self, epochs=LOCAL_EPOCHS, batch_size=BATCH_SIZE, verbose=0):
        self.model.fit(self.X, self.y, epochs=epochs, batch_size=batch_size, verbose=verbose)
        return self.get_weights()


class FLServer:
    def __init__(self, clients: List[FLClient]):
        self.clients = clients
        self.global_model = create_regressor(INPUT_LEN)
        self.global_weights = self.global_model.get_weights()

    def broadcast(self):
        for c in self.clients:
            c.set_weights(self.global_weights)

    def aggregate_fedavg(self, client_weights: List[Tuple[List[np.ndarray], int]]):
        """
        client_weights: list of tuples (weights_list, num_samples)
        Weighted FedAvg by num_samples.
        """
        total_samples = sum(n for _, n in client_weights)
        if total_samples == 0:
            return
        # initialize accumulator
        avg_w = [np.zeros_like(w) for w in client_weights[0][0]]
        for weights, n in client_weights:
            for i in range(len(avg_w)):
                avg_w[i] += weights[i] * (n / total_samples)
        self.global_weights = avg_w
        self.global_model.set_weights(self.global_weights)

    def evaluate_global(self, X_test: np.ndarray, y_test: np.ndarray):
        if X_test.shape[0] == 0:
            return {'mse': None, 'samples': 0}
        loss = self.global_model.evaluate(X_test, y_test, verbose=0)
        # Keras returns [loss, mse] for compiled metrics; we used MSE as loss and metric identical -> return loss
        if isinstance(loss, list):
            mse_val = loss[0]
        else:
            mse_val = float(loss)
        return {'mse': float(mse_val), 'samples': X_test.shape[0]}


# ---------------- Federated training orchestration ----------------
def run_federated_learning(clients_data: List[Tuple[np.ndarray, np.ndarray]],
                           X_test: np.ndarray, y_test: np.ndarray,
                           rounds: int = GLOBAL_ROUNDS,
                           failure_rate: float = FAILURE_RATE):
    # instantiate clients
    clients = [FLClient(i, X, y) for i, (X, y) in enumerate(clients_data)]
    server = FLServer(clients)
    print(f"Initialized {len(clients)} clients. Test set: {X_test.shape[0]} samples.")

    # initial broadcast
    server.broadcast()

    for rnd in range(1, rounds + 1):
        print(f"\n=== Global Round {rnd}/{rounds} ===")
        client_updates = []
        # each client optionally trains and sends update
        for c in clients:
            if random.random() < failure_rate:
                print(f" Client {c.id} skipped (simulated failure).")
                continue
            # set global weights (in case some clients lagged)
            c.set_weights(server.global_weights)
            weights = c.train()
            client_updates.append((weights, c.n))
            print(f" Client {c.id} trained on {c.n} samples.")

        # aggregate
        if client_updates:
            server.aggregate_fedavg(client_updates)
            # broadcast updated global to all clients
            server.broadcast()
        else:
            print(" No client updates this round.")

        # evaluate global model on held-out test set
        eval_info = server.evaluate_global(X_test, y_test)
        mse = eval_info['mse']
        if mse is not None:
            print(f" Global model MSE on test set: {mse:.6f}")
        else:
            print(" No test samples available for evaluation.")

    print("\nFederated training complete.")
    return server, clients


# ---------------- Main ----------------
if __name__ == "__main__":
    all_files = find_csv_files(DATA_DIR)
    clients_data, (X_test, y_test) = build_clients_from_files(all_files, num_clients=NUM_CLIENTS, input_len=INPUT_LEN)
    # Optionally normalize globally (we already normalized per-file; here we standardize global features)
    if X_test.shape[0] > 0:
        scaler = StandardScaler()
        # fit scaler on clients' data combined to standardize globally
        all_X = np.vstack([X for X, _ in clients_data] + ([X_test] if X_test.shape[0] > 0 else []))
        scaler.fit(all_X)
        clients_data = [(scaler.transform(X).astype(np.float32), y.astype(np.float32)) for X, y in clients_data]
        X_test = scaler.transform(X_test).astype(np.float32)
    else:
        # still ensure types
        clients_data = [(X.astype(np.float32), y.astype(np.float32)) for X, y in clients_data]

    # Run federated learning
    server, clients = run_federated_learning(clients_data, X_test, y_test, rounds=GLOBAL_ROUNDS, failure_rate=FAILURE_RATE)

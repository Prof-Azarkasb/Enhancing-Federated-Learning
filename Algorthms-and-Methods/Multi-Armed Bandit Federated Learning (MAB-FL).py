"""
MAB-FL (updated for Manjushree et al. 2025 ideas: Efficient model choice + blockchain-enabled trust + trust-aware MAB scheduling)
Uses Google 2019 Cluster sample CSVs placed in ./data/.
"""

import os
import glob
import time
import hashlib
import random
from collections import deque

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

# optional: psutil to measure CPU/memory (install if desired)
try:
    import psutil
except Exception:
    psutil = None

# ---------------------------
# Reproducibility & config
# ---------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_DIR = "./data"         # folder with Google 2019 Cluster CSV files
NUM_CLIENTS = 10            # number of federated clients (will take up to this many CSVs)
CLIENT_SAMPLES = 600        # samples per client (after tiling/padding)
INPUT_DIM = 64              # feature vector length per client
NUM_ROUNDS = 80             # federated rounds
LOCAL_EPOCHS = 2
BATCH_SIZE = 32
LEARNING_RATE = 0.005

MAB_EXPLORATION = 1.5       # UCB exploration constant c
COMM_COST_WEIGHT = 0.05     # lambda: penalty in reward for comm cost (MB)
LEDGER_VERIFICATION_PROB = 0.98   # simulated verification success probability
FAILURE_RATE = 0.12         # client fails to send update with this probability

# ---------------------------
# Utilities: load cluster data
# ---------------------------
def load_cluster_clients(data_dir=DATA_DIR, num_clients=NUM_CLIENTS, input_dim=INPUT_DIM, samples_per_client=CLIENT_SAMPLES):
    """
    Build up to `num_clients` client datasets from CSV files in data_dir.
    Each CSV file -> one pseudo-client; numeric first column is used as time series and converted into fixed-length vectors.
    Returns: list of tuples (X_client, y_client) and (X_test, y_test)
    y is regression target: mean usage (float)
    """
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}. Download the Kaggle dataset and place CSVs into this folder.")
    csv_files = csv_files[:num_clients*2]  # take more in case some files unreadable

    vectors = []
    for path in csv_files:
        try:
            df = pd.read_csv(path, low_memory=True)
        except Exception:
            continue
        numeric = df.select_dtypes(include=[np.number]).fillna(0)
        if numeric.shape[1] == 0:
            continue
        col = numeric.columns[0]
        arr = numeric[col].values.astype(np.float32)
        if arr.size == 0:
            continue
        # normalize time series locally
        if arr.max() > arr.min():
            arr = (arr - arr.min()) / (arr.max() - arr.min())
        else:
            arr = np.zeros_like(arr)
        # fixed-length vector
        if arr.size >= input_dim:
            vec = arr[:input_dim]
        else:
            reps = int(np.ceil(input_dim / max(1, arr.size)))
            vec = np.tile(arr, reps)[:input_dim]
        vectors.append(vec)

    if len(vectors) < num_clients + 50:
        # pad with random vectors to ensure enough samples
        while len(vectors) < num_clients + 100:
            vectors.append(np.random.rand(input_dim).astype(np.float32))

    X_all = np.stack(vectors, axis=0)
    y_all = X_all.mean(axis=1)  # regression target in [0,1]

    # shuffle and split into test + client pools
    idx = np.arange(len(X_all))
    np.random.shuffle(idx)
    test_size = max(200, int(0.2 * len(X_all)))
    test_idx = idx[:test_size]
    pool_idx = idx[test_size:]

    X_test = X_all[test_idx]
    y_test = y_all[test_idx]

    # create client datasets by partitioning the pool
    pool = X_all[pool_idx]
    pool_y = y_all[pool_idx]
    per_client = max(1, pool.shape[0] // num_clients)
    clients = []
    for i in range(num_clients):
        start = i * per_client
        end = start + per_client
        x = pool[start:end]
        y = pool_y[start:end]
        # augment to samples_per_client by tiling if needed
        if x.shape[0] < samples_per_client:
            reps = int(np.ceil(samples_per_client / max(1, x.shape[0])))
            x = np.tile(x, (reps, 1))[:samples_per_client]
            y = np.tile(y, reps)[:samples_per_client]
        clients.append((x.astype(np.float32), y.astype(np.float32)))
    return clients, (X_test.astype(np.float32), y_test.astype(np.float32))

# ---------------------------
# Simple MLP local model factory
# ---------------------------
def create_mlp(input_dim=INPUT_DIM):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE), loss='mse')
    return model

# ---------------------------
# Lightweight ledger (blockchain-style)
# ---------------------------
class SimpleLedger:
    def __init__(self):
        self.chain = []  # list of entries: dict(index, ts, client_id, contrib_hash, prev_hash)
    def latest_hash(self):
        return self.chain[-1]['contrib_hash'] if self.chain else "0"*64
    def append(self, client_id, contrib_bytes):
        prev = self.latest_hash()
        ts = time.time()
        h = hashlib.sha256()
        h.update(prev.encode('utf-8'))
        h.update(str(client_id).encode('utf-8'))
        h.update(str(ts).encode('utf-8'))
        h.update(contrib_bytes)
        ch = h.hexdigest()
        entry = {'index': len(self.chain), 'ts': ts, 'client_id': client_id, 'contrib_hash': ch, 'prev_hash': prev}
        self.chain.append(entry)
        return ch
    def validate_chain(self):
        for i in range(1, len(self.chain)):
            if self.chain[i]['prev_hash'] != self.chain[i-1]['contrib_hash']:
                return False
        return True

# ---------------------------
# Client object
# ---------------------------
class Client:
    def __init__(self, cid, x, y):
        self.id = cid
        self.x = x
        self.y = y
        self.n = x.shape[0]
        self.model = create_mlp()
        # simulate heterogeneous compute speed (affects training time)
        self.comp_power = random.uniform(0.6, 1.6)
        # initial trust (server-side maintained too)
        self.trust = 0.5

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def local_train(self, epochs=LOCAL_EPOCHS, batch_size=BATCH_SIZE):
        # measure (simulated) compute time depending on comp_power and data size
        t0 = time.time()
        self.model.fit(self.x, self.y, epochs=epochs, batch_size=batch_size, verbose=0)
        # simulate processing delay
        proc_time = (epochs * self.n / 1000.0) * self.comp_power
        time.sleep(min(proc_time, 0.5))  # keep short
        t1 = time.time()
        # evaluate local performance (mean squared error)
        preds = self.model.predict(self.x, verbose=0).reshape(-1)
        mse = float(np.mean((preds - self.y)**2))
        return mse, t1 - t0

    def serialize_weights(self):
        # flatten weights to bytes for hashing
        weights = self.get_weights()
        flat = np.concatenate([w.ravel() for w in weights]).astype(np.float32)
        return flat.tobytes()

# ---------------------------
# Server with trust and aggregation
# ---------------------------
class FederatedServer:
    def __init__(self, clients):
        self.clients = clients
        self.global_model = create_mlp()
        self.global_weights = self.global_model.get_weights()
        self.ledger = SimpleLedger()
        # estimate communication cost per client (MB) from model size
        self.model_bytes = len(self._weights_to_bytes(self.global_weights))
        self.comm_costs = {c.id: (self.model_bytes / (1024.0*1024.0)) for c in clients}
        # server-side trust map (initialized to client trust)
        self.trust = {c.id: c.trust for c in clients}
        self.trust_history = {c.id: deque(maxlen=50) for c in clients}

    def _weights_to_bytes(self, weights):
        flat = np.concatenate([w.ravel() for w in weights]).astype(np.float32)
        return flat.tobytes()

    def broadcast(self):
        for c in self.clients:
            c.set_weights(self.global_weights)

    def aggregate_single(self, client_weights, weight=1.0):
        # here client_weights is a single client's weights (list of arrays)
        # perform simple additive update: w_new = w_old + alpha * (w_client - w_old)
        # For simplicity use: global = client_weights (simulates Fed when single client)
        self.global_weights = client_weights
        self.global_model.set_weights(self.global_weights)

    def verify_and_append(self, client_id, contrib_bytes):
        # simulate verification: with high probability valid; otherwise invalid
        valid = random.random() < LEDGER_VERIFICATION_PROB
        ch = self.ledger.append(client_id, contrib_bytes)
        return valid, ch

    def update_trust(self, client_id, valid, observed_perf):
        old = self.trust.get(client_id, 0.5)
        if valid:
            # reward: higher performance means lower loss -> translate to positive
            # we convert mse to performance: perf = 1 / (1 + mse)
            perf = 1.0 / (1.0 + observed_perf)
            new = 0.85 * old + 0.15 * perf
        else:
            new = 0.7 * old - 0.2 * (1.0 / (1.0 + observed_perf))
        new = max(0.0, min(1.0, new))
        smoothed = 0.98 * old + 0.02 * new
        self.trust[client_id] = smoothed
        self.trust_history[client_id].append(smoothed)

    def get_adjusted_reward(self, client_id, perf_mse):
        # perf scalar: higher is better. convert mse->perf in (0,1]
        perf = 1.0 / (1.0 + perf_mse)
        trust = self.trust.get(client_id, 0.5)
        comm = self.comm_costs.get(client_id, 0.0)
        adjusted = perf * trust - COMM_COST_WEIGHT * comm
        return adjusted

# ---------------------------
# UCB scheduler with trust-aware reward
# ---------------------------
class MABScheduler:
    def __init__(self, client_ids, c=MAB_EXPLORATION):
        self.K = len(client_ids)
        self.ids = list(client_ids)
        self.counts = {cid: 0 for cid in self.ids}
        self.values = {cid: 0.0 for cid in self.ids}  # cumulative adjusted rewards
        self.t = 1
        self.c = c

    def select(self):
        # force one-play for unplayed arms
        for cid in self.ids:
            if self.counts[cid] == 0:
                return cid
        scores = {}
        for cid in self.ids:
            avg = self.values[cid] / max(1, self.counts[cid])
            bonus = self.c * np.sqrt(np.log(self.t) / max(1, self.counts[cid]))
            scores[cid] = avg + bonus
        # pick id with highest score
        chosen = max(scores.items(), key=lambda x: x[1])[0]
        return chosen

    def update(self, cid, adjusted_reward):
        self.counts[cid] += 1
        self.values[cid] += adjusted_reward
        self.t += 1

# ---------------------------
# Orchestrator
# ---------------------------
class MABFederatedLearning:
    def __init__(self, num_clients=NUM_CLIENTS):
        clients_data, test = load_cluster_clients()
        self.test_X, self.test_y = test
        # create Client objects
        self.clients = []
        for i, (x, y) in enumerate(clients_data[:num_clients]):
            self.clients.append(Client(i, x, y))
        self.server = FederatedServer(self.clients)
        self.scheduler = MABScheduler([c.id for c in self.clients])

    def run(self, rounds=NUM_ROUNDS):
        # initial broadcast
        self.server.broadcast()

        for r in range(1, rounds + 1):
            print(f"\n--- Round {r}/{rounds} ---")
            selected_id = self.scheduler.select()
            client = next(c for c in self.clients if c.id == selected_id)
            print(f" Scheduler selected client {client.id}")

            # simulate possible network failure
            if random.random() < FAILURE_RATE:
                print(f" Client {client.id} failed to send update (simulated).")
                # penalize slightly in scheduler
                self.scheduler.update(client.id, -0.1)
                continue

            # client trains locally
            mse, local_time = client.local_train()
            print(f" Client {client.id} local MSE: {mse:.6f}, local_time(s): {local_time:.3f}")

            # serialize contribution and verify (append to ledger)
            contrib_bytes = client.serialize_weights()
            valid, ch = self.server.verify_and_append(client.id, contrib_bytes)
            if not valid:
                print(f" Contribution from client {client.id} failed verification (simulated).")
            # update server trust
            self.server.update_trust(client.id, valid, mse)

            # server aggregates (single-client update)
            # here we simply replace global weights with client's weights to simulate single-client contribution
            client_weights = client.get_weights()
            start_time = time.time()
            self.server.aggregate_single(client_weights)
            end_time = time.time()
            latency = end_time - start_time
            print(f" Aggregation latency: {latency:.4f}s")

            # compute adjusted reward and update scheduler
            adj = self.server.get_adjusted_reward(client.id, mse)
            print(f" Adjusted reward (perf*trust - comm_penalty): {adj:.6f}")
            self.scheduler.update(client.id, adj)

            # Evaluate global model periodically
            if r % 5 == 0 or r == 1:
                loss = self.evaluate_global()
                print(f" Global MSE on test set: {loss:.6f}")
        # end rounds
        print("\nSimulation finished. Ledger valid:", self.server.ledger.validate_chain())

    def evaluate_global(self):
        preds = self.server.global_model.predict(self.test_X, verbose=0).reshape(-1)
        mse = float(np.mean((preds - self.test_y)**2))
        return mse

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    mb = MABFederatedLearning(num_clients=NUM_CLIENTS)
    mb.run(rounds=NUM_ROUNDS)

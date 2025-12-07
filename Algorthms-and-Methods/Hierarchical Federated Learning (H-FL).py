"""
Hierarchical Federated Learning (H-FL) implementation using Google 2019 Cluster data.
  python hfl_cluster.py
"""

import os
import glob
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, metrics
import time

# === Configuration Parameters ===
DATA_DIR = "./data"
INPUT_DIM = 64            # Length of input feature vector per client
NUM_EDGE_SERVERS = 2      # Number of edge servers
CLIENTS_PER_EDGE = 5      # Clients assigned per edge server
LOCAL_EPOCHS = 2
BATCH_SIZE = 32
GLOBAL_ROUNDS = 15
AGG_EDGE_TO_CLOUD_INTERVAL = 5  # aggregate to cloud every 5 edge rounds
LEARNING_RATE = 0.001
FAILURE_RATE = 0.1         # Probability that a client fails to send update in a round
TEST_SPLIT_RATIO = 0.2     # For global test set

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# === Utility: Load and preprocess Google Cluster data ===
def load_cluster_clients(data_dir=DATA_DIR, input_dim=INPUT_DIM,
                         num_clients_total=None):
    """
    Loads CSV files in data_dir, extracts numeric columns, builds fixed-length vectors per file (pseudo-client).
    Returns:
        clients_data: list of (X, y) for each client
        (X_test, y_test): global test set
    """
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}. Please place Google Cluster CSVs there.")

    # Optionally limit number of clients
    if num_clients_total is not None:
        csv_files = csv_files[:num_clients_total]

    feature_list = []
    for p in csv_files:
        try:
            df = pd.read_csv(p, low_memory=True)
        except Exception as e:
            print(f"Warning: could not read {p}: {e}")
            continue
        # pick numeric columns
        num_df = df.select_dtypes(include=[np.number]).fillna(0)
        if num_df.shape[1] == 0:
            continue
        # pick first numeric column as representative
        col = num_df.columns[0]
        arr = num_df[col].values.astype(np.float32)
        if arr.size == 0:
            continue
        # normalize
        if arr.max() > arr.min():
            arr = (arr - arr.min()) / (arr.max() - arr.min())
        else:
            arr = np.zeros_like(arr)
        # make fixed-length vector
        if arr.size >= input_dim:
            vec = arr[:input_dim]
        else:
            reps = int(np.ceil(input_dim / max(1, arr.size)))
            vec = np.tile(arr, reps)[:input_dim]
        feature_list.append(vec)

    if len(feature_list) < (NUM_EDGE_SERVERS * CLIENTS_PER_EDGE + 1):
        raise RuntimeError("Not enough valid client data derived from CSVs. Need at least "
                           f"{NUM_EDGE_SERVERS * CLIENTS_PER_EDGE + 1} valid files.")

    X_all = np.stack(feature_list, axis=0)
    # Produce synthetic labels: e.g. use mean of vector (resource usage) bucketed into classes
    # Here we do regression: target = mean value (in [0,1])
    y_all = X_all.mean(axis=1)

    # Split into global test set and remaining
    idx = np.arange(len(X_all))
    np.random.shuffle(idx)
    test_size = int(TEST_SPLIT_RATIO * len(X_all))
    test_idx = idx[:test_size]
    rest_idx = idx[test_size:]

    X_test = X_all[test_idx]
    y_test = y_all[test_idx]

    X_rest = X_all[rest_idx]
    y_rest = y_all[rest_idx]

    # Build clients
    clients_data = []
    per_client = (NUM_EDGE_SERVERS * CLIENTS_PER_EDGE)
    if X_rest.shape[0] < per_client:
        raise RuntimeError("Not enough remaining data to fill all clients.")
    per = X_rest.shape[0] // per_client
    for i in range(per_client):
        start = i * per
        end = start + per
        clients_data.append((X_rest[start:end], y_rest[start:end]))

    return clients_data, (X_test, y_test)

# === Define ML model ===
def create_model(input_dim):
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

# === H-FL components ===
class Client:
    def __init__(self, cid, X, y):
        self.id = cid
        self.X = X
        self.y = y
        self.n = X.shape[0]
        self.model = create_model(INPUT_DIM)

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def train_local(self, epochs=LOCAL_EPOCHS, batch_size=BATCH_SIZE):
        self.model.fit(self.X, self.y, epochs=epochs, batch_size=batch_size, verbose=0)
        return self.get_weights()

class EdgeServer:
    def __init__(self, edge_id, clients: list):
        self.id = edge_id
        self.clients = clients
        self.model = create_model(INPUT_DIM)

    def aggregate_clients(self):
        # simple average of client updates
        weights_list = [c.get_weights() for c in self.clients]
        new_w = []
        for layer_idx in range(len(weights_list[0])):
            layer_stack = np.stack([w[layer_idx] for w in weights_list], axis=0)
            new_w.append(np.mean(layer_stack, axis=0))
        self.model.set_weights(new_w)

    def distribute_to_clients(self):
        gw = self.model.get_weights()
        for c in self.clients:
            c.set_weights(gw)

class CloudServer:
    def __init__(self, edge_servers: list):
        self.edges = edge_servers
        self.model = create_model(INPUT_DIM)

    def aggregate_edges(self):
        weights_list = [e.model.get_weights() for e in self.edges]
        new_w = []
        for layer_idx in range(len(weights_list[0])):
            layer_stack = np.stack([w[layer_idx] for w in weights_list], axis=0)
            new_w.append(np.mean(layer_stack, axis=0))
        self.model.set_weights(new_w)

    def distribute_to_edges(self):
        gw = self.model.get_weights()
        for e in self.edges:
            e.model.set_weights(gw)

    def evaluate_global(self, X_test, y_test):
        loss = self.model.evaluate(X_test, y_test, verbose=0)
        return loss

# === Main H-FL workflow ===
def run_hfl():
    clients_data, (X_test, y_test) = load_cluster_clients()
    clients = [Client(i, data[0], data[1]) for i, data in enumerate(clients_data)]

    # assign clients to edges
    edges = []
    for ei in range(NUM_EDGE_SERVERS):
        group = clients[ei * CLIENTS_PER_EDGE : (ei+1)*CLIENTS_PER_EDGE]
        edges.append(EdgeServer(ei, group))

    cloud = CloudServer(edges)

    # initial global broadcast
    cloud.distribute_to_edges()
    for e in edges:
        e.distribute_to_clients()

    for rnd in range(1, GLOBAL_ROUNDS + 1):
        print(f"\n--- Global Round {rnd}/{GLOBAL_ROUNDS} ---")

        # Clients train locally
        for e in edges:
            for c in e.clients:
                if random.random() < FAILURE_RATE:
                    # simulate failure / dropout
                    continue
                c.train_local()

        # Edge aggregation
        for e in edges:
            e.aggregate_clients()

        # Edge → Cloud aggregation periodically
        if rnd % AGG_EDGE_TO_CLOUD_INTERVAL == 0:
            cloud.aggregate_edges()
            cloud.distribute_to_edges()
            for e in edges:
                e.distribute_to_clients()

        # Evaluate global model
        mse = cloud.evaluate_global(X_test, y_test)
        print(f"Global model MSE on test set: {mse:.6f}")

if __name__ == "__main__":
    run_hfl()

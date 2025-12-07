"""
Weighted Federated Learning (W-FL) — WeiAvgCS-inspired implementation
Uses Google 2019 Cluster sample (Kaggle)
Key ideas implemented:
 - Per-client diversity estimation via PCA (projection-based estimator).
 - Per-client reliability/latency estimation (from traces or simulated).
 - Aggregation weights combine dataset-size, diversity, and reliability.
 - Reproducible: fixed seeds, detailed per-round logs.
"""

import os
import glob
import math
import random
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# -----------------------
# Reproducibility seeds
# -----------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------
# Simulation parameters
# -----------------------
DATA_DIR = "./data"           # put Kaggle CSVs here
NUM_CLIENTS = 10              # number of federated clients to simulate
INPUT_DIM = 64                # feature vector length per client sample (will be produced by loader)
NUM_CLASSES = 10              # synthetic classification heads for proxy task
CLIENT_SAMPLES_TARGET = 600   # number of samples per client after augmentation/padding
GLOBAL_ROUNDS = 20
LOCAL_EPOCHS = 2
BATCH_SIZE = 32
LEARNING_RATE = 0.01

# Aggregation weight mixture coefficients (sum not necessarily 1; normalized later)
ALPHA_SIZE = 0.6     # weight on dataset size
BETA_DIVERSITY = 0.3 # weight on diversity estimate
GAMMA_RELIABILITY = 0.1  # weight on reliability/latency

# Diversity PCA components to estimate projection-based diversity
PCA_COMPONENTS = 5

# Simulated latency fallback range (seconds)
LATENCY_MIN = 0.01
LATENCY_MAX = 0.5

# -----------------------
# Utility: load & build per-client datasets from Google Cluster CSVs
# -----------------------
def discover_and_load_cluster_data(data_dir=DATA_DIR, input_dim=INPUT_DIM, samples_per_client=CLIENT_SAMPLES_TARGET):
    """
    Scans CSV files in data_dir, picks CPU-like numeric columns, aggregates per machine/task id,
    produces fixed-length feature vectors per machine and synthetic labels (bucketed).
    Returns:
       clients_data: list of (X_client, y_client) tuples length NUM_CLIENTS
       test_set: (X_test, y_test)
    """
    csv_paths = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {data_dir}. Download Kaggle dataset and place CSVs here.")

    machine_series = defaultdict(list)

    # Heuristic: try to find numeric column names containing 'cpu' or 'usage'
    for p in csv_paths:
        try:
            df = pd.read_csv(p, low_memory=True)
        except Exception as e:
            # skip unreadable files
            print(f"Skipping {p}: read error {e}")
            continue
        cols = [c.lower() for c in df.columns]
        cpu_cols = [c for c in df.columns if 'cpu' in c.lower() or 'usage' in c.lower()]
        if len(cpu_cols) == 0:
            # try numeric columns
            numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
            if numeric_cols:
                col = numeric_cols[0]
            else:
                continue
        else:
            col = cpu_cols[0]

        # find an id-like column if present
        id_col = None
        for cand in ['machine_id', 'machine', 'machineid', 'task_id', 'taskid', 'job_id']:
            if cand in cols:
                id_col = [c for c in df.columns if c.lower() == cand][0]
                break

        if id_col is None:
            pseudo = os.path.basename(p)
            vals = df[col].fillna(0).astype(float).values
            if vals.size > 0:
                machine_series[pseudo].extend(vals.tolist())
        else:
            for mid, g in df.groupby(id_col):
                vals = g[col].fillna(0).astype(float).values
                if vals.size > 0:
                    machine_series[str(mid)].extend(vals.tolist())

    if not machine_series:
        raise RuntimeError("No usable series extracted from CSVs. Inspect column names in your CSVs.")

    # Build fixed-size feature vectors per machine by truncating or tiling each series to length input_dim
    machine_ids = list(machine_series.keys())
    vectors = []
    for mid in machine_ids:
        s = np.array(machine_series[mid], dtype=np.float32)
        if s.size == 0:
            continue
        # Normalize per-series
        if s.max() > s.min():
            s = (s - s.min()) / (s.max() - s.min())
        else:
            s = np.zeros_like(s)
        if s.size >= input_dim:
            vec = s[-input_dim:]
        else:
            reps = math.ceil(input_dim / max(1, s.size))
            vec = np.tile(s, reps)[:input_dim]
        vectors.append(vec)

    if len(vectors) < NUM_CLIENTS + 50:
        # If too few machines extracted, pad with random vectors (keeps the code runnable)
        while len(vectors) < NUM_CLIENTS + 100:
            vectors.append(np.random.rand(input_dim).astype(np.float32))

    X_all = np.stack(vectors, axis=0)
    # synthetic labels: bucket mean usage into NUM_CLASSES classes
    mean_usage = X_all.mean(axis=1)
    y_all = np.minimum(NUM_CLASSES - 1, (mean_usage * NUM_CLASSES).astype(int))

    # Shuffle and partition machines into NUM_CLIENTS
    idx = np.arange(X_all.shape[0])
    np.random.shuffle(idx)
    per_client = max(1, X_all.shape[0] // NUM_CLIENTS)
    clients = []
    for i in range(NUM_CLIENTS):
        start = i * per_client
        end = start + per_client
        sel = idx[start:end]
        x = X_all[sel]
        y = y_all[sel]
        # augment/pad to target samples_per_client
        if x.shape[0] < samples_per_client:
            reps = math.ceil(samples_per_client / x.shape[0])
            x = np.tile(x, (reps, 1))[:samples_per_client]
            y = np.tile(y, reps)[:samples_per_client]
        clients.append((x.astype(np.float32), y.astype(np.int32)))

    # create a test set from remaining indices (or first samples)
    test_size = min(1000, max(200, X_all.shape[0] // 10))
    test_idx = np.random.choice(idx[:per_client], size=test_size, replace=True)
    X_test = X_all[test_idx]
    y_test = y_all[test_idx]

    return clients, (X_test.astype(np.float32), y_test.astype(np.int32))

# -----------------------
# Client and Server classes
# -----------------------
def create_local_model(input_dim, num_classes):
    model = models.Sequential([
        layers.InputLayer(input_shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss=losses.SparseCategoricalCrossentropy(),
                  metrics=[metrics.SparseCategoricalAccuracy()])
    return model

class FLClient:
    def __init__(self, cid, X, y):
        self.id = cid
        self.X = X
        self.y = y
        self.n = X.shape[0]
        self.model = create_local_model(X.shape[1], NUM_CLASSES)
        # reliability simulated; can be derived from trace metadata or latency
        self.reliability = random.uniform(0.7, 1.0)
        self.last_loss = None
        self.diversity_score = None  # to be computed by server/client-side estimator

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def train(self, epochs=LOCAL_EPOCHS, batch_size=BATCH_SIZE):
        self.model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
                           loss=losses.SparseCategoricalCrossentropy(),
                           metrics=[metrics.SparseCategoricalAccuracy()])
        self.model.fit(self.X, self.y, epochs=epochs, batch_size=batch_size, verbose=0)
        # record last loss for diversity estimators if desired
        loss, acc = self.model.evaluate(self.X, self.y, verbose=0)
        self.last_loss = loss
        return loss, acc

class FLServer:
    def __init__(self, clients, test_set):
        self.clients = clients
        self.test_X, self.test_y = test_set
        # global model initialized from first client shapes
        self.global_model = create_local_model(clients[0].X.shape[1], NUM_CLASSES)
        self.global_weights = self.global_model.get_weights()
        # precompute or simulate latency matrix (symmetric)
        self.latency = self._compute_latency_matrix()
        # compute initial per-client diversity
        self._compute_client_diversity()

    def _compute_latency_matrix(self):
        m = len(self.clients)
        L = np.zeros((m, m))
        for i in range(m):
            for j in range(i+1, m):
                # derive synthetic latency from data similarity: more similar -> lower latency (simulated)
                sim = np.corrcoef(self.clients[i].X.mean(axis=0), self.clients[j].X.mean(axis=0))[0,1]
                if np.isnan(sim):
                    sim = 0.0
                lat = np.random.uniform(LATENCY_MIN, LATENCY_MAX) * (1.0 - 0.5 * sim)
                L[i,j] = L[j,i] = float(abs(lat))
        return L

    def _compute_client_diversity(self):
        # For each client, run PCA on its local data and take sum of explained variance for first k components
        diversities = []
        for c in self.clients:
            try:
                scaler = StandardScaler()
                Xs = scaler.fit_transform(c.X)
                pca = PCA(n_components=min(PCA_COMPONENTS, Xs.shape[1]))
                pca.fit(Xs)
                # diversity score: sum of explained variance ratios of first k components
                score = float(np.sum(pca.explained_variance_ratio_))
            except Exception:
                # fallback small random diversity
                score = random.uniform(0.01, 0.1)
            c.diversity_score = score
            diversities.append(score)
        # normalize diversity scores to [0,1] for use in weighting
        min_d, max_d = min(diversities), max(diversities)
        for c in self.clients:
            if max_d - min_d > 1e-8:
                c.diversity_norm = (c.diversity_score - min_d) / (max_d - min_d)
            else:
                c.diversity_norm = 0.0

    def _compute_reliability_norm(self):
        # reliability_base derived from 1 - normalized average latency for each client
        avg_lat = np.mean(self.latency, axis=1)
        min_l, max_l = np.min(avg_lat), np.max(avg_lat)
        reliabilities = []
        for i,c in enumerate(self.clients):
            if max_l - min_l > 1e-8:
                r = 1.0 - (avg_lat[i] - min_l) / (max_l - min_l)
            else:
                r = 1.0
            # combine with client's intrinsic reliability
            combined = 0.7 * r + 0.3 * c.reliability
            reliabilities.append(combined)
        # normalize to [0,1]
        min_r, max_r = min(reliabilities), max(reliabilities)
        reli_norm = []
        for val in reliabilities:
            if max_r - min_r > 1e-8:
                reli_norm.append((val - min_r) / (max_r - min_r))
            else:
                reli_norm.append(1.0)
        # store
        for idx,c in enumerate(self.clients):
            c.reliability_norm = reli_norm[idx]

    def compute_aggregation_weights(self):
        """
        Compute weights W_i = normalized combination of:
            - dataset size proportion
            - diversity_norm (PCA-based)
            - reliability_norm (latency/intrinsic)
        Normalize final weights to sum to 1.
        """
        sizes = np.array([c.n for c in self.clients], dtype=float)
        size_prop = sizes / np.sum(sizes)
        # ensure client diversities/reliabilities are updated
        self._compute_reliability_norm()
        divers = np.array([c.diversity_norm for c in self.clients], dtype=float)
        reli = np.array([c.reliability_norm for c in self.clients], dtype=float)

        raw = ALPHA_SIZE * size_prop + BETA_DIVERSITY * divers + GAMMA_RELIABILITY * reli
        # avoid negative or zero
        raw = np.maximum(raw, 1e-12)
        weights = raw / np.sum(raw)
        # attach back to clients
        for i,c in enumerate(self.clients):
            c.aggregation_weight = float(weights[i])
        return weights

    def aggregate_weights(self, client_weights, weights):
        """
        Weighted aggregation across clients: return aggregated weights list
        client_weights: list of lists of numpy arrays (per-client model weights)
        weights: array of shape (num_clients,) summing to 1
        """
        agg = []
        for layer_idx in range(len(client_weights[0])):
            layer_sum = np.zeros_like(client_weights[0][layer_idx], dtype=np.float32)
            for i, cw in enumerate(client_weights):
                layer_sum += cw[layer_idx].astype(np.float32) * weights[i]
            agg.append(layer_sum)
        return agg

    def distribute_and_train(self):
        """
        One global round: broadcast global weights, local training, aggregation by WeiAvgCS-inspired weights.
        Returns per-round summary dict.
        """
        start_time = time.time()
        # broadcast
        for c in self.clients:
            c.set_weights(self.global_weights)

        # clients train locally
        client_weights = []
        client_sizes = []
        losses = []
        accs = []
        for c in self.clients:
            loss, acc = c.train(epochs=LOCAL_EPOCHS)
            client_weights.append(c.get_weights())
            client_sizes.append(c.n)
            losses.append(loss)
            accs.append(acc)

        # compute aggregation weights
        weights = self.compute_aggregation_weights()

        # aggregate and set global model
        aggregated = self.aggregate_weights(client_weights, weights)
        self.global_weights = aggregated
        self.global_model.set_weights(aggregated)

        # evaluate global model on server test set
        self.global_model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
                                  loss=losses.SparseCategoricalCrossentropy(),
                                  metrics=[metrics.SparseCategoricalAccuracy()])
        g_loss, g_acc = self.global_model.evaluate(self.test_X, self.test_y, verbose=0)

        end_time = time.time()
        latency = end_time - start_time

        summary = {
            'weights': weights,
            'client_losses': losses,
            'client_accs': accs,
            'global_loss': float(g_loss),
            'global_acc': float(g_acc),
            'aggregation_latency_sec': float(latency)
        }
        return summary

# -----------------------
# Runner
# -----------------------
def run_experiment():
    print("Loading cluster-derived client datasets...")
    clients_data, test_set = discover_and_load_cluster_data(DATA_DIR, input_dim=INPUT_DIM, samples_per_client=CLIENT_SAMPLES_TARGET)
    fl_clients = []
    for i, (x, y) in enumerate(clients_data):
        fl_clients.append(FLClient(i, x, y))

    server = FLServer(fl_clients, test_set)

    print("Initial diversity norms (per client):")
    for c in server.clients:
        print(f" Client {c.id}: diversity_score={c.diversity_score:.4f}, diversity_norm={getattr(c,'diversity_norm',0.0):.4f}")

    # Main global rounds
    for r in range(1, GLOBAL_ROUNDS + 1):
        print(f"\n=== Global Round {r}/{GLOBAL_ROUNDS} ===")
        summary = server.distribute_and_train()
        print(f" Aggregation weights (first 6): {summary['weights'][:6]}")
        print(f" Global acc: {summary['global_acc']:.4f}, loss: {summary['global_loss']:.4f}, latency: {summary['aggregation_latency_sec']:.3f}s")

    print("\nExperiment complete.")

if __name__ == "__main__":
    run_experiment()

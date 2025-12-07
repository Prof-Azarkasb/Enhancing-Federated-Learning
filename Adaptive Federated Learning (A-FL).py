"""
Advanced A-FL adapted to FedAWT + Google 2019 Cluster sample data loader.
- Reads CSV files placed in ./data/ (expects Google 2019 Cluster sample CSVs).
- Builds per-client local datasets by partitioning tasks/machines among synthetic 'clients'.
- Implements epoch allocation via a knapsack-like greedy optimizer under a time budget.
- Only requires: Python 3.8+, numpy, tensorflow (2.x), pandas (for CSV parsing).
  (pandas is used for convenience reading CSVs;)
"""

import os
import glob
import math
import random
import time
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses, optimizers, metrics

# ---------- Simulation & algorithmic params ----------
NUM_CLIENTS = 10
CLIENT_FRACTION = 0.6
GLOBAL_ROUNDS = 8
BATCH_SIZE = 32
BASE_LEARNING_RATE = 0.01

MIN_LOCAL_EPOCH = 1
MAX_LOCAL_EPOCH = 8
BASE_LOCAL_EPOCH = 2

# time-budget factor (how much wall-time per round we allow relative to naive baseline)
TIME_BUDGET_FACTOR = 1.2

DATA_DIR = "./data"  # place Kaggle-downloaded CSV files here

# ---------- Simple model ----------
def create_model():
    model = tf.keras.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

# ---------- Data ingestion: Google Cluster CSV discovery & simple preprocessing ----------
def discover_and_load_cluster_data(data_dir=DATA_DIR, input_dim=784, samples_per_client_target=600):
    """
    Attempt to load relevant CSVs from Google cluster sample and create `NUM_CLIENTS` local datasets.
    Strategy:
    - Search CSV files in data_dir.
    - Try to find files that include columns referring to CPU usage / memory / timestamp.
    - Build a per-'machine' or per-'task' aggregation and then partition machines across synthetic clients.
    Returns: list of tuples (X, y) per client, and (test_X, test_y)
    Note: labels are synthetic (we create a classification target) because Google trace is not labeled for ML classification.
    We instead create a proxy task: predict a bucketed CPU usage class from recent features.
    """
    import pandas as pd  # used only for convenient CSV parsing

    csv_paths = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {data_dir}. Please download the Kaggle dataset and place CSVs there.")

    # We'll scan files until we can extract per-machine or per-task mean CPU usage series.
    # Build a list of machine-series: dict machine_id -> list of cpu samples
    machine_series = defaultdict(list)
    # For simplicity try to find files that contain 'cpu' in header or known TRU-like structure
    for path in csv_paths:
        try:
            df = pd.read_csv(path, low_memory=True)
        except Exception as e:
            # skip heavy or incompatible files
            print(f"Skipping {path} due to read error: {e}")
            continue
        cols = [c.lower() for c in df.columns]
        # heuristic: if 'mean_cpu_usage' or 'cpu_utilization' or 'mean_cpu' exists, use it
        possible_cpu_cols = [c for c in df.columns if 'cpu' in c.lower()]
        # pick a candidate numeric CPU column if available
        cpu_col = None
        for c in possible_cpu_cols:
            if np.issubdtype(df[c].dtype, np.number):
                cpu_col = c
                break
        if cpu_col is None:
            # else skip
            continue

        # try to find a machine id or task id column
        id_col = None
        for cand in ['machine_id', 'machine', 'machineid', 'task_id', 'taskid', 'task', 'job_id']:
            if cand in cols:
                id_col = [c for c in df.columns if c.lower() == cand][0]
                break

        if id_col is None:
            # use filename as pseudo-machine id
            pseudo_id = os.path.basename(path)
            vals = df[cpu_col].fillna(0.0).astype(float).values
            machine_series[pseudo_id].extend(vals.tolist())
        else:
            # group by id_col and collect cpu samples
            for mid, g in df.groupby(id_col):
                vals = g[cpu_col].fillna(0.0).astype(float).values
                if len(vals) > 0:
                    machine_series[str(mid)].extend(vals.tolist())

    if not machine_series:
        raise RuntimeError("No usable CPU-like columns found in CSVs. Inspect dataset files and column names.")

    # Convert series into feature vectors (e.g., take last N samples or stat summaries)
    # We'll construct per-machine feature vector of fixed dimension (input_dim) by repeating/truncating.
    machine_ids = list(machine_series.keys())
    num_machines = len(machine_ids)
    per_machine_vectors = []
    for mid in machine_ids:
        series = np.array(machine_series[mid], dtype=np.float32)
        if series.size == 0:
            continue
        # normalize series to [0,1]
        smin, smax = series.min(), series.max()
        if smax > smin:
            series = (series - smin) / (smax - smin)
        else:
            series = np.zeros_like(series)
        # produce a fixed-size vector
        if series.size >= input_dim:
            vec = series[-input_dim:]
        else:
            # pad by repeating the series
            repeats = math.ceil(input_dim / series.size)
            big = np.tile(series, repeats)[:input_dim]
            vec = big
        per_machine_vectors.append(vec)

    # Build synthetic labels: bucket average CPU usage into 10 classes (0..9)
    X_all = np.stack(per_machine_vectors, axis=0)
    avg_cpu = X_all.mean(axis=1)
    # bucket into classes
    labels_all = np.minimum(9, (avg_cpu * 10).astype(int))

    # Now partition machines across NUM_CLIENTS
    samples = X_all.shape[0]
    idxs = np.arange(samples)
    np.random.shuffle(idxs)
    clients = []
    per_client_counts = max(1, samples // NUM_CLIENTS)
    for i in range(NUM_CLIENTS):
        start = i * per_client_counts
        end = start + per_client_counts
        if i == NUM_CLIENTS - 1:
            end = samples
        sel = idxs[start:end]
        if len(sel) == 0:
            # create a tiny random dataset
            x = np.random.rand(samples_per_client_target, input_dim).astype(np.float32)
            y = np.random.randint(0, 10, size=(samples_per_client_target,))
        else:
            x = X_all[sel]
            y = labels_all[sel]
        # optionally augment to reach target samples per client
        if x.shape[0] < samples_per_client_target:
            reps = math.ceil(samples_per_client_target / x.shape[0])
            x = np.tile(x, (reps, 1))[:samples_per_client_target]
            y = np.tile(y, reps)[:samples_per_client_target]
        clients.append((x.astype(np.float32), y.astype(np.int32)))
    # create a test set by sampling held-out machines
    test_size = min(1000, max(200, samples // 10))
    test_idx = np.random.choice(np.setdiff1d(np.arange(samples), idxs[:NUM_CLIENTS*per_client_counts]), size=test_size, replace=True)
    if test_idx.size == 0:
        test_X = X_all[:test_size]
        test_y = labels_all[:test_size]
    else:
        test_X = X_all[test_idx]
        test_y = labels_all[test_idx]

    return clients, (test_X.astype(np.float32), test_y.astype(np.int32))


# ---------- Client & Server with advanced epoch allocation ----------
class ClientNode:
    def __init__(self, cid, data, labels):
        self.id = cid
        self.data = data
        self.labels = labels
        self.model = create_model()
        self.optimizer = optimizers.SGD(learning_rate=BASE_LEARNING_RATE)
        self.loss_fn = losses.SparseCategoricalCrossentropy()
        self.metric = metrics.SparseCategoricalAccuracy()
        self.loss_history = []
        self.local_epoch = BASE_LOCAL_EPOCH
        # simulated compute cost per epoch (seconds): heterogeneous
        self.compute_cost = random.uniform(0.6, 2.0)

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def set_optimizer(self, name='SGD'):
        if name == 'SGD':
            self.optimizer = optimizers.SGD(learning_rate=BASE_LEARNING_RATE)
        elif name == 'Adam':
            self.optimizer = optimizers.Adam(learning_rate=BASE_LEARNING_RATE)
        elif name == 'Adagrad':
            self.optimizer = optimizers.Adagrad(learning_rate=BASE_LEARNING_RATE)
        else:
            self.optimizer = optimizers.SGD(learning_rate=BASE_LEARNING_RATE)

    def evaluate_local_loss(self):
        ds = tf.data.Dataset.from_tensor_slices((self.data, self.labels)).batch(BATCH_SIZE)
        total = 0.0
        n = 0
        for xb, yb in ds:
            logits = self.model(xb, training=False)
            total += tf.reduce_sum(self.loss_fn(yb, logits)).numpy()
            n += xb.shape[0]
        return total / max(1, n)

    def trial_epoch_and_estimate(self):
        """
        Run one epoch (trial) to estimate per-epoch reduction in loss.
        Returns: delta_loss (pre_loss - post_loss), time_cost_estimate (self.compute_cost)
        """
        pre = self.evaluate_local_loss()
        # do 1 epoch training
        ds = tf.data.Dataset.from_tensor_slices((self.data, self.labels)).shuffle(1000).batch(BATCH_SIZE)
        self.model.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=[self.metric])
        h = self.model.fit(ds, epochs=1, verbose=0)
        post = h.history['loss'][-1] if 'loss' in h.history else self.evaluate_local_loss()
        delta = max(0.0, pre - post)
        self.loss_history.append(post)
        if len(self.loss_history) > 5:
            self.loss_history = self.loss_history[-5:]
        return delta, self.compute_cost

    def train_local_epochs(self, epochs):
        if epochs <= 0:
            return None
        ds = tf.data.Dataset.from_tensor_slices((self.data, self.labels)).shuffle(1000).batch(BATCH_SIZE)
        self.model.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=[self.metric])
        h = self.model.fit(ds, epochs=epochs, verbose=0)
        last_loss = h.history['loss'][-1] if 'loss' in h.history else self.evaluate_local_loss()
        self.loss_history.append(last_loss)
        if len(self.loss_history) > 5:
            self.loss_history = self.loss_history[-5:]
        return last_loss

class FederatedServerAdvanced:
    def __init__(self, clients):
        self.clients = clients
        self.global_model = create_model()
        self.global_weights = self.global_model.get_weights()

    def broadcast(self, participants):
        # For simplicity, immediate set (real network simulated elsewhere)
        for c in participants:
            c.set_weights(self.global_weights)

    def aggregate(self, weight_list):
        new_w = []
        for tup in zip(*weight_list):
            new_w.append(np.mean(tup, axis=0))
        self.global_weights = new_w
        self.global_model.set_weights(new_w)

    def evaluate_global(self, test_X, test_y):
        self.global_model.compile(optimizer='adam', loss=losses.SparseCategoricalCrossentropy(), metrics=[metrics.SparseCategoricalAccuracy()])
        loss, acc = self.global_model.evaluate(test_X, test_y, verbose=0)
        return loss, acc

    def allocate_epochs_knapsack_greedy(self, participants, time_budget):
        """
        Greedy knapsack-like integer allocation:
        - For each participant estimate b_i (expected loss reduction per epoch) and t_i (time per epoch).
        - Initialize e_i = MIN_LOCAL_EPOCH for all.
        - Remaining_budget = time_budget - sum(t_i * e_i)
        - While remaining_budget >= min(t_i) :
            choose client with highest (b_i / t_i) among those not at MAX_LOCAL_EPOCH and allocate +1 epoch (update remaining_budget).
        Returns dict client_id->allocated_epochs
        """
        # step 1: estimate b_i and t_i using trial epoch if needed
        estimates = {}
        for c in participants:
            # if we have history, use average reduction per epoch from history:
            if len(c.loss_history) >= 2:
                # approximate: last reduction per epoch
                recent = c.loss_history[-2:]
                b_est = max(1e-6, recent[0] - recent[1])
                t_est = c.compute_cost
            else:
                # run a light trial epoch to estimate (this modifies model slightly)
                b_est, t_est = c.trial_epoch_and_estimate()
                # avoid zero b_est
                b_est = max(b_est, 1e-6)
            estimates[c.id] = {'b': b_est, 't': t_est, 'cur_e': MIN_LOCAL_EPOCH}

        # initial allocation: set everyone to MIN_LOCAL_EPOCH
        allocated = {c.id: MIN_LOCAL_EPOCH for c in participants}
        used_time = sum(estimates[c.id]['t'] * MIN_LOCAL_EPOCH for c in participants)
        remaining = time_budget - used_time

        # greedy loop
        # create a list of available clients
        while True:
            candidates = []
            for c in participants:
                cid = c.id
                if allocated[cid] >= MAX_LOCAL_EPOCH:
                    continue
                t_i = estimates[cid]['t']
                if t_i <= 0 or t_i > remaining:
                    continue
                b_i = estimates[cid]['b']
                # marginal benefit per unit time
                ratio = b_i / t_i
                candidates.append((ratio, cid, b_i, t_i))
            if not candidates:
                break
            # pick best ratio
            candidates.sort(reverse=True, key=lambda x: x[0])
            best = candidates[0]
            _, best_cid, _, best_t = best
            # allocate one epoch
            allocated[best_cid] += 1
            remaining -= best_t
            if remaining < min([estimates[cid]['t'] for cid in allocated if allocated[cid] < MAX_LOCAL_EPOCH] + [remaining + 1]):  # quick stop
                # continue loop; condition may stop naturally
                pass
            # loop continues until no candidate fits
        # ensure bounds
        for k in allocated:
            allocated[k] = int(max(MIN_LOCAL_EPOCH, min(MAX_LOCAL_EPOCH, allocated[k])))
        return allocated

    def run_round(self, round_idx, optimizer_name='SGD', time_budget=None):
        m = max(1, int(CLIENT_FRACTION * len(self.clients)))
        participants = random.sample(self.clients, m)
        # broadcast
        self.broadcast(participants)

        # set optimizers
        for c in participants:
            c.set_optimizer(optimizer_name)

        # set a default time_budget if none: TIME_BUDGET_FACTOR * naive baseline
        if time_budget is None:
            naive = sum(c.compute_cost * BASE_LOCAL_EPOCH for c in participants)
            time_budget = TIME_BUDGET_FACTOR * naive

        # allocate epochs using greedy knapsack-like
        allocation = self.allocate_epochs_knapsack_greedy(participants, time_budget)
        print(f"[Round {round_idx}] time_budget={time_budget:.2f}s allocation(sample): {list(allocation.items())[:6]}")

        # perform local training with allocated epochs
        weight_list = []
        for c in participants:
            e = allocation.get(c.id, BASE_LOCAL_EPOCH)
            last_loss = c.train_local_epochs(e)
            weight_list.append(c.get_weights())
            print(f"  Client {c.id}: epochs={e}, last_loss={last_loss:.4f}, cost_est={c.compute_cost * e:.2f}s")

        # aggregate
        self.aggregate(weight_list)
        return participants

# ---------- Runner ----------
def run_advanced_simulation():
    # load cluster data and create client datasets
    clients_data, test_set = discover_and_load_cluster_data(DATA_DIR)
    test_X, test_y = test_set

    # create client nodes
    nodes = []
    for i, (x, y) in enumerate(clients_data):
        n = ClientNode(i, x, y)
        nodes.append(n)

    server = FederatedServerAdvanced(nodes)

    for r in range(1, GLOBAL_ROUNDS + 1):
        print(f"\n=== Global Round {r}/{GLOBAL_ROUNDS} ===")
        server.run_round(r, optimizer_name='Adam', time_budget=None)
        loss, acc = server.evaluate_global(test_X, test_y)
        print(f"Global Eval after Round {r}: loss={loss:.4f}, acc={acc:.4f}")

if __name__ == "__main__":
    # Ensure DATA_DIR exists and has files
    if not os.path.isdir(DATA_DIR):
        raise RuntimeError(f"Data folder '{DATA_DIR}' not found. Please download Google 2019 Cluster sample from Kaggle and extract CSVs into this folder. Kaggle page: https://www.kaggle.com/datasets/derrickmwiti/google-2019-cluster-sample")
    run_advanced_simulation()

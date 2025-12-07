# fl_im_impl.py
import os
import time
import math
import random
import psutil
import numpy as np
import pandas as pd
import networkx as nx
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# -----------------------------
# Configurable experiment params
# -----------------------------
DATASET_PATH = "path/to/google_2019_cluster_sample.csv"  # 
NUM_NODES = 50           # use subset of machines for speed (adjust)
NUM_ROUNDS = 12
LOCAL_EPOCHS = 3
BATCH_SIZE = 32
INFLUENCE_TOP_RATIO = 0.5   # fraction of nodes considered as influential candidates initially
INFLUENCE_THRESHOLD_K = 0.0 # if <=0, threshold computed from distribution
VALIDATION_SPLIT = 0.2
SEED = 42
RANDOM_FAILURE_RATE = 0.0   # set >0 to simulate client comms failures

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------------
# Utility / preprocessing
# -----------------------------
def load_and_prepare(path, sample_machines=NUM_NODES):
    df = pd.read_csv(path)
    # remove initial id columns if present (some versions have two leading columns)
    # For safety, drop unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # simplify: make a binary target from 'priority' column (>= median -> 1)
    if 'priority' not in df.columns:
        raise ValueError("Dataset must contain a 'priority' column for this simplified implementation.")
    df = df.dropna(subset=['priority', 'machine_id', 'start_time', 'end_time'])
    df['priority_bin'] = (df['priority'] >= df['priority'].median()).astype(int)
    # Feature selection: choose a compact set that exists in the trace
    feats = []
    for col in ['resource_request','assigned_memory','average_usage','maximum_usage','cpu_usage_distribution','tail_cpu_usage_distribution']:
        if col in df.columns:
            feats.append(col)
    # fallback: if none present use numeric columns except identifiers
    if len(feats) == 0:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude = ['priority','priority_bin','machine_id','start_time','end_time','time']
        feats = [c for c in numeric_cols if c not in exclude][:6]
    df = df[['machine_id','start_time','end_time','priority','priority_bin'] + feats]
    # compute execution_time
    # start_time and end_time may be strings — coerce to numeric where possible
    df['start_time'] = pd.to_numeric(df['start_time'], errors='coerce')
    df['end_time'] = pd.to_numeric(df['end_time'], errors='coerce')
    df['exec_time'] = (df['end_time'] - df['start_time']).fillna(0).clip(lower=0.0)
    # select subset of machines (largest by count)
    machine_counts = df['machine_id'].value_counts().nlargest(sample_machines)
    df = df[df['machine_id'].isin(machine_counts.index)].copy()
    # fill missing feature values reasonably
    for c in feats:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(df[c].median())
    # standardize features per global scaler (server-side)
    scaler = StandardScaler()
    df[feats] = scaler.fit_transform(df[feats])
    return df, feats

# -----------------------------
# Model factory (simple predictor)
# -----------------------------
def build_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# -----------------------------
# Node / Client class
# -----------------------------
class NodeClient:
    def __init__(self, node_id, df_node, feature_cols, local_epochs=LOCAL_EPOCHS):
        self.node_id = node_id
        self.df = df_node.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.local_epochs = local_epochs
        self.model = build_model(input_dim=len(feature_cols))
        # Node statistics for influence scoring
        self.avg_exec_time = float(self.df['exec_time'].mean()) if len(self.df)>0 else 0.0
        self.failure_rate = float((self.df.get('failed',0)==1).mean()) if 'failed' in self.df.columns else 0.0
        self.avg_usage = float(self.df['average_usage'].mean()) if 'average_usage' in self.df.columns else 0.0
        # For behavior trajectory capture
        self.trajectory_buffer = []  # list of tuples (x_vec, predicted_label, reward)
        # create local train/test split
        X = self.df[self.feature_cols].values
        y = self.df['priority_bin'].values
        if len(X) > 4:
            self.X_train, self.X_hold, self.y_train, self.y_hold = train_test_split(X, y, test_size=VALIDATION_SPLIT, random_state=SEED)
        else:
            self.X_train, self.X_hold, self.y_train, self.y_hold = X, X, y, y

    def estimate_local_performance(self):
        # quick proxy: success fraction and average exec_time (lower exec_time => better)
        success = 1.0 - self.failure_rate
        perf = success * (1.0 / (1.0 + self.avg_exec_time))  # in (0,1]
        return perf

    def local_train(self, epochs=None):
        if epochs is None:
            epochs = self.local_epochs
        start_cpu = psutil.cpu_percent(interval=None)
        start_mem = psutil.virtual_memory().percent
        # fit model on node's local data
        if len(self.X_train) == 0:
            return None, None  # no data
        self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=BATCH_SIZE, verbose=0)
        # collect simple trajectories: for each sample in hold-out, record prediction & reward proxy
        preds = (self.model.predict(self.X_hold, verbose=0) > 0.5).astype(int).flatten()
        # define reward as success (1) * inverse normalized exec time for that sample (use node avg if not available)
        rewards = []
        for idx in range(len(self.X_hold)):
            # use node-level exec_time as proxy (could be per-sample if available)
            rt = self.avg_exec_time
            reward = (1.0 if preds[idx] == self.y_hold[idx] else 0.0) * (1.0 / (1.0 + rt))
            rewards.append(reward)
        # store trajectories (we compress to vector summaries for efficiency)
        for i in range(len(self.X_hold)):
            self.trajectory_buffer.append((self.X_hold[i], int(preds[i]), float(rewards[i])))
        end_cpu = psutil.cpu_percent(interval=None)
        end_mem = psutil.virtual_memory().percent
        cpu_delta = end_cpu - start_cpu
        mem_delta = end_mem - start_mem
        # return model weights and a small per-node summary
        return self.model.get_weights(), {
            'cpu_delta': cpu_delta,
            'mem_delta': mem_delta,
            'avg_reward': np.mean(rewards) if len(rewards)>0 else 0.0,
            'modal_action': int(pd.Series(preds).mode().iloc[0]) if len(preds)>0 else 0
        }

# -----------------------------
# Influence scoring utilities
# -----------------------------
def compute_influence_scores(nodes):
    """
    Compute influence score I(n_i,t) for each node based on:
    - reliability/perf proxy (estimate_local_performance),
    - resource-efficiency (inverse exec time),
    - historical avg reward (if available).
    Returns dict {node_id: score}
    """
    scores = {}
    # collect raw measures
    perfs = []
    avg_rewards = []
    exec_inv = []
    for n in nodes:
        perf = n.estimate_local_performance()
        perfs.append(perf)
        avg_r = np.mean([t[2] for t in n.trajectory_buffer]) if len(n.trajectory_buffer)>0 else 0.0
        avg_rewards.append(avg_r)
        exec_inv.append(1.0/(1.0 + n.avg_exec_time))
    # normalize
    def norm_list(lst):
        a = np.array(lst, dtype=float)
        if a.max() - a.min() < 1e-9:
            return np.ones_like(a)
        return (a - a.min()) / (a.max() - a.min())
    np_perfs = norm_list(perfs)
    np_rewards = norm_list(avg_rewards)
    np_exec = norm_list(exec_inv)
    # combine with weights (tunable)
    w1, w2, w3 = 0.4, 0.3, 0.3
    for i,n in enumerate(nodes):
        scores[n.node_id] = w1*np_perfs[i] + w2*np_rewards[i] + w3*np_exec[i]
    return scores

# -----------------------------
# Behavior-aware aggregation stages
# -----------------------------
def filter_low_value_trajectories(nodes, adaptive_factor=0.75):
    """
    Stage 1: remove trajectories with mean reward below adaptive threshold.
    We compute a threshold per-node as adaptive_factor * node_mean_reward, and keep trajectories >= threshold_global
    """
    all_rewards = []
    for n in nodes:
        all_rewards.extend([t[2] for t in n.trajectory_buffer])
    if len(all_rewards) == 0:
        return  # nothing to do
    global_thr = np.mean(all_rewards) * adaptive_factor
    # filter buffers in-place: keep only trajectories with reward >= global_thr
    for n in nodes:
        n.trajectory_buffer = [t for t in n.trajectory_buffer if t[2] >= global_thr]

def detect_cross_node_patterns(nodes):
    """
    Stage 2: assign pattern-frequency weight per node based on the modal predicted action
    We return a dict reliability_by_node[node_id] initialised with pattern frequency fraction.
    """
    modal_counts = {}
    for n in nodes:
        if len(n.trajectory_buffer)==0:
            modal = None
        else:
            modal = pd.Series([t[1] for t in n.trajectory_buffer]).mode().iloc[0]
        modal_counts.setdefault(modal, []).append(n.node_id)
    # compute weight proportional to number of nodes sharing same modal prediction
    reliability = {}
    for n in nodes:
        if len(n.trajectory_buffer)==0:
            reliability[n.node_id] = 0.0
        else:
            modal = pd.Series([t[1] for t in n.trajectory_buffer]).mode().iloc[0]
            freq = len(modal_counts.get(modal, []))
            reliability[n.node_id] = freq / max(1, len(nodes))
    return reliability

def evaluate_trajectory_stability(nodes, perturb_sigma=0.05, repeats=5):
    """
    Stage 3: perturb features and recompute predicted rewards to estimate variance.
    Lower variance -> higher stability score in [0,1].
    Returns dict stability_score[node_id] between 0 and 1.
    """
    stability = {}
    for n in nodes:
        if len(n.trajectory_buffer)==0:
            stability[n.node_id] = 0.0
            continue
        # collect original X_hold
        X_hold = np.vstack([t[0] for t in n.trajectory_buffer])
        # baseline preds and rewards (already have reward in buffer)
        base_rewards = np.array([t[2] for t in n.trajectory_buffer])
        simulated_means = []
        for r in range(repeats):
            noise = np.random.normal(loc=0.0, scale=perturb_sigma, size=X_hold.shape)
            Xp = X_hold + noise
            preds = (n.model.predict(Xp, verbose=0) > 0.5).astype(int).flatten()
            # recompute reward proxy
            rewards = []
            for i in range(len(preds)):
                reward = (1.0 if preds[i] == n.y_hold[i] else 0.0) * (1.0 / (1.0 + n.avg_exec_time))
                rewards.append(reward)
            simulated_means.append(np.mean(rewards))
        var = np.var(simulated_means)
        # convert variance to stability score (smaller var -> higher score)
        stab = 1.0 / (1.0 + var*100.0)  # scale factor for sensitivity
        stability[n.node_id] = float(np.clip(stab, 0.0, 1.0))
    return stability

def compose_refinement_operator(nodes, influence_scores, pattern_weights, stability_scores, alpha_influence=0.6):
    """
    Compose a per-node scalar factor that will scale the local update.
    Final weight = alpha_influence * normalized_influence + (1-alpha_influence)*composite_behavior_score
    where composite_behavior_score = pattern_weight * stability_score * normalized_avg_reward
    Returns dict final_scalar[node_id]
    """
    # normalize influence
    vals = np.array(list(influence_scores.values()), dtype=float)
    if vals.max() - vals.min() < 1e-9:
        norm_infl = {k:1.0 for k in influence_scores}
    else:
        norm_vals = (vals - vals.min()) / (vals.max() - vals.min())
        norm_infl = {k: float(norm_vals[i]) for i,k in enumerate(influence_scores.keys())}
    # normalize avg_reward per node
    avg_rewards = {n.node_id: (np.mean([t[2] for t in n.trajectory_buffer]) if len(n.trajectory_buffer)>0 else 0.0) for n in nodes}
    ar_vals = np.array(list(avg_rewards.values()))
    if ar_vals.max()-ar_vals.min() < 1e-9:
        norm_rewards = {k:1.0 for k in avg_rewards}
    else:
        nr = (ar_vals - ar_vals.min()) / (ar_vals.max() - ar_vals.min())
        norm_rewards = {k: float(nr[i]) for i,k in enumerate(avg_rewards.keys())}
    final = {}
    for n in nodes:
        pid = n.node_id
        pattern_w = pattern_weights.get(pid, 0.0)
        stab = stability_scores.get(pid, 0.0)
        reward_n = norm_rewards.get(pid, 0.0)
        composite = pattern_w * stab * reward_n
        final_scalar = alpha_influence * norm_infl.get(pid, 0.0) + (1.0 - alpha_influence) * composite
        final[pid] = float(np.clip(final_scalar, 0.0, 1.0))
    return final

# -----------------------------
# Server-side aggregation applying φ(·)
# -----------------------------
def aggregate_validated_updates(nodes, client_updates, final_scalars, learning_rate=1.0):
    """
    client_updates: dict node_id -> weights (list of numpy arrays)
    final_scalars: dict node_id -> scalar in [0,1]
    returns: aggregated_weights (list of arrays) suitable for set_weights
    """
    node_ids = list(client_updates.keys())
    # compute normalized influence weights for node-level weighting
    scalars = np.array([final_scalars[nid] for nid in node_ids], dtype=float)
    if scalars.sum() == 0:
        norm_weights = np.ones_like(scalars) / len(scalars)
    else:
        norm_weights = scalars / scalars.sum()
    # weighted aggregation across layers
    # assume each client update is a list of numpy arrays matching model.get_weights()
    aggregated = []
    # zip across layers
    for layer_weights in zip(*[client_updates[nid] for nid in node_ids]):
        layer_stack = np.stack(layer_weights, axis=0)  # shape (num_nodes, ...)
        weighted = np.tensordot(norm_weights, layer_stack, axes=(0,0))
        aggregated.append(weighted.astype(layer_stack.dtype))
    return aggregated

# -----------------------------
# Main FL-IM orchestration (server side)
# -----------------------------
def run_fl_im(df, feature_cols):
    # create nodes from machine_id groups
    nodes = []
    for i, (mid, sub) in enumerate(df.groupby('machine_id')):
        nodes.append(NodeClient(node_id=mid, df_node=sub, feature_cols=feature_cols))
    # initial influence estimation (simple)
    # warm-up: let nodes train once locally to populate trajectory buffers
    initial_subset = nodes
    print(f"[Init] warming up {len(initial_subset)} nodes with local training")
    for n in initial_subset:
        w, summary = n.local_train(epochs=1)
    # main rounds
    server_model = build_model(input_dim=len(feature_cols))
    # initialize server weights as average of local models if available
    # (or random init)
    for rnd in range(1, NUM_ROUNDS+1):
        print(f"\n=== Round {rnd}/{NUM_ROUNDS} ===")
        # recompute influence scores
        influence_scores = compute_influence_scores(nodes)
        # threshold selection
        infl_vals = np.array(list(influence_scores.values()))
        if INFLUENCE_THRESHOLD_K <= 0:
            tau = infl_vals.mean() + 0.0 * infl_vals.std()  # simple threshold; tune lambda if desired
        else:
            tau = INFLUENCE_THRESHOLD_K
        selected = [n for n in nodes if influence_scores[n.node_id] >= tau]
        # optionally limit number of selected to top-k ratio
        top_k = max(1, int(len(nodes) * INFLUENCE_TOP_RATIO))
        selected = sorted(selected, key=lambda n: influence_scores[n.node_id], reverse=True)[:top_k]
        print(f"Selected {len(selected)} nodes (top ratio).")

        client_updates = {}
        # local training and sending updates
        for n in selected:
            # simulate possible communication failure
            if random.random() < RANDOM_FAILURE_RATE:
                print(f"Node {n.node_id} failed to send update (simulated).")
                continue
            weights, summary = n.local_train()  # trains and appends to trajectory_buffer
            if weights is None:
                continue
            client_updates[n.node_id] = deepcopy(weights)

        if len(client_updates) == 0:
            print("No updates received this round.")
            continue

        # Behavior-aware aggregation pipeline:
        # Stage 1: filter low-value trajectories (global)
        filter_low_value_trajectories(selected, adaptive_factor=0.75)
        # Stage 2: cross-node pattern detection
        pattern_weights = detect_cross_node_patterns(selected)
        # Stage 3: stability evaluation (perturbation-based)
        stability_scores = evaluate_trajectory_stability(selected, perturb_sigma=0.05, repeats=4)
        # Stage 4: compose final per-node scalar (influence + behavior)
        final_scalars = compose_refinement_operator(selected, influence_scores, pattern_weights, stability_scores, alpha_influence=0.65)

        # Aggregate validated updates (φ applied via scaling)
        aggregated_weights = aggregate_validated_updates(selected, client_updates, final_scalars)
        server_model.set_weights(aggregated_weights)

        # Evaluate global model on pooled holdout from nodes (quick aggregate)
        X_val = np.vstack([n.X_hold for n in selected if hasattr(n,'X_hold') and len(n.X_hold)>0])
        y_val = np.hstack([n.y_hold for n in selected if hasattr(n,'y_hold') and len(n.y_hold)>0])
        if len(X_val)>0:
            preds = (server_model.predict(X_val, verbose=0) > 0.5).astype(int).flatten()
            acc = accuracy_score(y_val, preds)
            f1 = f1_score(y_val, preds)
            print(f"[Round Eval] accuracy={acc:.4f}, f1={f1:.4f}")
        else:
            print("[Round Eval] no validation samples available.")

        # Dynamic adaptation (simple): adjust INFLUENCE_TOP_RATIO mildly if performance stalls
        # (here a placeholder; in experiments this becomes a controlled update)
        # end round

    # Final evaluation on pooled dataset
    X_all = np.vstack([n.X_hold for n in nodes if hasattr(n,'X_hold') and len(n.X_hold)>0])
    y_all = np.hstack([n.y_hold for n in nodes if hasattr(n,'y_hold') and len(n.y_hold)>0])
    if len(X_all)>0:
        preds = (server_model.predict(X_all, verbose=0) > 0.5).astype(int).flatten()
        acc = accuracy_score(y_all, preds)
        f1 = f1_score(y_all, preds)
        print(f"\nFinal Global Model - accuracy={acc:.4f}, f1={f1:.4f}")
    else:
        print("No evaluation samples for final model.")

    return server_model

# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    # ensure dataset path set
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Please set DATASET_PATH to your local Google Cluster CSV file. Current: {DATASET_PATH}")
    df, feature_cols = load_and_prepare(DATASET_PATH, sample_machines=NUM_NODES)
    model = run_fl_im(df, feature_cols)

"""
GNNs-Based Federated Learning (GNNs-FL) — TensorFlow implementation (no torch/torch_geometric)
- Uses Google 2019 Cluster sample placed under ./data/ (Kaggle dataset: derrickmwiti/google-2019-cluster-sample)
- Builds per-client feature vectors from numeric CSV columns.
- Constructs k-NN graph among clients.
- Implements a small Graph Attention Network (GAT)-style layer in TensorFlow Keras.
- Trains the GNN to predict a synthetic resource-allocation target derived from traces (e.g., normalized CPU demand).
- Provides neighbor selection based on learned attention weights.
Run: python gnn_fl_cluster_tf.py
"""

import os
import glob
import math
import random
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, losses, metrics

# ---------------- reproducibility ----------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------- config ----------------
DATA_DIR = "./data"           # place Kaggle CSVs here
INPUT_DIM = 64               # feature vector length per node
NUM_CLIENTS = 20             # how many pseudo-clients (will be min(#files, NUM_CLIENTS))
K_NEIGHBORS = 4             # graph connectivity
GNN_HIDDEN = 32
GNN_OUTPUT = 1               # resource allocation scalar (regression)
BATCH_SIZE = 4
EPOCHS = 100
LR = 0.005
TOP_K_NEIGHBORS_SELECT = 5   # neighbor selection k
TEST_RATIO = 0.2

# ---------------- data loader: build node features from CSVs ----------------
def load_cluster_node_features(data_dir: str = DATA_DIR, input_dim: int = INPUT_DIM, max_nodes: int = NUM_CLIENTS):
    """
    Scan CSVs in data_dir, pick numeric columns, aggregate per-file into a fixed-length vector.
    Returns:
        X (N x input_dim), y (N,) synthetic regression targets, file_keys (list)
    """
    csv_paths = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {data_dir}. Please download Kaggle dataset and extract CSVs into this folder.")
    # we'll build one node per CSV file (pseudo-client) up to max_nodes
    csv_paths = csv_paths[:max_nodes]
    vectors = []
    keys = []
    for p in csv_paths:
        try:
            df = pd.read_csv(p, low_memory=True)
        except Exception:
            continue
        numeric = df.select_dtypes(include=[np.number]).fillna(0.0)
        if numeric.shape[1] == 0:
            continue
        # choose first numeric column as representative time-series (heuristic)
        col = numeric.columns[0]
        vals = numeric[col].values.astype(np.float32)
        if vals.size == 0:
            continue
        # normalize local series
        if vals.max() > vals.min():
            vals = (vals - vals.min()) / (vals.max() - vals.min())
        else:
            vals = np.zeros_like(vals)
        # produce fixed-length vector by truncation/padding via tiling
        if vals.size >= input_dim:
            vec = vals[:input_dim]
        else:
            reps = math.ceil(input_dim / max(1, vals.size))
            vec = np.tile(vals, reps)[:input_dim]
        vectors.append(vec)
        keys.append(os.path.basename(p))
    X = np.stack(vectors, axis=0)
    # produce synthetic scalar targets: e.g., normalized mean CPU usage scaled -> resource allocation target
    y_raw = X.mean(axis=1)  # in [0,1] because we normalized series
    # map to range [0.1, 1.0] as allocation fraction
    y = 0.1 + 0.9 * y_raw
    return X.astype(np.float32), y.astype(np.float32), keys

# ---------------- graph builder (k-NN) ----------------
def build_knn_graph(X: np.ndarray, k: int = K_NEIGHBORS):
    """
    Build undirected k-NN adjacency list and neighbor indices.
    Returns:
        neighbors: list of arrays of neighbor indices per node
        edge_index: 2 x E numpy int array (source, target) for edges in both directions
    """
    nbrs = NearestNeighbors(n_neighbors=min(k+1, X.shape[0]), algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)
    # indices includes node itself at position 0; drop self
    neighbors = []
    src = []
    dst = []
    for i in range(X.shape[0]):
        neigh = indices[i][indices[i] != i][:k]
        neighbors.append(neigh)
        for j in neigh:
            src.append(i)
            dst.append(j)
            # also add reverse to ensure undirected
            src.append(j)
            dst.append(i)
    edge_index = np.vstack([np.array(src, dtype=np.int32), np.array(dst, dtype=np.int32)])
    return neighbors, edge_index

# ---------------- Simple GAT-like layer implemented in TensorFlow ----------------
class SimpleGATLayer(layers.Layer):
    """
    Single-head attention GAT-like layer (per-graph, single graph at a time).
    Input: node features (N, F)
    Edge_index provided externally (2, E)
    Output: updated node features (N, out_features)
    """
    def __init__(self, out_features, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.out_features = out_features
        self.activation = activation or (lambda x: x)
    
    def build(self, input_shape):
        F = input_shape[-1]
        # linear transform W
        self.W = self.add_weight(shape=(F, self.out_features), initializer='glorot_uniform', name='W')
        # attention vector a (for concatenated features)
        self.a = self.add_weight(shape=(2 * self.out_features, 1), initializer='glorot_uniform', name='a')
        super().build(input_shape)
    
    def call(self, x, edge_index):
        """
        x: (N, F)
        edge_index: (2, E)
        returns: (N, out_features)
        """
        N = tf.shape(x)[0]
        h = tf.matmul(x, self.W)  # (N, out_features)
        # gather edge features
        src = edge_index[0]
        dst = edge_index[1]
        h_src = tf.gather(h, src)  # (E, out)
        h_dst = tf.gather(h, dst)  # (E, out)
        # concat and compute attention scores
        concat = tf.concat([h_src, h_dst], axis=1)  # (E, 2*out)
        e = tf.squeeze(tf.nn.leaky_relu(tf.matmul(concat, self.a)), axis=1)  # (E,)
        # for numerical stability, group by destination node for softmax
        # compute softmax per destination node
        # we need per-dst indices
        dst_int = dst.numpy() if isinstance(dst, tf.Tensor) else np.array(dst)
        e_np = e.numpy()
        # build per-destination softmax
        attn = np.zeros_like(e_np, dtype=np.float32)
        # group indices by dst
        groups = {}
        for idx_e, d in enumerate(dst_int):
            groups.setdefault(int(d), []).append(idx_e)
        for d, idx_list in groups.items():
            scores = e_np[idx_list]
            exp = np.exp(scores - np.max(scores))
            soft = exp / (np.sum(exp) + 1e-16)
            for ii, val in enumerate(idx_list):
                attn[val] = soft[ii]
        attn_tf = tf.convert_to_tensor(attn.reshape(-1, 1), dtype=tf.float32)  # (E,1)
        # message passing: multiply h_src by attn and sum per destination node
        messages = h_src * attn_tf  # (E, out)
        # aggregate messages per destination
        aggregated = tf.zeros((N, self.out_features), dtype=tf.float32)
        # use numpy gather-add to accumulate
        agg_np = aggregated.numpy()
        src_np = src.numpy() if isinstance(src, tf.Tensor) else np.array(src)
        dst_np = dst.numpy() if isinstance(dst, tf.Tensor) else np.array(dst)
        messages_np = messages.numpy()
        for e_i, d in enumerate(dst_np):
            agg_np[int(d)] += messages_np[e_i]
        aggregated = tf.convert_to_tensor(agg_np)
        return self.activation(aggregated)

# ---------------- GNN model (2-layer) ----------------
class SmallGATModel(Model):
    def __init__(self, hidden_dim=GNN_HIDDEN, out_dim=GNN_OUTPUT):
        super().__init__()
        self.gat1 = SimpleGATLayer(hidden_dim, activation=tf.nn.elu)
        self.gat2 = SimpleGATLayer(out_dim, activation=None)  # output scalar per node

    def call(self, x, edge_index):
        h1 = self.gat1(x, edge_index)  # (N, hidden)
        out = self.gat2(h1, edge_index)  # (N, out_dim)
        return out  # (N, out_dim)

# ---------------- training & utility functions ----------------
def train_gnn(X, y, edge_index, test_ratio=TEST_RATIO, epochs=EPOCHS):
    """
    X: (N, F) node features
    y: (N,) target scalars
    edge_index: (2, E) adjacency edges
    Trains GNN to predict y from X and graph structure.
    """
    N = X.shape[0]
    # train/test split by node
    idx = np.arange(N)
    train_idx, test_idx = train_test_split(idx, test_size=test_ratio, random_state=SEED)
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # standardize X features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    X_s = scaler.transform(X)

    # convert to tensors
    X_tensor = tf.convert_to_tensor(X_s, dtype=tf.float32)
    edge_index_tf = tf.convert_to_tensor(edge_index, dtype=tf.int32)

    model = SmallGATModel()
    optimizer = optimizers.Adam(learning_rate=LR)
    loss_fn = losses.MeanSquaredError()

    # training loop (per epoch: compute predictions for all nodes and compute loss only on train nodes)
    for ep in range(1, epochs+1):
        with tf.GradientTape() as tape:
            preds = model(X_tensor, edge_index_tf)  # (N,1)
            preds_vec = tf.squeeze(preds, axis=1)  # (N,)
            loss_val = loss_fn(y_train, tf.gather(preds_vec, train_idx))
        grads = tape.gradient(loss_val, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if ep % 10 == 0 or ep == 1 or ep == epochs:
            # evaluate
            preds_np = preds_vec.numpy()
            train_mse = np.mean((preds_np[train_idx] - y_train)**2)
            test_mse = np.mean((preds_np[test_idx] - y_test)**2)
            print(f"Epoch {ep:03d} Train MSE: {train_mse:.6f} Test MSE: {test_mse:.6f}")

    # final model and scaler returned
    return model, scaler, train_idx, test_idx

def neighbor_selection_from_attention(model: SmallGATModel, X_tensor: tf.Tensor, edge_index: np.ndarray, node_id: int, top_k: int):
    """
    Use the first-layer attention scores encoded in SimpleGATLayer to select top-k neighbors.
    NOTE: SimpleGATLayer stores no persistent edge attention; recompute attentions similarly to call().
    We replicate the attention computation in SimpleGATLayer.gather style to recover attention weights.
    """
    # Recompute h after first linear transform to extract attention scores
    # We'll access layer weights
    W1 = model.gat1.W.numpy()  # (F, hidden)
    a1 = model.gat1.a.numpy()  # (2*hidden, 1)
    h = X_tensor.numpy().dot(W1)  # (N, hidden)
    src = edge_index[0]
    dst = edge_index[1]
    concat = np.concatenate([h[src], h[dst]], axis=1)  # (E, 2*hidden)
    e = np.squeeze(np.maximum(0.01 * concat.dot(a1), -1000))  # leaky_relu approx with slope 0.01
    # we need attention scores for edges incoming to node_id
    incoming_idx = np.where(dst == node_id)[0]
    if incoming_idx.size == 0:
        return []
    scores = e[incoming_idx]
    # softmax over incoming edges
    exp = np.exp(scores - np.max(scores))
    soft = exp / (np.sum(exp) + 1e-16)
    neigh_src = src[incoming_idx]
    ranked = sorted(zip(neigh_src.tolist(), soft.tolist()), key=lambda x: x[1], reverse=True)
    topk = [int(t[0]) for t in ranked[:top_k]]
    return topk

# ---------------- main runnable demo ----------------
def main():
    print("Loading cluster-derived node features...")
    X, y, keys = load_cluster_node_features(DATA_DIR, INPUT_DIM, max_nodes=NUM_CLIENTS)
    print(f"Loaded {X.shape[0]} nodes (files) with input dim {X.shape[1]}.")

    print("Building k-NN graph...")
    neighbors, edge_index = build_knn_graph(X, k=K_NEIGHBORS)
    print(f"Constructed graph with {edge_index.shape[1]} directed edges (both directions included).")

    print("Training GNN model...")
    model, scaler, train_idx, test_idx = train_gnn(X, y, edge_index, test_ratio=TEST_RATIO, epochs=EPOCHS)

    # show neighbor selection for a few nodes
    X_tensor = tf.convert_to_tensor(scaler.transform(X), dtype=tf.float32)
    for nid in range(min(5, X.shape[0])):
        topk = neighbor_selection_from_attention(model, X_tensor, edge_index, node_id=nid, top_k=TOP_K_NEIGHBORS_SELECT)
        print(f"Node {nid} ({keys[nid]}): selected neighbors -> {topk}")

    # compute resource allocation predictions
    preds = model(X_tensor, tf.convert_to_tensor(edge_index, dtype=tf.int32)).numpy().squeeze()
    print("Sample resource allocation predictions (first 10 nodes):")
    for i in range(min(10, len(preds))):
        print(f" Node {i}: target={y[i]:.3f} pred={preds[i]:.3f}")

    # evaluation MSE on test nodes
    mse = np.mean((preds[test_idx] - y[test_idx])**2)
    print(f"Final Test MSE: {mse:.6f}")

if __name__ == "__main__":
    main()

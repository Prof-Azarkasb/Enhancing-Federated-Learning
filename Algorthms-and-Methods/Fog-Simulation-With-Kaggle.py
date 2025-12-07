"""
Fog Simulation Environment with Real Task Dataset
Features:
- 50 fog nodes with heterogeneous CPU, RAM, energy budgets.
- Network topology modeled as a graph using NetworkX.
- Edge latency sampled from Normal(10ms, 5ms) and clipped to positive.
- Tasks represented as 32-dimensional feature vectors (from Kaggle Google Cluster 2019 dataset).
- Local training simulated as small numeric updates to local model vectors.
- Local steps per node total 1200; synchronization occurs every 100 steps -> 12 global rounds.
- At each sync, only top-k influential nodes upload; global aggregation weighted by influence.
- Reproducible via random seed.
python fog_simulation_with_kaggle.py
"""

import random
import itertools
from typing import List, Dict
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import MinMaxScaler

# -------------------------
# Configuration parameters
# -------------------------
SEED = 42
NUM_NODES = 50
TASK_FEATURE_DIM = 32
LOCAL_TOTAL_STEPS = 1200
SYNC_INTERVAL = 100             # sync every 100 local steps
GLOBAL_ROUNDS = LOCAL_TOTAL_STEPS // SYNC_INTERVAL  # should be 12
TOP_K_INFLUENCERS = 2           # top-2 strategy
TASKS_PER_STEP = 1              # tasks processed per local step (simple)
EDGE_PROB = 0.10                # prob that an edge exists between node pairs
LATENCY_MEAN_MS = 10.0
LATENCY_STD_MS = 5.0
CPU_MIN_GHZ = 1.0
CPU_MAX_GHZ = 2.5
RAM_MIN_GB = 0.5
RAM_MAX_GB = 4.0
ENERGY_MIN_mAH = 1500
ENERGY_MAX_mAH = 3500
NUM_TASKS = 2000                # total tasks to sample initially
REPEAT_EXPERIMENTS = 1

random.seed(SEED)
np.random.seed(SEED)

# -------------------------
# Load and preprocess dataset
# -------------------------
DATASET_PATH = "google_cluster_trace.csv"  # 

df = pd.read_csv(DATASET_PATH)

# Ensure required fields exist
for field in ["priority", "time", "start_time", "end_time", "assigned_memory"]:
    if field not in df.columns:
        df[field] = 0

# Derived fields
df["TaskID"] = df.index
df["ArrivalTime"] = pd.to_numeric(df["time"], errors="coerce").fillna(0)
df["CompletionTime"] = pd.to_numeric(df["end_time"], errors="coerce").fillna(0)
df["StartTime"] = pd.to_numeric(df["start_time"], errors="coerce").fillna(0)
df["WaitingTime"] = (df["StartTime"] - df["ArrivalTime"]).clip(lower=0)
df["ProcessingTime"] = (df["CompletionTime"] - df["StartTime"]).clip(lower=0)
df["Delay"] = df["ProcessingTime"]
df["ResourceRequirements"] = pd.to_numeric(df["assigned_memory"], errors="coerce").fillna(0)
df["TaskStatus"] = pd.to_numeric(df.get("failed", 0), errors="coerce").fillna(0)

# Composite features
df["ResReq_Proc"] = df["ResourceRequirements"] + df["ProcessingTime"]
df["Status_Priority"] = df["TaskStatus"] + df["priority"]
df["Proc_Delay"] = df["ProcessingTime"] + df["Delay"]
df["Proc_Wait"] = df["ProcessingTime"] + df["WaitingTime"]

# Final feature selection
feature_cols = [
    "TaskID", "priority", "ArrivalTime", "CompletionTime", "WaitingTime",
    "ProcessingTime", "ResourceRequirements", "TaskStatus", "Delay",
    "ResReq_Proc", "Status_Priority", "Proc_Delay", "Proc_Wait"
]

df_final = df[feature_cols].copy()

# Normalize (except TaskID which we keep as integer ID)
scaler = MinMaxScaler()
df_scaled = df_final.copy()
df_scaled.iloc[:, 1:] = scaler.fit_transform(df_final.iloc[:, 1:])

# Iterator for sequential sampling
task_iter = itertools.cycle(df_scaled.values)

def make_task_vector():
    """Return one task feature vector (32-dim) from Kaggle dataset."""
    vec = next(task_iter)
    return vec.astype(float)

# -------------------------
# Utility functions
# -------------------------
def clipped_normal(mean: float, std: float, lower: float = 0.1):
    v = random.gauss(mean, std)
    return max(v, lower)

# -------------------------
# Fog node class
# -------------------------
class FogNode:
    def __init__(self, node_id: int, cpu_ghz: float, ram_gb: float, energy_mAh: int, seed: int = None):
        self.id = node_id
        self.cpu = cpu_ghz
        self.ram = ram_gb
        self.energy = energy_mAh
        self.model = np.zeros(TASK_FEATURE_DIM, dtype=float)
        self.local_step = 0
        self.local_tasks: List[np.ndarray] = []
        if seed is not None:
            random.seed(seed + node_id)

    def assign_tasks(self, tasks: List[np.ndarray]):
        self.local_tasks.extend(tasks)

    def simulate_local_update(self, steps: int = 1):
        for _ in range(steps):
            if not self.local_tasks:
                self.model *= 0.999
            else:
                task = random.choice(self.local_tasks)
                complexity = 0.5 + task[1]  # priority as complexity factor
                step_size = 0.01 * (self.cpu / CPU_MAX_GHZ) / complexity
                grad = (task - self.model)
                self.model += step_size * grad
                self.energy = max(0, self.energy - int(0.0001 * complexity * 1000))
            self.local_step += 1

    def get_model(self) -> np.ndarray:
        return self.model.copy()

# -------------------------
# Simulation environment
# -------------------------
class FogSimulation:
    def __init__(self, seed: int = SEED):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.G = nx.Graph()
        self.nodes: Dict[int, FogNode] = {}
        self.global_model = np.zeros(TASK_FEATURE_DIM, dtype=float)
        self.total_communication_bytes = 0
        self._build_nodes()
        self._build_edges()
        self.task_pool = [make_task_vector() for _ in range(NUM_TASKS)]
        self._distribute_tasks_stratified()

    def _build_nodes(self):
        for i in range(NUM_NODES):
            cpu = random.uniform(CPU_MIN_GHZ, CPU_MAX_GHZ)
            ram = random.uniform(RAM_MIN_GB, RAM_MAX_GB)
            energy = random.randint(ENERGY_MIN_mAH, ENERGY_MAX_mAH)
            self.G.add_node(i, cpu=cpu, ram=ram, energy=energy)
            self.nodes[i] = FogNode(i, cpu_ghz=cpu, ram_gb=ram, energy_mAh=energy, seed=self.seed)

    def _build_edges(self):
        for i in range(NUM_NODES):
            for j in range(i + 1, NUM_NODES):
                if random.random() < EDGE_PROB:
                    latency = clipped_normal(LATENCY_MEAN_MS, LATENCY_STD_MS, lower=1.0)
                    self.G.add_edge(i, j, latency=latency)
        components = list(nx.connected_components(self.G))
        if len(components) > 1:
            comp_list = components
            for idx in range(len(comp_list) - 1):
                a = random.choice(list(comp_list[idx]))
                b = random.choice(list(comp_list[idx + 1]))
                latency = clipped_normal(LATENCY_MEAN_MS, LATENCY_STD_MS, lower=1.0)
                self.G.add_edge(a, b, latency=latency)

    def _distribute_tasks_stratified(self):
        for idx, task in enumerate(self.task_pool):
            node_id = idx % NUM_NODES
            self.nodes[node_id].assign_tasks([task])

    def compute_influence_scores(self) -> Dict[int, float]:
        deg = nx.degree_centrality(self.G)
        bet = nx.betweenness_centrality(self.G, normalized=True)
        clo = nx.closeness_centrality(self.G)
        scores = {}
        for n in self.G.nodes():
            raw = 0.4 * deg.get(n, 0.0) + 0.4 * bet.get(n, 0.0) + 0.2 * clo.get(n, 0.0)
            scores[n] = raw
        total = sum(scores.values()) + 1e-12
        for k in scores:
            scores[k] /= total
        return scores

    def select_top_k(self, scores: Dict[int, float], k: int = TOP_K_INFLUENCERS) -> List[int]:
        sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [n for n, s in sorted_nodes[:k]]

    def aggregate_models(self, selected_node_ids: List[int], scores: Dict[int, float]) -> np.ndarray:
        if not selected_node_ids:
            return self.global_model.copy()
        agg = np.zeros_like(self.global_model)
        weight_sum = 0.0
        for nid in selected_node_ids:
            w = scores.get(nid, 0.0)
            agg += w * self.nodes[nid].get_model()
            weight_sum += w
            bytes_sent = TASK_FEATURE_DIM * 8
            self.total_communication_bytes += bytes_sent
        if weight_sum > 0:
            agg /= weight_sum
        else:
            agg = self.global_model.copy()
        return agg

    def broadcast_global_model(self, new_model: np.ndarray, selected_node_ids: List[int]):
        for nid in range(NUM_NODES):
            bytes_recv = TASK_FEATURE_DIM * 8 * 0.6
            self.total_communication_bytes += int(bytes_recv)
            self.nodes[nid].model = new_model.copy()

    def simulate(self, verbose: bool = True):
        if verbose:
            print(f"Starting simulation: {NUM_NODES} nodes, {GLOBAL_ROUNDS} global rounds, sync every {SYNC_INTERVAL} steps.")
            print(f"Initial total tasks: {len(self.task_pool)}")
        for round_idx in range(GLOBAL_ROUNDS):
            if verbose:
                print(f"\n=== Global Round {round_idx + 1}/{GLOBAL_ROUNDS} ===")
            for nid, node in self.nodes.items():
                node.simulate_local_update(steps=SYNC_INTERVAL)
            scores = self.compute_influence_scores()
            selected = self.select_top_k(scores, k=TOP_K_INFLUENCERS)
            if verbose:
                print(f"Selected influencer nodes for upload: {selected}")
            new_global = self.aggregate_models(selected, scores)
            ground_truth = np.mean(self.task_pool, axis=0)
            sim_before = np.dot(self.global_model, ground_truth) / (np.linalg.norm(self.global_model) * np.linalg.norm(ground_truth) + 1e-12)
            sim_after = np.dot(new_global, ground_truth) / (np.linalg.norm(new_global) * np.linalg.norm(ground_truth) + 1e-12)
            if verbose:
                print(f"Similarity to ground truth: before={sim_before:.4f}, after={sim_after:.4f}")
            self.global_model = new_global
            self.broadcast_global_model(new_global, selected)

        if verbose:
            total_kb = self.total_communication_bytes / 1024.0
            print("\n=== Simulation complete ===")
            print(f"Total communication (simulated): {total_kb:.2f} KiB")
            avg_energy = np.mean([n.energy for n in self.nodes.values()])
            avg_cpu = np.mean([n.cpu for n in self.nodes.values()])
            print(f"Avg remaining energy: {avg_energy:.1f} mAh, Avg CPU: {avg_cpu:.2f} GHz")

# -------------------------
# Run main simulation
# -------------------------
def main():
    sim = FogSimulation(seed=SEED)
    sim.simulate(verbose=True)

if __name__ == "__main__":
    main()

"""
drfl_google_cluster.py

Deep Reinforcement Federated Learning (DR-FL) adapted to Google 2019 Cluster sample.
- Task (environment): given sliding-window of resource usage (INPUT_LEN), choose allocation action in [0,1]
  Target = next-step resource usage (normalized). Reward = - (action - target)^2 (maximize negative MSE)
- Local agent: simple DDPG-like agent (Actor + Critic) implemented with TensorFlow 2.x
- Server: Federated averaging of actor & critic weights after each federated round
"""

import os
import glob
import random
import time
from collections import deque
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Optional psutil for resource monitoring (install if desired)
try:
    import psutil
except Exception:
    psutil = None

# ---------------- Config ----------------
DATA_DIR = "./data"
NUM_CLIENTS = 8             # number of pseudo-clients to simulate (will take up to this many CSVs)
INPUT_LEN = 20              # sliding-window length (state dimension)
LOCAL_EPISODES = 20         # number of local episodes per federated round (each episode samples from client dataset)
LOCAL_STEPS_PER_EPISODE = 50
BATCH_SIZE = 64
GLOBAL_ROUNDS = 12
LR_ACTOR = 1e-3
LR_CRITIC = 1e-3
GAMMA = 0.99
TAU = 0.01                  # soft update rate for target networks
REPLAY_CAPACITY = 20000
OU_THETA = 0.15
OU_SIGMA = 0.2
FAILURE_RATE = 0.12         # sim client fails to send update
TEST_HOLDOUT_RATIO = 0.15
RANDOM_SEED = 42

# reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ---------------- Data utilities ----------------
def list_csv_files(data_dir: str) -> List[str]:
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}. Download Kaggle dataset and place CSVs there.")
    return sorted(files)


def series_to_windows(path: str, input_len: int = INPUT_LEN, step: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read CSV file, pick first numeric column, normalize per-file to [0,1], build sliding windows.
    Returns X (N_samples x input_len), y (N_samples,) where y is next-step value (regression target).
    """
    df = pd.read_csv(path, low_memory=True)
    numeric = df.select_dtypes(include=[np.number]).fillna(0.0)
    if numeric.shape[1] == 0:
        return np.empty((0, input_len), dtype=np.float32), np.empty((0,), dtype=np.float32)
    col = numeric.columns[0]
    series = numeric[col].values.astype(np.float32)
    if series.size <= input_len:
        return np.empty((0, input_len), dtype=np.float32), np.empty((0,), dtype=np.float32)
    # per-file min-max normalize
    if series.max() > series.min():
        series = (series - series.min()) / (series.max() - series.min())
    else:
        series = np.zeros_like(series)
    X_list, y_list = [], []
    for i in range(0, len(series) - input_len, step):
        X_list.append(series[i:i + input_len])
        y_list.append(series[i + input_len])
    if not X_list:
        return np.empty((0, input_len), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return np.stack(X_list, axis=0).astype(np.float32), np.array(y_list, dtype=np.float32)


def build_clients_and_test(files: List[str], num_clients: int = NUM_CLIENTS, input_len: int = INPUT_LEN
                           ) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Tuple[np.ndarray, np.ndarray]]:
    """
    Build per-client datasets from CSV files, with some files held out as global test set.
    Returns:
      clients_data: list of (X_client, y_client)
      (X_test, y_test)
    """
    files = files.copy()
    random.shuffle(files)
    n_holdout = max(1, int(len(files) * TEST_HOLDOUT_RATIO))
    holdout = files[:n_holdout]
    pool = files[n_holdout:]

    # build test
    X_test_list, y_test_list = [], []
    for f in holdout:
        Xf, yf = series_to_windows(f, input_len)
        if Xf.shape[0] > 0:
            X_test_list.append(Xf); y_test_list.append(yf)
    if X_test_list:
        X_test = np.vstack(X_test_list); y_test = np.concatenate(y_test_list)
    else:
        X_test = np.empty((0,input_len), dtype=np.float32); y_test = np.empty((0,), dtype=np.float32)

    # build per-file windows for pool
    per_file = []
    for f in pool:
        Xf, yf = series_to_windows(f, input_len)
        if Xf.shape[0] > 0:
            per_file.append((Xf, yf))

    if not per_file:
        raise RuntimeError("No usable data windows extracted from files. Check dataset.")

    # distribute files round-robin to clients
    clients_chunks = [ ([],[]) for _ in range(num_clients) ]
    for i,(Xf,yf) in enumerate(per_file):
        idx = i % num_clients
        clients_chunks[idx][0].append(Xf); clients_chunks[idx][1].append(yf)

    clients = []
    for X_chunks, y_chunks in clients_chunks:
        if X_chunks:
            Xc = np.vstack(X_chunks); yc = np.concatenate(y_chunks)
        else:
            # fallback small random dataset
            Xc = np.random.rand(50, input_len).astype(np.float32)
            yc = np.random.rand(50).astype(np.float32)
        clients.append((Xc, yc))
    return clients, (X_test, y_test)


# ---------------- DDPG-like agent ----------------
class OUActionNoise:
    def __init__(self, mean, std_dev, theta=OU_THETA, dt=1e-2, x0=None):
        self.theta = theta
        self.mean = np.array(mean, dtype=np.float32)
        self.std_dev = np.array(std_dev, dtype=np.float32)
        self.dt = dt
        self.x_prev = x0 if x0 is not None else np.zeros_like(self.mean)
    def __call__(self):
        x = (self.x_prev + self.theta * (self.mean - self.x_prev) * self.dt +
             self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape))
        self.x_prev = x
        return x
    def reset(self):
        self.x_prev = np.zeros_like(self.mean)


class ReplayBuffer:
    def __init__(self, capacity=REPLAY_CAPACITY):
        self.buffer = deque(maxlen=capacity)
    def add(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        idx = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        s, a, r, s2, d = zip(*[self.buffer[i] for i in idx])
        return (np.array(s), np.array(a), np.array(r, dtype=np.float32), np.array(s2), np.array(d))
    def __len__(self):
        return len(self.buffer)


def build_actor(state_dim, action_dim=1):
    inp = tf.keras.Input(shape=(state_dim,))
    x = tf.keras.layers.Dense(128, activation='relu')(inp)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    out = tf.keras.layers.Dense(action_dim, activation='tanh')(x)  # output in [-1,1]
    model = tf.keras.Model(inp, out)
    return model


def build_critic(state_dim, action_dim=1):
    s_in = tf.keras.Input(shape=(state_dim,))
    a_in = tf.keras.Input(shape=(action_dim,))
    x = tf.keras.layers.Concatenate()([s_in, a_in])
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    out = tf.keras.layers.Dense(1, activation='linear')(x)
    model = tf.keras.Model([s_in, a_in], out)
    return model


class DDPGAgent:
    def __init__(self, state_dim, action_dim=1, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor = build_actor(state_dim, action_dim)
        self.critic = build_critic(state_dim, action_dim)
        self.target_actor = build_actor(state_dim, action_dim)
        self.target_critic = build_critic(state_dim, action_dim)
        # compile optimizers
        self.actor_optimizer = tf.keras.optimizers.Adam(lr_actor)
        self.critic_optimizer = tf.keras.optimizers.Adam(lr_critic)
        # initialize targets equal to originals
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())
        # replay and noise
        self.replay = ReplayBuffer()
        self.noise = OUActionNoise(mean=np.zeros((action_dim,)), std_dev=OU_SIGMA * np.ones((action_dim,)))
    def get_action(self, state, add_noise=True):
        s = np.expand_dims(state.astype(np.float32), axis=0)
        a = self.actor(s).numpy()[0]
        if add_noise:
            a = a + self.noise()
        # scale tanh [-1,1] to [0,1]
        return np.clip((a + 1.0) / 2.0, 0.0, 1.0)
    def remember(self, s, a, r, s2, done=False):
        self.replay.add(s, a, r, s2, done)
    def soft_update(self):
        # tau blending
        aw = self.actor.get_weights(); taw = self.target_actor.get_weights()
        cw = self.critic.get_weights(); tcw = self.target_critic.get_weights()
        new_taw = [TAU * w + (1 - TAU) * tw for w, tw in zip(aw, taw)]
        new_tcw = [TAU * w + (1 - TAU) * tw for w, tw in zip(cw, tcw)]
        self.target_actor.set_weights(new_taw); self.target_critic.set_weights(new_tcw)
    def train_from_replay(self, batch_size=BATCH_SIZE):
        sample = self.replay.sample(batch_size)
        if sample is None:
            return
        s_batch, a_batch, r_batch, s2_batch, done_batch = sample
        # Critic update
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(s2_batch)
            target_q = tf.squeeze(self.target_critic([s2_batch, target_actions]), axis=1)
            y = r_batch + GAMMA * (1.0 - done_batch) * target_q
            q_values = tf.squeeze(self.critic([s_batch, a_batch]), axis=1)
            critic_loss = tf.reduce_mean(tf.square(y - q_values))
        grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))
        # Actor update (policy gradient via critic)
        with tf.GradientTape() as tape2:
            actions_pred = self.actor(s_batch)
            actor_loss = -tf.reduce_mean(self.critic([s_batch, actions_pred]))
        grads2 = tape2.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads2, self.actor.trainable_variables))
        # soft update targets
        self.soft_update()


# ---------------- Federated aggregation ----------------
def average_weights(list_of_weightlists: List[List[np.ndarray]]) -> List[np.ndarray]:
    """
    FedAvg: average corresponding numpy arrays across models.
    list_of_weightlists: list of model.get_weights() outputs
    """
    if not list_of_weightlists:
        return None
    avg = []
    n_models = len(list_of_weightlists)
    for layer_idx in range(len(list_of_weightlists[0])):
        layer_stack = np.stack([w[layer_idx] for w in list_of_weightlists], axis=0)
        avg.append(np.mean(layer_stack, axis=0))
    return avg


# ---------------- DR-FL orchestration ----------------
class DRFL:
    def __init__(self, clients_data: List[Tuple[np.ndarray, np.ndarray]], test_data: Tuple[np.ndarray, np.ndarray]):
        self.clients = []
        for i,(X,y) in enumerate(clients_data):
            agent = DDPGAgent(state_dim=X.shape[1], action_dim=1)
            # store dataset on agent
            agent.dataset_X = X
            agent.dataset_y = y
            agent.client_id = i
            self.clients.append(agent)
        self.test_X, self.test_y = test_data
        # initialize global actor/critic as average of initial clients
        self.global_actor_weights = average_weights([c.actor.get_weights() for c in self.clients])
        self.global_critic_weights = average_weights([c.critic.get_weights() for c in self.clients])
        # set global to each client
        for c in self.clients:
            c.actor.set_weights(self.global_actor_weights)
            c.critic.set_weights(self.global_critic_weights)
    def local_train_agent(self, agent: DDPGAgent):
        """
        Train agent locally using its dataset as environment.
        Each episode samples random starting indices and runs LOCAL_STEPS_PER_EPISODE steps.
        Reward = - (action - target)^2
        """
        X, y = agent.dataset_X, agent.dataset_y
        n_samples = X.shape[0]
        if n_samples < 2:
            return
        agent.noise.reset()
        for ep in range(LOCAL_EPISODES):
            # random start
            idx = np.random.randint(0, max(1, n_samples - LOCAL_STEPS_PER_EPISODE - 1))
            for t in range(LOCAL_STEPS_PER_EPISODE):
                s = X[idx + t]
                target = y[idx + t]
                a = agent.get_action(s, add_noise=True)  # shape (1,) array
                # reward (negative squared error)
                r = -float((a[0] - target) ** 2)
                s2 = X[idx + t + 1] if (idx + t + 1) < n_samples else X[-1]
                done = False
                # store experience
                agent.remember(s, np.array(a, dtype=np.float32), r, s2, done)
                # train from replay occasionally
                agent.train_from_replay(batch_size=BATCH_SIZE)
            # end episode
        # after local training, return actor & critic weights
        return agent.actor.get_weights(), agent.critic.get_weights()
    def federated_round(self):
        """
        Each client trains locally; we collect weights from clients that successfully send updates (simulate failure)
        Then perform FedAvg on actor & critic weights and broadcast.
        """
        collected_actor = []
        collected_critic = []
        for agent in self.clients:
            if random.random() < FAILURE_RATE:
                print(f"Client {agent.client_id} failed to send update (simulated).")
                continue
            res = self.local_train_agent(agent)
            if res is None:
                continue
            aw, cw = res
            collected_actor.append(aw); collected_critic.append(cw)
            print(f"Client {agent.client_id} finished local training; replay buffer size={len(agent.replay)}")
        if collected_actor:
            new_actor = average_weights(collected_actor)
            new_critic = average_weights(collected_critic)
            # update global and broadcast
            self.global_actor_weights = new_actor
            self.global_critic_weights = new_critic
            for agent in self.clients:
                agent.actor.set_weights(self.global_actor_weights)
                agent.critic.set_weights(self.global_critic_weights)
            return True
        return False
    def evaluate_global_policy(self):
        """
        Use global actor to predict on test set and compute MSE vs target
        """
        if self.test_X.shape[0] == 0:
            return None
        # ensure actor weights set in a temporary model
        actor_model = build_actor(self.test_X.shape[1], 1)
        actor_model.set_weights(self.global_actor_weights)
        preds = actor_model.predict(self.test_X, batch_size=128).reshape(-1)
        preds = np.clip((preds + 1.0) / 2.0, 0.0, 1.0)  # scale tanh-> [0,1]
        mse = float(np.mean((preds - self.test_y) ** 2))
        return mse


# ---------------- Main script ----------------
def main():
    files = list_csv_files(DATA_DIR)
    clients_data, test = build_clients_and_test(files, num_clients=NUM_CLIENTS, input_len=INPUT_LEN)
    # optionally global standardization (we'll standardize features across all clients & test)
    all_X = np.vstack([X for X,_ in clients_data] + ([test[0]] if test[0].shape[0] > 0 else []))
    scaler = StandardScaler()
    scaler.fit(all_X)
    clients_data = [(scaler.transform(X).astype(np.float32), y.astype(np.float32)) for X,y in clients_data]
    test_X = scaler.transform(test[0]).astype(np.float32) if test[0].shape[0] > 0 else test[0]
    test_y = test[1].astype(np.float32)

    drfl = DRFL(clients_data, (test_X, test_y))

    print(f"Starting DR-FL: {len(drfl.clients)} clients, global rounds={GLOBAL_ROUNDS}")
    for r in range(1, GLOBAL_ROUNDS + 1):
        print(f"\n=== Federated Round {r}/{GLOBAL_ROUNDS} ===")
        updated = drfl.federated_round()
        if not updated:
            print("No updates collected this round.")
        mse = drfl.evaluate_global_policy()
        if mse is not None:
            print(f"Global policy test MSE after round {r}: {mse:.6f}")
        else:
            print("No test set available for evaluation.")
        # optional resource usage report
        if psutil:
            print(f"System CPU%: {psutil.cpu_percent(interval=0.5)} Memory%: {psutil.virtual_memory().percent}")
    print("\nDR-FL simulation finished.")


if __name__ == "__main__":
    main()

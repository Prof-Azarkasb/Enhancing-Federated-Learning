import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import random
from collections import deque
import time  # For measuring communication latency
import psutil  # For measuring resource consumption

# Define constants
STATE_SIZE = 10  # Example state size (can be adjusted based on task offloading model)
ACTION_SIZE = 2  # Task offloading and resource allocation
BATCH_SIZE = 64
GAMMA = 0.99  # Discount factor
TAU = 0.005  # Target network update rate
LR_ACTOR = 0.001
LR_CRITIC = 0.002
MAX_BUFFER = 1000000

# Define the Actor (Policy) Network
def build_actor(state_size, action_size):
    """Creates the Actor network for policy approximation in MADDPG."""
    inputs = layers.Input(shape=(state_size,))
    out = layers.Dense(128, activation="relu")(inputs)
    out = layers.Dense(128, activation="relu")(out)
    outputs = layers.Dense(action_size, activation="tanh")(out)
    model = tf.keras.Model(inputs, outputs)
    return model

# Define the Critic (Value) Network
def build_critic(state_size, action_size):
    """Creates the Critic network to evaluate actions."""
    state_input = layers.Input(shape=(state_size,))
    action_input = layers.Input(shape=(action_size,))
    concat = layers.Concatenate()([state_input, action_input])
    
    out = layers.Dense(128, activation="relu")(concat)
    out = layers.Dense(128, activation="relu")(out)
    outputs = layers.Dense(1)(out)  # Q-value prediction
    model = tf.keras.Model([state_input, action_input], outputs)
    return model

# Ornstein-Uhlenbeck noise for exploration
class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_deviation = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mean - self.x_prev) * self.dt + self.std_deviation * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x_initial if self.x_initial is not None else np.zeros_like(self.mean)

# Replay buffer to store experience tuples (state, action, reward, next_state)
class ReplayBuffer:
    def __init__(self, max_size=MAX_BUFFER):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

# Define the MADDPG agent
class MADDPGAgent:
    def __init__(self, state_size, action_size):
        # Actor and Critic Networks
        self.actor = build_actor(state_size, action_size)
        self.target_actor = build_actor(state_size, action_size)
        self.critic = build_critic(state_size, action_size)
        self.target_critic = build_critic(state_size, action_size)
        
        # Replay Buffer
        self.replay_buffer = ReplayBuffer()
        
        # Noise for exploration
        self.noise = OUActionNoise(mean=np.zeros(action_size), std_deviation=0.2 * np.ones(action_size))
        
        # Optimizers
        self.actor_optimizer = tf.keras.optimizers.Adam(LR_ACTOR)
        self.critic_optimizer = tf.keras.optimizers.Adam(LR_CRITIC)
        
        # Update target models
        self.update_target(self.target_actor, self.actor, tau=1.0)
        self.update_target(self.target_critic, self.critic, tau=1.0)
    
    def update_target(self, target_weights, weights, tau):
        """Updates target network parameters using soft update with rate tau."""
        for (a, b) in zip(target_weights.variables, weights.variables):
            a.assign(b * tau + a * (1 - tau))
    
    def act(self, state):
        """Selects an action based on the current policy (with added noise for exploration)."""
        state = np.expand_dims(state, axis=0)
        action = self.actor(state)
        return action.numpy()[0] + self.noise()
    
    def remember(self, state, action, reward, next_state):
        """Stores experience tuple in replay buffer."""
        self.replay_buffer.add((state, action, reward, next_state))
    
    def train(self, batch_size=BATCH_SIZE):
        """Trains the agent using sampled experiences from the replay buffer."""
        if len(self.replay_buffer.buffer) < batch_size:
            return
        
        # Sample experiences from buffer
        minibatch = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states = map(np.vstack, zip(*minibatch))
        
        # Critic update
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_states)
            target_q = self.target_critic([next_states, target_actions])
            q_values = rewards + GAMMA * target_q
            critic_loss = tf.reduce_mean(tf.square(q_values - self.critic([states, actions])))
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        
        # Actor update
        with tf.GradientTape() as tape:
            actions_pred = self.actor(states)
            actor_loss = -tf.reduce_mean(self.critic([states, actions_pred]))  # Maximize Q-value
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        # Soft update of target networks
        self.update_target(self.target_actor, self.actor, TAU)
        self.update_target(self.target_critic, self.critic, TAU)

# Define the Federated Learning Aggregator
class FederatedAggregator:
    def __init__(self, agents):
        self.agents = agents  # List of MADDPG agents
    
    def aggregate_models(self):
        """Aggregates the weights of the agents in the federated learning system."""
        avg_actor_weights = []
        avg_critic_weights = []
        
        # Initialize with zeros of the same shape as the weights
        for i in range(len(self.agents[0].actor.get_weights())):
            avg_actor_weights.append(np.zeros_like(self.agents[0].actor.get_weights()[i]))
            avg_critic_weights.append(np.zeros_like(self.agents[0].critic.get_weights()[i]))
        
        # Accumulate weights from each agent
        for agent in self.agents:
            actor_weights = agent.actor.get_weights()
            critic_weights = agent.critic.get_weights()
            for i in range(len(avg_actor_weights)):
                avg_actor_weights[i] += actor_weights[i]
                avg_critic_weights[i] += critic_weights[i]
        
        # Average the weights
        for i in range(len(avg_actor_weights)):
            avg_actor_weights[i] /= len(self.agents)
            avg_critic_weights[i] /= len(self.agents)
        
        # Update the global model with averaged weights
        for agent in self.agents:
            agent.actor.set_weights(avg_actor_weights)
            agent.critic.set_weights(avg_critic_weights)

# Define the Server for Federated Learning
class Server:
    def __init__(self, agents):
        self.aggregator = FederatedAggregator(agents)
    
    def update_global_model(self, client_weights):
        """
        Updates the global model with the aggregated weights from clients.
        Also measures communication latency.
        Args:
            client_weights (list of lists): List containing the model weights from all clients.
        """
        start_time = time.time()  # Start time for measuring latency
        self.aggregator.aggregate_models()
        end_time = time.time()  # End time for measuring latency
        
        latency = end_time - start_time  # Calculate latency
        print(f"Communication Latency: {latency:.4f} seconds")

    def federated_learning_with_failures(self, rounds=10, epochs=1, failure_rate=0.2):
        """
        Simulates federated learning with random client failures to evaluate robustness.
        """
        for rnd in range(rounds):
            print(f"Round {rnd + 1}/{rounds} of Federated Learning with Failures")
            
            client_weights = []
            for agent in self.aggregator.agents:
                # Simulate network failure for some clients
                if random.random() > failure_rate:
                    agent.train(epochs=epochs)
                    client_weights.append(agent.actor.get_weights())  # Get actor weights
                else:
                    print(f"Agent failed to communicate.")
            
            if client_weights:
                self.update_global_model(client_weights)
            print("\n")

# Define the clients (agents)
clients = [MADDPGAgent(STATE_SIZE, ACTION_SIZE) for _ in range(5)]

# Create the server to manage federated learning
server = Server(clients)

# Run federated learning with client failures
server.federated_learning_with_failures(rounds=5, epochs=5)

# Resource Consumption Monitoring
cpu_usage = psutil.cpu_percent(interval=1)
memory_usage = psutil.virtual_memory().percent
print(f"CPU Usage: {cpu_usage}%, Memory Usage: {memory_usage}%")

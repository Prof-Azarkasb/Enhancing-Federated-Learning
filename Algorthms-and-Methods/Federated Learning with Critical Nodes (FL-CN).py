import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers
import random
import time  # For measuring communication latency
import psutil  # For measuring resource consumption

# Simulation Parameters
NUM_CLIENTS = 10  # Number of clients in the federated learning system
CLIENT_FRACTION = 0.6  # Fraction of clients selected in each round
EPOCHS = 5  # Number of local epochs for each client
BATCH_SIZE = 32  # Batch size for local training
LEARNING_RATE = 0.01  # Learning rate for optimizers

# Parameters for critical task indexing and node selection
CRITICAL_NODE_THRESHOLD = 0.7  # Threshold to determine a critical node based on computational power/latency
MAX_CRITICAL_NODES = 5  # Maximum number of critical nodes that can be selected

# Critical Task Indexing Scheduler (CTIS) Implementation
class CriticalNode:
    """Represents a node in the federated learning environment with specific computational capabilities."""
    
    def __init__(self, data, labels, comp_power, latency):
        """Initializes a node with local data, computational power, and latency."""
        self.data = data
        self.labels = labels
        self.comp_power = comp_power  # Computational power of the node (higher is better)
        self.latency = latency  # Latency in processing tasks (lower is better)
        self.model = self.create_model()  # Each node has its own model for local training
        self.optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)  # Default optimizer
        
    def create_model(self):
        """Defines the local model architecture for each client (can be customized)."""
        model = tf.keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(784,)),
            layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer=self.optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def train(self, epochs=EPOCHS, batch_size=BATCH_SIZE):
        """Trains the local model on the node's data while measuring resource consumption."""
        start_cpu = psutil.cpu_percent(interval=None)
        start_memory = psutil.virtual_memory().percent
        
        self.model.fit(self.data, self.labels, epochs=epochs, batch_size=batch_size, verbose=0)
        
        end_cpu = psutil.cpu_percent(interval=None)
        end_memory = psutil.virtual_memory().percent
        
        print(f"Client - CPU Consumption: {end_cpu - start_cpu:.2f}%, Memory Consumption: {end_memory - start_memory:.2f}%")
    
    def get_weights(self):
        """Returns the local model weights after training."""
        return self.model.get_weights()
    
    def set_weights(self, weights):
        """Sets the model weights for this node."""
        self.model.set_weights(weights)
    
    def get_node_priority(self):
        """Calculates the priority score of a node based on computational power and latency."""
        return self.comp_power / (self.latency + 1e-9)  # To avoid division by zero

# Federated Learning Server with Critical Node Selection
class FederatedServer:
    """Simulates the federated learning process with critical node selection."""
    
    def __init__(self, nodes):
        """Initializes the server with a list of participating nodes."""
        self.nodes = nodes
        self.global_model = self.create_global_model()
    
    def create_global_model(self):
        """Creates the global model that will be shared across all nodes."""
        model = tf.keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(784,)),
            layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def critical_node_selection(self):
        """Selects critical nodes based on their computational power and latency."""
        priority_scores = [(node, node.get_node_priority()) for node in self.nodes]
        priority_scores.sort(key=lambda x: x[1], reverse=True)  # Sort nodes by priority (higher is better)
        
        critical_nodes = [node for node, score in priority_scores if score > CRITICAL_NODE_THRESHOLD]
        if len(critical_nodes) > MAX_CRITICAL_NODES:
            critical_nodes = critical_nodes[:MAX_CRITICAL_NODES]  # Limit to maximum allowed critical nodes
        return critical_nodes
    
    def update_global_model(self, client_weights):
        """
        Updates the global model with the aggregated weights from clients.
        Also measures communication latency.
        Args:
            client_weights (list of lists): List containing the model weights from all clients.
        """
        start_time = time.time()  # Start time for measuring latency
        aggregated_weights = self.aggregate_weights(client_weights)
        self.global_model.set_weights(aggregated_weights)
        end_time = time.time()  # End time for measuring latency
        
        latency = end_time - start_time  # Calculate latency
        print(f"Communication Latency: {latency:.4f} seconds")

    def federated_round(self):
        """Performs one round of federated learning by aggregating weights from critical nodes."""
        selected_nodes = self.critical_node_selection()
        if len(selected_nodes) < int(CLIENT_FRACTION * len(self.nodes)):
            additional_nodes = random.sample(self.nodes, int(CLIENT_FRACTION * len(self.nodes)) - len(selected_nodes))
            selected_nodes.extend(additional_nodes)
        
        # Step 1: Distribute global model weights to selected clients
        global_weights = self.global_model.get_weights()
        for node in selected_nodes:
            node.set_weights(global_weights)
        
        # Step 2: Each selected node performs local training
        for node in selected_nodes:
            node.train()
        
        # Step 3: Gather the updated weights from all selected nodes
        client_weights = [node.get_weights() for node in selected_nodes]
        
        # Step 4: Update global model with aggregated weights
        self.update_global_model(client_weights)
    
    def aggregate_weights(self, client_weights):
        """Aggregates weights from all participating nodes by averaging them."""
        new_weights = []
        for weights in zip(*client_weights):
            new_weights.append(np.mean(weights, axis=0))
        return new_weights
    
    def evaluate_global_model(self, test_data, test_labels):
        """Evaluates the global model on the test data."""
        loss, accuracy = self.global_model.evaluate(test_data, test_labels, verbose=0)
        return loss, accuracy

# Generate synthetic data for demonstration (replace with real datasets in practice)
def generate_synthetic_data(num_samples, input_shape, num_classes):
    """Generates synthetic data for testing purposes."""
    data = np.random.rand(num_samples, input_shape)
    labels = np.random.randint(0, num_classes, num_samples)
    return data, labels

# Robustness to Network Failure
def simulate_network_failures(clients, failure_rate=0.2):
    """Simulates network failure for clients."""
    active_clients = []
    for client in clients:
        if random.random() > failure_rate:
            active_clients.append(client)
        else:
            print(f"Client {client} failed to communicate.")
    return active_clients

# Simulation of Federated Learning with Critical Nodes
def run_federated_learning_with_critical_nodes(num_rounds, test_data, test_labels):
    """Runs federated learning simulation with critical node selection over multiple rounds."""
    
    # Generate synthetic clients (nodes) with different computational power and latency
    nodes = []
    for _ in range(NUM_CLIENTS):
        data, labels = generate_synthetic_data(600, 784, 10)
        comp_power = np.random.rand() * 2  # Random computational power between 0 and 2
        latency = np.random.rand()  # Random latency between 0 and 1
        nodes.append(CriticalNode(data, labels, comp_power, latency))
    
    # Initialize the server
    server = FederatedServer(nodes)
    
    # Federated learning rounds
    for round_num in range(num_rounds):
        print(f"Federated Round {round_num + 1}/{num_rounds}")
        
        # Simulate network failures
        active_nodes = simulate_network_failures(nodes)
        server.nodes = active_nodes  # Update server's nodes to only active ones
        
        server.federated_round()
        loss, accuracy = server.evaluate_global_model(test_data, test_labels)
        print(f"Global Model Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
    
# Example usage
if __name__ == "__main__":
    # Generate synthetic test data (replace with real test data in practice)
    test_data, test_labels = generate_synthetic_data(1000, 784, 10)
    
    # Run the federated learning simulation
    run_federated_learning_with_critical_nodes(num_rounds=5, test_data=test_data, test_labels=test_labels)

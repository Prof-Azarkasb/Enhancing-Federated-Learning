import numpy as np
from scipy.optimize import minimize
import tensorflow as tf
import time  # For measuring communication latency
import psutil  # For measuring resource consumption
import random

# Define the server class for managing global model and aggregation
class Server:
    def __init__(self, global_model):
        self.global_model = global_model  # Global model to be shared with clients
    
    # Aggregate the local models based on unpruned weights and masks
    def aggregate_models(self, client_models, masks):
        aggregated_model = np.zeros_like(self.global_model)
        count_non_zero = np.zeros_like(self.global_model)
        
        # Aggregate only unpruned parts of local models
        for model, mask in zip(client_models, masks):
            aggregated_model += model * mask
            count_non_zero += mask
        
        # Average the non-pruned weights to form the global model
        count_non_zero = np.where(count_non_zero == 0, 1, count_non_zero)  # Avoid division by zero
        self.global_model = aggregated_model / count_non_zero
    
    # Derive subnetworks based on masks for clients
    def derive_subnetwork(self, mask):
        return self.global_model * mask

    # Update the global model with the aggregated weights from clients and measure latency
    def update_global_model(self, client_weights):
        start_time = time.time()  # Start time for measuring latency
        aggregated_weights = self.aggregate_models(client_weights)
        self.global_model = aggregated_weights
        end_time = time.time()  # End time for measuring latency
        
        latency = end_time - start_time  # Calculate latency
        print(f"Communication Latency: {latency:.4f} seconds")


# Define the client class for training local models and pruning
class Client:
    def __init__(self, model, local_data, pruning_rate=0.2):
        self.model = model  # Local model for this client
        self.local_data = local_data  # Client-specific dataset
        self.pruning_rate = pruning_rate  # Pruning rate to control sparsity
        self.mask = np.ones_like(self.model)  # Initial mask is all ones (no pruning)

    # Function to apply ADMM weight pruning
    def admm_weight_pruning(self):
        """
        ADMM weight pruning involves solving a non-convex optimization problem using alternating
        direction method of multipliers (ADMM). We decompose the loss function into subproblems, 
        alternating between solving the weight update and sparse constraint optimization.
        """
        def loss_function(weights):
            # Define the loss function: cross-entropy + L2 regularization for sparse training
            predictions = self.local_data["X"] @ weights
            loss = np.mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.local_data["y"], logits=predictions))
            regularization = 0.01 * np.sum(weights ** 2)  # L2 regularization
            return loss + regularization
        
        # ADMM Optimization
        def admm_update(weights, z, u, rho=1.0):
            # Update rule for ADMM with regularization and Lagrangian multipliers
            weights_new = minimize(lambda w: loss_function(w) + rho * np.linalg.norm(w - z + u), weights).x
            z_new = np.maximum(0, weights_new + u - self.pruning_rate)  # Euclidean projection
            u_new = u + weights_new - z_new
            return weights_new, z_new, u_new

        # Initial ADMM variables
        z = np.copy(self.model)
        u = np.zeros_like(self.model)
        
        # Iteratively update weights, z, and u according to ADMM rules
        for _ in range(10):  # Set the number of ADMM iterations
            self.model, z, u = admm_update(self.model, z, u)
        
        # Apply pruning by setting weights below threshold to zero
        self.prune_weights()

    # Prune the weights based on the pruning rate and update the mask
    def prune_weights(self):
        """
        In the pruning step, we eliminate weights with the smallest magnitudes, 
        defined by the pruning rate, and update the mask accordingly.
        """
        threshold = np.percentile(np.abs(self.model), self.pruning_rate * 100)
        self.mask = np.where(np.abs(self.model) >= threshold, 1, 0)
        self.model *= self.mask  # Zero out pruned weights

    # Local training on the client-side data after weight pruning
    def local_training(self, epochs=5, learning_rate=0.01):
        """
        Standard local training using gradient descent. The pruned weights remain 
        zero and are not updated during training, preserving the sparsity.
        """
        start_cpu = psutil.cpu_percent(interval=None)
        start_memory = psutil.virtual_memory().percent
        
        for epoch in range(epochs):
            gradients = np.dot(self.local_data["X"].T, (self.local_data["y"] - self.local_data["X"] @ self.model))
            self.model += learning_rate * gradients * self.mask  # Only update unpruned weights

        end_cpu = psutil.cpu_percent(interval=None)
        end_memory = psutil.virtual_memory().percent
        
        print(f"Client - CPU Consumption: {end_cpu - start_cpu:.2f}%, Memory Consumption: {end_memory - start_memory:.2f}%")

    # Upload the pruned model and mask to the server for aggregation
    def upload_model(self):
        return self.model, self.mask

# Federated learning simulation with server and clients
def federated_learning_simulation(server, clients, rounds=10, failure_rate=0.2):
    """
    The main loop for simulating federated learning with personalized subnetworks.
    Each round, the clients train locally, prune the model, and upload their pruned
    models and masks to the server, which aggregates them into a global model.
    This version includes random client failures to evaluate robustness.
    """
    for round_num in range(rounds):
        print(f"Round {round_num + 1} / {rounds}")
        client_models, client_masks = [], []

        # Each client trains locally and prunes the model
        for client in clients:
            # Simulate network failure for some clients
            if random.random() > failure_rate:
                client.local_training()
                client.admm_weight_pruning()
                model, mask = client.upload_model()
                client_models.append(model)
                client_masks.append(mask)
            else:
                print(f"Client failed to communicate.")
        
        # Server aggregates the pruned models and updates the global model
        if client_models:  # Only update if there are successful clients
            server.update_global_model(client_models)
        
        # Server derives subnetworks and distributes them to clients
        for client in clients:
            subnetwork = server.derive_subnetwork(client.mask)
            client.model = subnetwork  # Update client model with the derived subnetwork

# Evaluate scalability by gradually increasing the number of clients
def evaluate_scalability(server, initial_clients, test_data, test_labels, rounds=10, epochs=1, max_clients=50, step=10):
    """
    Evaluates the scalability of the federated learning system by gradually increasing the number of clients.
    Args:
        server (Server): The central server.
        initial_clients (list of Client): Initial list of clients.
        test_data (np.array): Test dataset.
        test_labels (np.array): Test labels.
        rounds (int): Number of FL rounds.
        epochs (int): Local epochs per client.
        max_clients (int): Maximum number of clients to test.
        step (int): Step increment for the number of clients.
    """
    current_clients = initial_clients.copy()
    
    for num_clients in range(len(initial_clients), max_clients + 1, step):
        print(f"Evaluating with {num_clients} clients.")
        # Add new clients dynamically
        for i in range(len(current_clients), num_clients):
            new_client = Client(np.random.randn(10, 1), *load_client_data())  # Load new client data
            current_clients.append(new_client)
        
        federated_learning_simulation(server, current_clients, rounds=rounds)
        print("\n")

# Example usage
if __name__ == "__main__":
    # Define a simple global model (for demonstration purposes)
    global_model = np.random.randn(10, 1)

    # Create server
    server = Server(global_model)

    # Generate some random local data for clients (replace with real datasets)
    client_data = [
        {"X": np.random.randn(100, 10), "y": np.random.randint(0, 2, (100, 1))},
        {"X": np.random.randn(100, 10), "y": np.random.randint(0, 2, (100, 1))},
        {"X": np.random.randn(100, 10), "y": np.random.randint(0, 2, (100, 1))},
    ]

    # Create clients
    clients = [Client(np.copy(global_model), data) for data in client_data]

    # Run federated learning simulation
    federated_learning_simulation(server, clients)

    # Evaluate scalability
    evaluate_scalability(server, clients, None, None)

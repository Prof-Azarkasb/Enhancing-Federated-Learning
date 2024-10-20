import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers
import random
import time  # For measuring communication latency
import psutil  # For measuring resource consumption

# Parameters for federated learning
NUM_CLIENTS = 10  # Number of clients
CLIENT_FRACTION = 0.5  # Fraction of clients to participate in each round
EPOCHS = 5  # Number of local epochs for each client
BATCH_SIZE = 32  # Batch size for local training
LEARNING_RATE = 0.01  # Learning rate for optimizers

# Define a simple model for demonstration
def create_model():
    """Creates a simple neural network model with 2 dense layers."""
    model = tf.keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(784,)),
        layers.Dense(10, activation='softmax')
    ])
    return model

# Client class representing a participating device in federated learning
class Client:
    def __init__(self, data, labels):
        """Initializes the client with its own local dataset."""
        self.data = data
        self.labels = labels
        self.model = create_model()
        self.optimizer = None  # Placeholder for the adaptive optimizer (will be assigned later)
    
    def set_optimizer(self, optimizer_name):
        """Sets the optimizer for local training."""
        if optimizer_name == 'Adagrad':
            self.optimizer = optimizers.Adagrad(learning_rate=LEARNING_RATE)
        elif optimizer_name == 'Adam':
            self.optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
        elif optimizer_name == 'Yogi':
            self.optimizer = Yogi(learning_rate=LEARNING_RATE)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        self.model.compile(optimizer=self.optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    def train(self, epochs=EPOCHS, batch_size=BATCH_SIZE):
        """Trains the local model on the client's data while measuring resource consumption."""
        start_cpu = psutil.cpu_percent(interval=None)
        start_memory = psutil.virtual_memory().percent
        
        self.model.fit(self.data, self.labels, epochs=epochs, batch_size=batch_size, verbose=0)
        
        end_cpu = psutil.cpu_percent(interval=None)
        end_memory = psutil.virtual_memory().percent
        
        print(f"Client - CPU Consumption: {end_cpu - start_cpu:.2f}%, Memory Consumption: {end_memory - start_memory:.2f}%")

    def get_weights(self):
        """Returns the client's local model weights."""
        return self.model.get_weights()
    
    def set_weights(self, weights):
        """Sets the client's local model weights."""
        self.model.set_weights(weights)

# Yogi optimizer (an adaptive optimizer)
class Yogi(optimizers.Optimizer):
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-3, name="Yogi", **kwargs):
        """Yogi optimizer for adaptive federated optimization."""
        super(Yogi, self).__init__(name, **kwargs)
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m = None  # First moment estimate
        self.v = None  # Second moment estimate
    
    def apply_gradients(self, grads_and_vars):
        """Applies gradients to the variables using Yogi update rules."""
        if self.m is None:
            self.m = [tf.zeros_like(var) for grad, var in grads_and_vars]
            self.v = [tf.zeros_like(var) for grad, var in grads_and_vars]
        
        for (grad, var), m, v in zip(grads_and_vars, self.m, self.v):
            m.assign(self.beta_1 * m + (1 - self.beta_1) * grad)
            v.assign(v + (1 - self.beta_2) * tf.sign(tf.square(grad) - v) * tf.square(grad))
            var.assign_sub(self.learning_rate * m / (tf.sqrt(v) + self.epsilon))

# Federated Learning Server
class FederatedServer:
    def __init__(self, clients):
        """Initializes the server with a list of clients."""
        self.clients = clients
        self.global_model = create_model()  # Global model maintained by the server
    
    def aggregate_weights(self, client_weights):
        """Aggregates the weights from multiple clients by averaging them."""
        new_weights = []
        for weights_list_tuple in zip(*client_weights):
            new_weights.append(np.mean(weights_list_tuple, axis=0))
        return new_weights

    def update_global_model(self, client_weights):
        """
        Updates the global model with the aggregated weights from clients.
        Also measures communication latency.
        """
        start_time = time.time()  # Start time for measuring latency
        aggregated_weights = self.aggregate_weights(client_weights)
        self.global_model.set_weights(aggregated_weights)
        end_time = time.time()  # End time for measuring latency
        
        latency = end_time - start_time  # Calculate latency
        print(f"Communication Latency: {latency:.4f} seconds")
    
    def federated_round(self, optimizer_name):
        """Performs a single round of federated learning, updating the global model."""
        selected_clients = random.sample(self.clients, int(CLIENT_FRACTION * len(self.clients)))
        
        # Step 1: Broadcast global model weights to clients
        global_weights = self.global_model.get_weights()
        for client in selected_clients:
            client.set_weights(global_weights)
        
        # Step 2: Each client trains on its local data
        for client in selected_clients:
            client.set_optimizer(optimizer_name)  # Set the optimizer
            client.train()
        
        # Step 3: Clients send updated weights to the server
        client_weights = [client.get_weights() for client in selected_clients]
        
        # Step 4: Server aggregates the client weights
        self.update_global_model(client_weights)
    
    def evaluate_global_model(self, test_data, test_labels):
        """Evaluates the global model on a test dataset."""
        self.global_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        loss, accuracy = self.global_model.evaluate(test_data, test_labels, verbose=0)
        return loss, accuracy

    def federated_learning_with_failures(self, failure_rate=0.2):
        """
        Simulates federated learning with random client failures to evaluate robustness.
        """
        for rnd in range(10):  # Simulate 10 rounds
            print(f"Round {rnd + 1} of Federated Learning with Failures")
            
            client_weights = []
            for client in self.clients:
                # Simulate network failure for some clients
                if random.random() > failure_rate:
                    client.set_optimizer("Adam")  # Example optimizer
                    client.train()
                    client_weights.append(client.get_weights())
                else:
                    print(f"Client failed to communicate.")
            
            if client_weights:
                self.update_global_model(client_weights)
            
            # Evaluate global model
            loss, accuracy = self.evaluate_global_model(test_data, test_labels)
            print(f"Global Model - Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")

# Sample usage of Adaptive Federated Learning (A-FL)
def adaptive_federated_learning(clients, test_data, test_labels, optimizer_name, rounds=10):
    """Simulates the Adaptive Federated Learning process over multiple rounds."""
    server = FederatedServer(clients)
    
    for round_num in range(rounds):
        print(f"Starting round {round_num + 1}/{rounds} with {optimizer_name} optimizer...")
        server.federated_round(optimizer_name)
        
        # Evaluate global model after each round
        loss, accuracy = server.evaluate_global_model(test_data, test_labels)
        print(f"Round {round_num + 1} - Global Model Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")

# Example simulation (replace with real datasets)
def generate_fake_data(num_samples, input_shape, num_classes):
    """Generates random synthetic data for testing purposes."""
    data = np.random.rand(num_samples, input_shape)
    labels = np.random.randint(0, num_classes, num_samples)
    return data, labels

if __name__ == "__main__":
    # Generate synthetic data for clients (replace with real federated datasets)
    clients = []
    for _ in range(NUM_CLIENTS):
        data, labels = generate_fake_data(600, 784, 10)  # 600 samples per client, 784 features (e.g., MNIST)
        clients.append(Client(data, labels))
    
    # Generate synthetic test data
    test_data, test_labels = generate_fake_data(1000, 784, 10)
    
    # Run Adaptive Federated Learning with different optimizers
    for optimizer in ['Adagrad', 'Adam', 'Yogi']:
        print(f"\nRunning A-FL with {optimizer} optimizer")
        adaptive_federated_learning(clients, test_data, test_labels, optimizer)
        
    # Evaluate robustness to network failure
    print("\nRunning Federated Learning with Failures")
    server = FederatedServer(clients)
    server.federated_learning_with_failures(failure_rate=0.2)

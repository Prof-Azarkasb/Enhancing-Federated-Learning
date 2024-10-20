# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import time  # For measuring communication latency
import psutil  # For measuring resource consumption
import random  # For simulating network failures

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class EdgeClient:
    """
    Represents an edge device (client) in the hierarchical federated learning system.
    Each edge client trains a local model on its data and sends updates to the regional server.
    """
    def __init__(self, client_id, data, labels):
        self.client_id = client_id
        self.data = data
        self.labels = labels
        self.data_size = len(data)  # Size of client's dataset
        self.local_model = self.create_model()

    def create_model(self):
        """
        Defines the local neural network model architecture.
        Returns:
            model (tf.keras.Model): A neural network model for classification.
        """
        model = models.Sequential([
            layers.InputLayer(input_shape=(30,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_local_model(self, epochs=1, batch_size=32):
        """
        Trains the local model on the client's local data.
        Args:
            epochs (int): Number of epochs to train the local model.
            batch_size (int): Batch size for local training.
        Returns:
            history (tf.keras.callbacks.History): Training history for analysis.
        """
        start_cpu = psutil.cpu_percent(interval=None)
        start_memory = psutil.virtual_memory().percent
        
        history = self.local_model.fit(self.data, self.labels, epochs=epochs, batch_size=batch_size, verbose=0)
        
        end_cpu = psutil.cpu_percent(interval=None)
        end_memory = psutil.virtual_memory().percent
        
        print(f"Client {self.client_id} - CPU Consumption: {end_cpu - start_cpu:.2f}%, Memory Consumption: {end_memory - start_memory:.2f}%")
        return history

    def get_local_model_weights(self):
        """
        Returns the current weights of the local model.
        Returns:
            list: List of model weight arrays.
        """
        return self.local_model.get_weights()

    def set_local_model_weights(self, global_weights):
        """
        Sets the client's local model weights to the global model weights.
        Args:
            global_weights (list): The global model weights.
        """
        self.local_model.set_weights(global_weights)


class RegionalServer:
    """
    Represents a regional server in the hierarchical federated learning framework.
    Regional servers aggregate updates from edge clients and send the aggregated model to the central server.
    """
    def __init__(self, region_id, clients):
        self.region_id = region_id
        self.clients = clients  # List of edge clients assigned to this regional server
        self.regional_model = self.create_regional_model()

    def create_regional_model(self):
        """
        Defines the regional model architecture, identical to the edge clients' models.
        Returns:
            model (tf.keras.Model): Regional model architecture.
        """
        model = models.Sequential([
            layers.InputLayer(input_shape=(30,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def aggregate_client_weights(self):
        """
        Aggregates the model weights from all edge clients in the region.
        Returns:
            aggregated_weights (list): Aggregated weights from all clients.
        """
        client_weights = [client.get_local_model_weights() for client in self.clients]
        aggregated_weights = [np.zeros_like(w) for w in client_weights[0]]
        
        total_data_size = sum(client.data_size for client in self.clients)
        
        # Aggregate weights using weighted average based on the size of client datasets
        for idx, weights in enumerate(client_weights):
            for i in range(len(aggregated_weights)):
                aggregated_weights[i] += weights[i] * (self.clients[idx].data_size / total_data_size)
        
        return aggregated_weights

    def update_regional_model(self):
        """
        Updates the regional model with aggregated weights from edge clients.
        """
        aggregated_weights = self.aggregate_client_weights()
        self.regional_model.set_weights(aggregated_weights)

    def distribute_weights_to_clients(self):
        """
        Distributes the updated regional model's weights back to the edge clients.
        """
        global_weights = self.regional_model.get_weights()
        for client in self.clients:
            client.set_local_model_weights(global_weights)

    def get_regional_model_weights(self):
        """
        Returns the current weights of the regional model to be sent to the central server.
        """
        return self.regional_model.get_weights()


class CentralServer:
    """
    Represents the central server in the hierarchical federated learning framework.
    The central server aggregates updates from regional servers.
    """
    def __init__(self, regional_servers):
        self.regional_servers = regional_servers  # List of regional servers
        self.global_model = self.create_global_model()

    def create_global_model(self):
        """
        Defines the global model architecture, identical to the regional models.
        Returns:
            model (tf.keras.Model): Global model architecture.
        """
        model = models.Sequential([
            layers.InputLayer(input_shape=(30,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def aggregate_regional_weights(self):
        """
        Aggregates the model weights from all regional servers.
        Returns:
            aggregated_weights (list): Aggregated weights from all regional servers.
        """
        regional_weights = [server.get_regional_model_weights() for server in self.regional_servers]
        aggregated_weights = [np.zeros_like(w) for w in regional_weights[0]]
        
        total_data_size = sum(len(server.clients) for server in self.regional_servers)  # Total number of clients
        
        # Aggregate weights using weighted average based on the number of clients in each region
        for idx, weights in enumerate(regional_weights):
            region_size = len(self.regional_servers[idx].clients)
            for i in range(len(aggregated_weights)):
                aggregated_weights[i] += weights[i] * (region_size / total_data_size)
        
        return aggregated_weights

    def update_global_model(self):
        """
        Updates the global model with aggregated weights from regional servers.
        """
        aggregated_weights = self.aggregate_regional_weights()
        self.global_model.set_weights(aggregated_weights)

    def distribute_weights_to_regional_servers(self):
        """
        Distributes the updated global model's weights back to the regional servers.
        """
        global_weights = self.global_model.get_weights()
        for server in self.regional_servers:
            server.regional_model.set_weights(global_weights)

    def evaluate_global_model(self, test_data, test_labels):
        """
        Evaluates the global model on the test dataset.
        Args:
            test_data (np.array): Test data.
            test_labels (np.array): Test labels.
        Returns:
            evaluation_metrics (dict): Dictionary containing accuracy and loss.
        """
        loss, accuracy = self.global_model.evaluate(test_data, test_labels, verbose=0)
        return {'loss': loss, 'accuracy': accuracy}

    def federated_learning_with_failures(self, rounds=10, epochs=1, failure_rate=0.2):
        """
        Simulates federated learning with random client failures to evaluate robustness.
        """
        for rnd in range(rounds):
            print(f"Round {rnd + 1}/{rounds} of Federated Learning with Failures")
            
            for server in self.regional_servers:
                server.update_regional_model()  # Update model with clients
                server.distribute_weights_to_clients()  # Distribute weights

            client_weights = []
            for server in self.regional_servers:
                for client in server.clients:
                    # Simulate network failure for some clients
                    if random.random() > failure_rate:
                        client.train_local_model(epochs=epochs)
                        client_weights.append(client.get_local_model_weights())
                    else:
                        print(f"Client {client.client_id} failed to communicate.")

            if client_weights:
                self.update_global_model()

            # Evaluate global model
            eval_metrics = self.evaluate_global_model(test_data, test_labels)
            print(f"Global Model - Accuracy: {eval_metrics['accuracy']:.4f}, Loss: {eval_metrics['loss']:.4f}")


# Load and preprocess the dataset
def load_and_prepare_data():
    """
    Loads and preprocesses the Breast Cancer dataset from sklearn.
    Splits the data into client-specific datasets and a global test set.
    Returns:
        clients (list): List of EdgeClient objects.
        test_data (np.array): Test data.
        test_labels (np.array): Test labels.
    """
    # Load the dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Split the training data among clients
    num_clients = 5
    clients = []
    for client_id in range(num_clients):
        client_data, _, client_labels, _ = train_test_split(X_train, y_train, test_size=0.8, random_state=client_id)
        clients.append(EdgeClient(client_id, client_data, client_labels))

    return clients, X_test, y_test


# Main execution
if __name__ == "__main__":
    # Load data and create clients
    clients, test_data, test_labels = load_and_prepare_data()
    
    # Create regional servers
    num_regions = 2
    regional_servers = [RegionalServer(region_id, clients[region_id::num_regions]) for region_id in range(num_regions)]
    
    # Create central server
    central_server = CentralServer(regional_servers)

    # Run federated learning with network failures
    central_server.federated_learning_with_failures(rounds=10, epochs=1)

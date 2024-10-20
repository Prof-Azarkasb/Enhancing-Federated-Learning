# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import time  # For measuring communication latency
import psutil  # For measuring resource consumption
import random  # For simulating network failure

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Simulating a basic Personalized Federated Learning (PFL) environment
class Client:
    """
    Represents a client in the personalized federated learning framework. 
    Each client holds a local dataset, trains a local model, and personalizes 
    the global model using its local data distribution.
    """
    def __init__(self, client_id, data, labels):
        self.client_id = client_id
        self.data = data
        self.labels = labels
        self.data_size = len(data)  # Size of client dataset
        self.local_model = self.create_model()
        self.personalized_model = self.create_model()  # Personalized model for each client

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
        history = self.local_model.fit(self.data, self.labels, epochs=epochs, batch_size=batch_size, verbose=0)
        return history

    def personalize_model(self, global_weights, personalization_epochs=1, batch_size=32):
        """
        Personalizes the global model to the client's local data by further training 
        the global model on the local data for a few epochs.
        Args:
            global_weights (list): Weights of the global model.
            personalization_epochs (int): Number of epochs to personalize the model.
            batch_size (int): Batch size for training the personalized model.
        """
        self.personalized_model.set_weights(global_weights)
        self.personalized_model.fit(self.data, self.labels, epochs=personalization_epochs, batch_size=batch_size, verbose=0)

    def evaluate_personalized_model(self, test_data, test_labels):
        """
        Evaluates the personalized model on test data.
        Args:
            test_data (np.array): Test data for evaluation.
            test_labels (np.array): Test labels for evaluation.
        Returns:
            evaluation_metrics (dict): Dictionary containing accuracy and loss.
        """
        loss, accuracy = self.personalized_model.evaluate(test_data, test_labels, verbose=0)
        return {'loss': loss, 'accuracy': accuracy}

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

    def resource_consumption(self):
        """
        Measures and returns the resource consumption of the client during training.
        Returns:
            dict: Dictionary containing CPU and Memory usage.
        """
        start_cpu = psutil.cpu_percent(interval=None)
        start_memory = psutil.virtual_memory().percent

        # Dummy training for measurement purpose
        self.train_local_model(epochs=1)

        end_cpu = psutil.cpu_percent(interval=None)
        end_memory = psutil.virtual_memory().percent
        
        return {
            'cpu_consumption': end_cpu - start_cpu,
            'memory_consumption': end_memory - start_memory
        }


class Server:
    """
    Represents the central server in the personalized federated learning framework.
    The server facilitates communication between clients and provides a global model 
    that clients can use to initialize their personalized models.
    """
    def __init__(self, num_clients):
        self.global_model = self.create_global_model()
        self.clients = []
        self.num_clients = num_clients

    def create_global_model(self):
        """
        Defines the global model architecture, identical to the clients' models.
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

    def aggregate_weights(self, client_weights):
        """
        Aggregates weights from all clients using simple averaging (no weighting here).
        Args:
            client_weights (list of lists): List containing the model weights from all clients.
        Returns:
            aggregated_weights (list): Aggregated weights from all clients.
        """
        # Initialize the aggregate with zeros of the same shape as the weights
        aggregated_weights = [np.zeros_like(w) for w in client_weights[0]]
        
        # Average weights from all clients
        for weights in client_weights:
            for i in range(len(aggregated_weights)):
                aggregated_weights[i] += weights[i]
        
        # Average out the weights
        aggregated_weights = [w / self.num_clients for w in aggregated_weights]
        
        return aggregated_weights

    def update_global_model(self, client_weights):
        """
        Updates the global model with aggregated weights from clients.
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

    def evaluate_global_model(self, test_data, test_labels):
        """
        Evaluates the global model on a test dataset.
        Args:
            test_data (np.array): Test data.
            test_labels (np.array): Test labels.
        Returns:
            evaluation_metrics (dict): Dictionary containing accuracy and loss.
        """
        loss, accuracy = self.global_model.evaluate(test_data, test_labels, verbose=0)
        return {'loss': loss, 'accuracy': accuracy}

    def federated_learning_with_failures(self, clients, test_data, test_labels, rounds=10, epochs=1, failure_rate=0.2):
        """
        Simulates federated learning with random client failures to evaluate robustness.
        """
        for rnd in range(rounds):
            print(f"Round {rnd + 1}/{rounds} of Federated Learning with Failures")
            
            client_weights = []
            for client in clients:
                # Simulate network failure for some clients
                if random.random() > failure_rate:
                    client.train_local_model(epochs=epochs)
                    client_weights.append(client.get_local_model_weights())
                else:
                    print(f"Client {client.client_id} failed to communicate.")
            
            if client_weights:
                self.update_global_model(client_weights)
            
            # Evaluate global model
            eval_metrics = self.evaluate_global_model(test_data, test_labels)
            print(f"Global Model - Accuracy: {eval_metrics['accuracy']:.4f}, Loss: {eval_metrics['loss']:.4f}")

# Load and preprocess the dataset
def load_and_prepare_data():
    """
    Loads and preprocesses the Breast Cancer dataset from sklearn.
    Splits the data into client-specific datasets and a global test set.
    Returns:
        clients (list): List of client objects with their respective datasets.
        test_data (np.array): Global test data.
        test_labels (np.array): Global test labels.
    """
    data = load_breast_cancer()
    X = data['data']
    y = data['target']
    
    # Standardizing the data for better performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the data into training (80%) and testing (20%)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Simulate multiple clients by splitting the training data
    num_clients = 5
    client_data_size = len(X_train) // num_clients
    clients = []
    
    for i in range(num_clients):
        start_index = i * client_data_size
        end_index = start_index + client_data_size
        client = Client(i, X_train[start_index:end_index], y_train[start_index:end_index])
        clients.append(client)
    
    return clients, X_test, y_test

# Personalized Federated Learning procedure
def personalized_federated_learning():
    """
    Main function to run the personalized federated learning simulation.
    Initializes clients, the server, and executes the federated learning process.
    """
    clients, test_data, test_labels = load_and_prepare_data()
    server = Server(num_clients=len(clients))
    
    # Run the federated learning process with potential client failures
    server.federated_learning_with_failures(clients, test_data, test_labels, rounds=10, epochs=1)

# Run the personalized federated learning simulation
personalized_federated_learning()

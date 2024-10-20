# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Simulating a basic Federated Learning (FL) environment with weighted aggregation
class Client:
    """
    Represents a client in the federated learning framework. Each client
    holds a local dataset and trains a model using this dataset.
    """
    def __init__(self, client_id, data, labels):
        self.client_id = client_id
        self.data = data
        self.labels = labels
        self.data_size = len(data)  # Client dataset size, used for weighted aggregation
        self.model = self.create_model()

    def create_model(self):
        """
        Defines the local neural network model architecture.
        Returns:
            model (tf.keras.Model): A simple neural network model for classification.
        """
        model = models.Sequential([
            layers.InputLayer(input_shape=(30,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, epochs=1, batch_size=32):
        """
        Trains the model on local data.
        Args:
            epochs (int): Number of epochs to train.
            batch_size (int): Batch size for training.
        Returns:
            history (tf.keras.callbacks.History): Training history for analysis.
        """
        history = self.model.fit(self.data, self.labels, epochs=epochs, batch_size=batch_size, verbose=0)
        return history

    def get_weights(self):
        """
        Returns the current model weights after training.
        Returns:
            list: List of model weight arrays.
        """
        return self.model.get_weights()

    def set_weights(self, global_weights):
        """
        Sets the model's weights to the globally aggregated weights.
        Args:
            global_weights (list): The aggregated weights from the server.
        """
        self.model.set_weights(global_weights)

class Server:
    """
    Represents the central server in the federated learning framework. It aggregates
    the weights from all participating clients and updates the global model using 
    weighted aggregation based on client importance (e.g., dataset size).
    """
    def __init__(self, num_clients):
        self.global_model = self.create_model()
        self.clients = []
        self.num_clients = num_clients

    def create_model(self):
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

    def aggregate_weights(self, client_weights, client_sizes):
        """
        Aggregates weights from all clients using weighted averaging.
        Clients with larger datasets contribute more to the global model update.
        Args:
            client_weights (list of lists): List containing weights from all clients.
            client_sizes (list): List of client dataset sizes used as weights.
        Returns:
            aggregated_weights (list): Aggregated (weighted) weights.
        """
        # Total size of all clients' datasets
        total_size = np.sum(client_sizes)
        
        # Initialize the aggregate with zeros of the same shape as the weights
        weighted_weights = [np.zeros_like(w) for w in client_weights[0]]
        
        for i, weights in enumerate(client_weights):
            for j in range(len(weighted_weights)):
                weighted_weights[j] += weights[j] * (client_sizes[i] / total_size)
        
        return weighted_weights

    def update_global_model(self, client_weights, client_sizes):
        """
        Updates the global model with the aggregated weights from clients
        using weighted averaging based on client dataset sizes.
        Args:
            client_weights (list of lists): List containing the model weights from all clients.
            client_sizes (list): List of client dataset sizes used as weights.
        """
        aggregated_weights = self.aggregate_weights(client_weights, client_sizes)
        self.global_model.set_weights(aggregated_weights)

    def evaluate_global_model(self, test_data, test_labels):
        """
        Evaluates the global model on a validation or test dataset.
        Args:
            test_data (np.array): Test data.
            test_labels (np.array): Ground truth labels for the test data.
        Returns:
            evaluation_metrics (dict): Dictionary containing loss and accuracy.
        """
        loss, accuracy = self.global_model.evaluate(test_data, test_labels, verbose=0)
        return {'loss': loss, 'accuracy': accuracy}

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

# Weighted Federated Learning procedure
def weighted_federated_learning(server, clients, test_data, test_labels, rounds=10, epochs=1):
    """
    Implements the Weighted Federated Learning process, where the server aggregates
    model updates from multiple clients after each round of local training, with 
    the aggregation being weighted by the size of each client's dataset.
    Args:
        server (Server): The central server object managing aggregation.
        clients (list of Client): List of client objects participating in the training.
        test_data (np.array): Test dataset for evaluating the global model.
        test_labels (np.array): Test labels for model evaluation.
        rounds (int): Number of federated learning communication rounds.
        epochs (int): Number of local epochs clients train per round.
    """
    for rnd in range(rounds):
        print(f"Round {rnd + 1}/{rounds} of Weighted Federated Learning")
        
        # Step 1: Clients train locally
        client_weights = []
        client_sizes = []
        for client in clients:
            client.train(epochs=epochs)
            client_weights.append(client.get_weights())
            client_sizes.append(client.data_size)
        
        # Step 2: Server aggregates client weights using weighted aggregation
        server.update_global_model(client_weights, client_sizes)
        
        # Step 3: Distribute the global model back to the clients
        for client in clients:
            client.set_weights(server.global_model.get_weights())
        
        # Step 4: Evaluate global model on test data
        eval_metrics = server.evaluate_global_model(test_data, test_labels)
        print(f"Global Model - Accuracy: {eval_metrics['accuracy']:.4f}, Loss: {eval_metrics['loss']:.4f}")

# Run the experiment
if __name__ == "__main__":
    clients, X_test, y_test = load_and_prepare_data()
    server = Server(num_clients=len(clients))
    
    # Run the Weighted Federated Learning process
    weighted_federated_learning(server, clients, X_test, y_test, rounds=20, epochs=2)

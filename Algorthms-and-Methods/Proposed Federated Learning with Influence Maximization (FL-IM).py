import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random
import time  # For measuring communication latency
import psutil  # For measuring resource consumption

# Define parameters
NUMBER_OF_ROUNDS = 10
LOCAL_EPOCHS = 5
COMMUNICATION_THRESHOLD = 0.5
DATASET_PATH = "path/to/google_2019_cluster_sample.csv"  # Update with the correct path

# Load dataset
def load_data(path):
    """
    Load the dataset from the specified path.
    Args:
        path (str): Path to the dataset file.
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    data = pd.read_csv(path)
    return data

# Preprocess data
def preprocess_data(data):
    """
    Preprocess the dataset by handling missing values and normalizing the data.
    Args:
        data (pd.DataFrame): The raw dataset.
    Returns:
        pd.DataFrame: Preprocessed dataset.
    """
    # Fill missing values (example strategy)
    data.fillna(method='ffill', inplace=True)
    # Normalize features (example strategy)
    data = (data - data.mean()) / data.std()
    return data

# Identify influential nodes based on a specific criterion (e.g., node performance)
def identify_influential_nodes(data):
    """
    Identify influential nodes from the dataset based on their performance metrics.
    Args:
        data (pd.DataFrame): The preprocessed dataset.
    Returns:
        list: List of influential node indices.
    """
    influential_nodes = random.sample(range(len(data)), k=int(len(data) * 0.2))  # 20% of nodes
    return influential_nodes

# Initialize local model
def initialize_local_model(input_shape):
    """
    Initialize a local model for each node.
    Args:
        input_shape (int): Number of features in the dataset.
    Returns:
        keras.Model: A compiled Keras model.
    """
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(input_shape,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Assuming binary classification
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train local model
def train_local_model(model, X, y):
    """
    Train the local model on the node's dataset.
    Args:
        model (keras.Model): The local model to be trained.
        X (np.array): Feature data for training.
        y (np.array): Labels for training.
    Returns:
        np.array: Model weights after training.
    """
    model.fit(X, y, epochs=LOCAL_EPOCHS, verbose=0)
    return model.get_weights()

# Update global model by averaging local updates
def update_global_model(global_model, local_weights):
    """
    Update the global model by aggregating local model weights.
    Args:
        global_model (keras.Model): The global model to be updated.
        local_weights (list): List of weights from local models.
    Returns:
        keras.Model: Updated global model.
    """
    # Average the weights from local models
    averaged_weights = []
    for weights in zip(*local_weights):
        averaged_weights.append(np.mean(weights, axis=0))
    global_model.set_weights(averaged_weights)
    return global_model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test dataset.
    Args:
        model (keras.Model): The model to evaluate.
        X_test (np.array): Feature data for testing.
        y_test (np.array): Labels for testing.
    Returns:
        float: Accuracy of the model.
        float: F1-score of the model.
    """
    y_pred = (model.predict(X_test) > 0.5).astype(int)  # Threshold for binary classification
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, f1

# Class definitions for Client and Server
class Client:
    def __init__(self, client_id, data):
        self.client_id = client_id
        self.data = data
        self.model = initialize_local_model(input_shape=data.shape[1] - 1)  # Exclude target column

    def train(self, epochs=LOCAL_EPOCHS):
        """
        Trains the local model and measures resource consumption.
        """
        start_cpu = psutil.cpu_percent(interval=None)
        start_memory = psutil.virtual_memory().percent
        
        X = self.data.drop('target', axis=1).values  # Replace 'target' with actual target column name
        y = self.data['target'].values  # Replace 'target' with actual target column name
        self.model.fit(X, y, epochs=epochs, verbose=0)
        
        end_cpu = psutil.cpu_percent(interval=None)
        end_memory = psutil.virtual_memory().percent
        
        print(f"Client {self.client_id} - CPU Consumption: {end_cpu - start_cpu:.2f}%, Memory Consumption: {end_memory - start_memory:.2f}%")
        return self.model.get_weights()

class Server:
    def __init__(self):
        self.global_model = None

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

    def aggregate_weights(self, client_weights):
        """
        Aggregate weights from clients (average).
        Args:
            client_weights (list): Weights from all clients.
        Returns:
            list: Aggregated weights.
        """
        # Average the weights from local models
        averaged_weights = []
        for weights in zip(*client_weights):
            averaged_weights.append(np.mean(weights, axis=0))
        return averaged_weights

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
                    client_weights.append(client.train(epochs=epochs))
                else:
                    print(f"Client {client.client_id} failed to communicate.")
            
            if client_weights:
                self.update_global_model(client_weights)
            
            # Evaluate global model
            accuracy, f1 = evaluate_model(self.global_model, test_data.drop('target', axis=1).values, test_data['target'].values)
            print(f"Global Model - Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")

# Main Federated Learning with Influence Maximization algorithm
def FL_IM_algorithm():
    # Load and preprocess data
    data = load_data(DATASET_PATH)
    preprocessed_data = preprocess_data(data)

    # Split data into features and labels
    X = preprocessed_data.drop('target', axis=1).values  # Replace 'target' with actual target column name
    y = preprocessed_data['target'].values  # Replace 'target' with actual target column name
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize global model
    server = Server()
    server.global_model = initialize_local_model(input_shape=X.shape[1])
    
    # Prepare clients
    clients = []
    for i in range(10):  # Create 10 clients for demonstration
        client_data = pd.DataFrame(data={
            'feature1': np.random.rand(100),  # Replace with actual features
            'feature2': np.random.rand(100),  # Replace with actual features
            'target': np.random.randint(0, 2, size=100)  # Random binary target
        })
        clients.append(Client(client_id=i, data=client_data))

    # Start Federated Learning process
    server.federated_learning_with_failures(clients, X_test, y_test)

# Execute the FL-IM algorithm
if __name__ == "__main__":
    FL_IM_algorithm()

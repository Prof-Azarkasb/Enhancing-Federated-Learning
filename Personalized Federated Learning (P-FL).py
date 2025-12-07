# Import necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
import time
import psutil
import random

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Parameters
LOCAL_EPOCHS = 2
BATCH_SIZE = 32
LEARNING_RATE = 0.01
GLOBAL_ROUNDS = 10
PERSONALIZATION_FACTOR = 0.5  # Weight of discrepancy for personalized model
FAILURE_RATE = 0.2

# Load Google 2019 Cluster Sample Dataset
def load_google_cluster_data(path="google_cluster_data.csv", num_clients=5):
    """
    Loads and preprocesses Google 2019 Cluster Sample Dataset .
    Splits the data among clients and produces global test set.
    """
    df = pd.read_csv(path)
    
    # Use only numeric features (CPU/Memory usage)
    features = df.select_dtypes(include=np.number).fillna(0)
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features.values)
    
    # Generate pseudo labels (e.g., usage buckets for classification)
    y = np.digitize(X_scaled[:, 0], bins=np.percentile(X_scaled[:, 0], np.linspace(0,100,11))) - 1
    
    # Split into train/test
    split_idx = int(0.8 * len(X_scaled))
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Split train among clients
    client_data_size = len(X_train) // num_clients
    clients = []
    for i in range(num_clients):
        start = i * client_data_size
        end = start + client_data_size
        clients.append({
            'data': X_train[start:end],
            'labels': y_train[start:end]
        })
    
    return clients, X_test, y_test

# Client class for P-FL
class Client:
    def __init__(self, client_id, data, labels, input_dim, num_classes):
        self.client_id = client_id
        self.data = data
        self.labels = labels
        self.data_size = len(data)
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.local_model = self.create_model()
        self.personalized_model = self.create_model()
    
    def create_model(self):
        model = models.Sequential([
            layers.InputLayer(input_shape=(self.input_dim,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model
    
    def train_local_model(self, epochs=LOCAL_EPOCHS, batch_size=BATCH_SIZE):
        self.local_model.fit(self.data, self.labels, epochs=epochs, batch_size=batch_size, verbose=0)
    
    def personalize_model(self, global_weights, discrepancy_factor=PERSONALIZATION_FACTOR, epochs=1, batch_size=BATCH_SIZE):
        # Compute discrepancy-based personalized weights
        local_weights = self.local_model.get_weights()
        personalized_weights = [gw + discrepancy_factor*(lw - gw)
                                for gw, lw in zip(global_weights, local_weights)]
        self.personalized_model.set_weights(personalized_weights)
        # Further train on local data
        self.personalized_model.fit(self.data, self.labels, epochs=epochs, batch_size=batch_size, verbose=0)
    
    def get_local_weights(self):
        return self.local_model.get_weights()
    
    def set_local_weights(self, global_weights):
        self.local_model.set_weights(global_weights)
    
    def evaluate_personalized(self, X_test, y_test):
        loss, acc = self.personalized_model.evaluate(X_test, y_test, verbose=0)
        return {'loss': loss, 'accuracy': acc}
    
    def resource_usage(self):
        start_cpu = psutil.cpu_percent(interval=None)
        start_mem = psutil.virtual_memory().percent
        self.train_local_model(epochs=1)
        end_cpu = psutil.cpu_percent(interval=None)
        end_mem = psutil.virtual_memory().percent
        return {'cpu': end_cpu-start_cpu, 'memory': end_mem-start_mem}

# Server class for P-FL
class Server:
    def __init__(self, input_dim, num_classes):
        self.global_model = self.create_global_model(input_dim, num_classes)
    
    def create_global_model(self, input_dim, num_classes):
        model = models.Sequential([
            layers.InputLayer(input_shape=(input_dim,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model
    
    def aggregate_weights(self, client_weights):
        aggregated = []
        for weights in zip(*client_weights):
            aggregated.append(np.mean(weights, axis=0))
        return aggregated
    
    def update_global_model(self, client_weights):
        start_time = time.time()
        aggregated = self.aggregate_weights(client_weights)
        self.global_model.set_weights(aggregated)
        end_time = time.time()
        print(f"Communication Latency: {end_time - start_time:.4f}s")
    
    def federated_learning_round(self, clients, failure_rate=FAILURE_RATE):
        selected_weights = []
        for client in clients:
            if random.random() > failure_rate:
                client.train_local_model()
                selected_weights.append(client.get_local_weights())
            else:
                print(f"Client {client.client_id} failed to communicate.")
        if selected_weights:
            self.update_global_model(selected_weights)
    
    def evaluate_global_model(self, X_test, y_test):
        loss, acc = self.global_model.evaluate(X_test, y_test, verbose=0)
        return {'loss': loss, 'accuracy': acc}

# Main P-FL procedure
def personalized_federated_learning_google_cluster():
    clients_data, X_test, y_test = load_google_cluster_data()
    input_dim = X_test.shape[1]
    num_classes = len(np.unique(y_test))
    
    clients = [Client(i, c['data'], c['labels'], input_dim, num_classes)
               for i, c in enumerate(clients_data)]
    server = Server(input_dim, num_classes)
    
    for rnd in range(GLOBAL_ROUNDS):
        print(f"--- Global Round {rnd+1} ---")
        server.federated_learning_round(clients)
        # Personalize models after global aggregation
        global_weights = server.global_model.get_weights()
        for client in clients:
            client.personalize_model(global_weights)
        eval_metrics = np.mean([client.evaluate_personalized(X_test, y_test)['accuracy'] for client in clients])
        print(f"Average Personalized Accuracy: {eval_metrics:.4f}")

# Run the P-FL simulation
if __name__ == "__main__":
    personalized_federated_learning_google_cluster()

import numpy as np
import random
import time
import psutil

# Constants for federated learning setup
NUM_CLIENTS = 10  # Number of participating clients
NUM_ROUNDS = 100  # Number of global training rounds
LEARNING_RATE = 0.01  # Learning rate for local updates
TRANSMISSION_COST = 0.1  # Network transmission cost (bit-based compression)
MOMENTUM = 0.9  # Momentum for signSGD with momentum
FAILURE_RATE = 0.2  # Probability of client failure during communication

# Client class simulating each federated learning client
class Client:
    def __init__(self, client_id, data_size, computation_power):
        self.client_id = client_id
        self.data_size = data_size
        self.computation_power = computation_power
        self.model = np.random.randn(10)  # Initialize model with random weights
        self.momentum = np.zeros_like(self.model)  # Initialize momentum term

    def compute_local_gradient(self, global_model):
        """
        Compute the local gradient based on the current global model.
        The gradient is computed stochastically, simulating real-world federated learning.
        """
        local_gradient = (self.model - global_model) + np.random.normal(scale=0.01, size=self.model.shape)
        return local_gradient
    
    def update_model(self, global_model, sign_gradient):
        """
        Update the local model using the global model and the signed gradient.
        The local model is updated using momentum-based signSGD, which only uses the sign of the gradient.
        """
        self.momentum = MOMENTUM * self.momentum + sign_gradient
        self.model = global_model - LEARNING_RATE * np.sign(self.momentum)

    def train(self, epochs=1, batch_size=32):
        """
        Trains the model on local data while measuring resource consumption.
        """
        start_cpu = psutil.cpu_percent(interval=None)
        start_memory = psutil.virtual_memory().percent
        
        # Simulate training process
        for _ in range(epochs):
            _ = self.model * 0.9  # Dummy operation to simulate training
        
        end_cpu = psutil.cpu_percent(interval=None)
        end_memory = psutil.virtual_memory().percent
        
        print(f"Client {self.client_id} - CPU Consumption: {end_cpu - start_cpu:.2f}%, Memory Consumption: {end_memory - start_memory:.2f}%")

# Gradient Federated Learning Server class
class FederatedServer:
    def __init__(self):
        self.global_model = np.random.randn(10)  # Initialize global model randomly

    def aggregate_sign_gradients(self, client_gradients):
        """
        Aggregates gradients using a majority vote on the signs of the gradient components.
        This approach allows for 1-bit communication, compressing the gradients significantly.
        """
        sign_aggregate = np.sign(np.sum(np.sign(client_gradients), axis=0))
        return sign_aggregate

# Gradient Federated Learning (Gra-FL) class
class GradientFederatedLearning:
    def __init__(self, num_clients, num_rounds):
        self.clients = [Client(client_id=i, data_size=random.randint(1000, 5000), 
                               computation_power=random.uniform(1, 10)) for i in range(num_clients)]
        self.num_rounds = num_rounds
        self.server = FederatedServer()
    
    def run(self):
        """
        Execute the Gradient Federated Learning (Gra-FL) process, involving local updates, 
        compression of gradient signs, and global model aggregation using majority voting.
        """
        for round_num in range(self.num_rounds):
            print(f"--- Round {round_num + 1} ---")
            client_gradients = []
            communication_latency_total = 0

            # Each client computes the local gradient and transmits the sign of the gradient
            for client in self.clients:
                if random.random() > FAILURE_RATE:  # Simulate network failure for some clients
                    start_time = time.time()
                    local_gradient = client.compute_local_gradient(self.server.global_model)
                    sign_gradient = np.sign(local_gradient)
                    client_gradients.append(sign_gradient)
                    end_time = time.time()
                    
                    # Measure communication latency
                    communication_latency = end_time - start_time
                    communication_latency_total += communication_latency
                    print(f"Client {client.client_id} - Communication Latency: {communication_latency:.4f} seconds")
                else:
                    print(f"Client {client.client_id} failed to communicate.")
            
            if client_gradients:
                # Server aggregates the signed gradients using majority voting
                aggregated_sign_gradient = self.server.aggregate_sign_gradients(client_gradients)
                print(f"Aggregated Sign Gradient: {aggregated_sign_gradient}")
                
                # Update global model using the aggregated sign of gradients
                self.server.global_model -= LEARNING_RATE * aggregated_sign_gradient
                print(f"Updated Global Model: {self.server.global_model}")
                
                # Each client updates its local model using the signed gradient
                for client in self.clients:
                    client.update_model(self.server.global_model, aggregated_sign_gradient)

            # Log the average model performance for this round (for evaluation purposes)
            average_loss = self.evaluate_global_model()
            print(f"Average Loss after Round {round_num + 1}: {average_loss}")
            print(f"Total Communication Latency for Round {round_num + 1}: {communication_latency_total:.4f} seconds\n")
    
    def evaluate_global_model(self):
        """
        Evaluate the global model on all clients, simulating global model performance.
        For simplicity, we use a mock evaluation with randomly generated losses.
        """
        total_loss = 0
        for client in self.clients:
            loss = np.linalg.norm(client.model - self.server.global_model)
            total_loss += loss
        average_loss = total_loss / len(self.clients)
        return average_loss

# Running the Gradient Federated Learning (Gra-FL) simulation
if __name__ == "__main__":
    gra_fl = GradientFederatedLearning(num_clients=NUM_CLIENTS, num_rounds=NUM_ROUNDS)
    gra_fl.run()

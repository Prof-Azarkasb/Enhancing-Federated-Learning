import numpy as np
import random
import psutil  # For measuring resource consumption
import time  # For measuring communication latency
from collections import defaultdict

# Constants for the federated learning environment
NUM_CLIENTS = 10  # Number of participating clients in FL
NUM_ROUNDS = 100  # Total number of training rounds in FL
MAX_RESOURCES = 100  # Maximum available resources at each edge node (task scheduling)
RESOURCE_SCALING_FACTOR = 0.7  # Resource adjustment factor
TRANSMISSION_COST = 0.1  # Network transmission cost
FAILURE_RATE = 0.2  # Rate of client failure simulation

# Client class to simulate each federated learning client
class Client:
    def __init__(self, client_id, data_size, computation_power):
        self.client_id = client_id
        self.data_size = data_size
        self.computation_power = computation_power
        self.model = None  # Placeholder for the client's local model
    
    def compute_loss(self, model):
        """
        Simulates the client's loss computation based on model updates.
        The client's contribution depends on the local dataset and model parameters.
        """
        # Simulating model loss using a random factor for simplicity
        loss = np.random.rand() * self.data_size / self.computation_power
        return loss

    def update_model(self, global_model):
        """
        Updates the local model with the global model received from the server.
        """
        self.model = global_model

    def train(self, epochs=1, batch_size=32):
        """
        Trains the model on local data while measuring resource consumption.
        """
        start_cpu = psutil.cpu_percent(interval=None)
        start_memory = psutil.virtual_memory().percent
        
        # Simulate local training
        for _ in range(epochs):
            _ = self.compute_loss(self.model)  # Simulating model training
        
        end_cpu = psutil.cpu_percent(interval=None)
        end_memory = psutil.virtual_memory().percent
        
        print(f"Client {self.client_id} - CPU Consumption: {end_cpu - start_cpu:.2f}%, Memory Consumption: {end_memory - start_memory:.2f}%")
        
        return np.random.rand()  # Return a simulated model update

# Greedy Task Scheduler class
class GreedyTaskScheduler:
    def __init__(self, clients, max_resources):
        self.clients = clients  # List of client objects
        self.resource_pool = max_resources  # Available resources for task scheduling
    
    def allocate_resources(self):
        """
        Greedily allocates resources to the clients based on their computation power.
        Higher computation power clients get higher priority in task scheduling.
        """
        # Sort clients by their computation power (descending order)
        sorted_clients = sorted(self.clients, key=lambda x: x.computation_power, reverse=True)
        resource_allocation = defaultdict(float)
        
        # Greedy allocation of resources based on computation power
        for client in sorted_clients:
            allocation = min(self.resource_pool, client.computation_power * RESOURCE_SCALING_FACTOR)
            resource_allocation[client.client_id] = allocation
            self.resource_pool -= allocation
            
            if self.resource_pool <= 0:
                break
        
        return resource_allocation

# Federated Learning server
class FederatedServer:
    def __init__(self):
        self.global_model = None  # Placeholder for global model
    
    def aggregate_models(self, client_models):
        """
        Aggregates the models from all clients to update the global model.
        """
        # Averaging the models (simplified aggregation rule)
        aggregated_model = sum(client_models) / len(client_models)
        return aggregated_model

    def update_global_model(self, client_models):
        """
        Updates the global model with the aggregated models from clients.
        Also measures communication latency.
        """
        start_time = time.time()  # Start time for measuring latency
        self.global_model = self.aggregate_models(client_models)
        end_time = time.time()  # End time for measuring latency
        
        latency = end_time - start_time  # Calculate latency
        print(f"Communication Latency: {latency:.4f} seconds")

# Main Federated Learning process with Greedy FL
class GreedyFL:
    def __init__(self, num_clients, num_rounds, max_resources):
        self.clients = [Client(client_id=i, data_size=random.randint(1000, 5000), 
                               computation_power=random.uniform(1, 10)) for i in range(num_clients)]
        self.num_rounds = num_rounds
        self.scheduler = GreedyTaskScheduler(self.clients, max_resources)
        self.server = FederatedServer()

    def run(self):
        """
        Executes the Greedy Federated Learning algorithm over multiple rounds.
        """
        for round_num in range(self.num_rounds):
            print(f"--- Round {round_num + 1} ---")
            
            # Allocate resources based on client computation power
            resource_allocation = self.scheduler.allocate_resources()
            print(f"Resource Allocation: {resource_allocation}")
            
            client_losses = []
            client_models = []
            successful_clients = 0
            
            # Each client computes its local update and returns its loss
            for client in self.clients:
                # Simulate network failure for some clients
                if random.random() > FAILURE_RATE:
                    loss = client.train(epochs=1)  # Training the client model
                    client_losses.append(loss)
                    client_models.append(np.random.rand())  # Placeholder for local model update
                    successful_clients += 1
                else:
                    print(f"Client {client.client_id} failed to communicate.")
            
            if successful_clients > 0:
                # Aggregation of models at the server
                self.server.update_global_model(client_models)
                print(f"Global Model Updated: {self.server.global_model}")
            
            # Logging average loss for this round
            avg_loss = sum(client_losses) / len(client_losses) if client_losses else float('inf')
            print(f"Average Loss in Round {round_num + 1}: {avg_loss}")

# Running the Greedy Federated Learning simulation
if __name__ == "__main__":
    greedy_fl = GreedyFL(num_clients=NUM_CLIENTS, num_rounds=NUM_ROUNDS, max_resources=MAX_RESOURCES)
    greedy_fl.run()

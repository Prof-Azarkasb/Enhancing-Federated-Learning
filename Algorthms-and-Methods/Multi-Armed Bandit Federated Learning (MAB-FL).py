import numpy as np
import random
import time  # For measuring communication latency
import psutil  # For measuring resource consumption

# Constants for Federated Learning setup
NUM_CLIENTS = 10  # Number of participating clients (arms in MAB)
NUM_ROUNDS = 100  # Number of global communication rounds
LEARNING_RATE = 0.01  # Learning rate for local model updates
EXPLORATION_FACTOR = 2  # Exploration factor for UCB policy
COMMUNICATION_COST = 0.1  # Cost of communication per client
NON_IID_FACTOR = 0.5  # Non-iid factor to simulate heterogeneous data
FAILURE_RATE = 0.2  # Probability of client communication failure

# Client class simulating each federated learning client
class Client:
    def __init__(self, client_id, local_data_distribution, computation_power):
        self.client_id = client_id
        self.local_data_distribution = local_data_distribution  # Data distribution characteristic for non-i.i.d. data
        self.computation_power = computation_power  # Client's computational power
        self.model = np.random.randn(10)  # Initialize local model randomly
        self.local_model_performance = 0  # Placeholder for model accuracy or loss after training
    
    def compute_local_gradient(self, global_model):
        noise_factor = np.random.normal(scale=self.local_data_distribution)
        local_gradient = (self.model - global_model) + noise_factor
        return local_gradient
    
    def update_model(self, global_model, gradient):
        self.model = global_model - LEARNING_RATE * gradient
    
    def simulate_training(self):
        noise = np.random.normal(scale=0.1)
        self.local_model_performance = 1 / (1 + np.exp(-self.computation_power)) - noise  # Simulate accuracy
    
    def get_local_performance(self):
        return self.local_model_performance

# Federated Learning Server class
class FederatedServer:
    def __init__(self):
        self.global_model = np.random.randn(10)  # Initialize global model randomly

    def aggregate_client_gradients(self, client_gradients):
        return np.mean(client_gradients, axis=0)

    def update_global_model(self, client_weights):
        start_time = time.time()  # Start time for measuring latency
        aggregated_weights = self.aggregate_client_gradients(client_weights)
        self.global_model -= LEARNING_RATE * aggregated_weights
        end_time = time.time()  # End time for measuring latency
        
        latency = end_time - start_time  # Calculate latency
        print(f"Communication Latency: {latency:.4f} seconds")

# Multi-Armed Bandit with Upper Confidence Bound (UCB) for client selection
class MABClientScheduler:
    def __init__(self, num_clients):
        self.num_clients = num_clients
        self.client_rewards = np.zeros(num_clients)  # Cumulative rewards (performance) for each client
        self.client_selection_counts = np.zeros(num_clients)  # Number of times each client has been selected
        self.round_counter = 1  # To keep track of the number of rounds
    
    def select_client(self):
        ucb_values = np.zeros(self.num_clients)
        
        for client in range(self.num_clients):
            if self.client_selection_counts[client] > 0:
                avg_reward = self.client_rewards[client] / self.client_selection_counts[client]
                ucb_values[client] = avg_reward + EXPLORATION_FACTOR * np.sqrt(np.log(self.round_counter) / self.client_selection_counts[client])
            else:
                ucb_values[client] = float('inf')  # Ensure exploration
            
        selected_client = np.argmax(ucb_values)  # Select client with highest UCB value
        return selected_client
    
    def update_client_reward(self, client_id, reward):
        self.client_rewards[client_id] += reward
        self.client_selection_counts[client_id] += 1
        self.round_counter += 1

# Multi-Armed Bandit Federated Learning (MAB-FL) class
class MABFederatedLearning:
    def __init__(self, num_clients, num_rounds):
        self.clients = [
            Client(client_id=i, local_data_distribution=random.uniform(0.1, NON_IID_FACTOR),
                   computation_power=random.uniform(1, 10)) for i in range(num_clients)
        ]
        self.num_rounds = num_rounds
        self.server = FederatedServer()
        self.scheduler = MABClientScheduler(num_clients)
    
    def run(self):
        for round_num in range(self.num_rounds):
            print(f"--- Round {round_num + 1} ---")
            
            selected_client_id = self.scheduler.select_client()
            selected_client = self.clients[selected_client_id]
            
            # Simulate client communication failure
            if random.random() > FAILURE_RATE:
                local_gradient = selected_client.compute_local_gradient(self.server.global_model)
                selected_client.simulate_training()
                local_performance = selected_client.get_local_performance()
                print(f"Client {selected_client_id} Performance: {local_performance}")

                # Update the client's model locally
                selected_client.update_model(self.server.global_model, local_gradient)
                
                # Server updates the global model and measures latency
                self.server.update_global_model([local_gradient])
                
                # Update the scheduler with the observed reward (performance)
                self.scheduler.update_client_reward(selected_client_id, local_performance)
            else:
                print(f"Client {selected_client_id} failed to communicate.")
            
            average_loss = self.evaluate_global_model()
            print(f"Average Loss after Round {round_num + 1}: {average_loss}")
    
    def evaluate_global_model(self):
        total_loss = 0
        for client in self.clients:
            loss = np.linalg.norm(client.model - self.server.global_model)
            total_loss += loss
        average_loss = total_loss / len(self.clients)
        return average_loss

def evaluate_scalability(server, initial_clients, num_rounds, max_clients=50, step=10):
    for num_clients in range(len(initial_clients), max_clients + 1, step):
        print(f"\nEvaluating with {num_clients} clients.")
        # Dynamically add new clients
        for i in range(len(initial_clients), num_clients):
            new_client = Client(i, *load_client_data())  # Load new client data
            initial_clients.append(new_client)
        
        # Run federated learning simulation
        mab_fl = MABFederatedLearning(num_clients=num_clients, num_rounds=num_rounds)
        mab_fl.run()

# Running the Multi-Armed Bandit Federated Learning (MAB-FL) simulation
if __name__ == "__main__":
    mab_fl = MABFederatedLearning(num_clients=NUM_CLIENTS, num_rounds=NUM_ROUNDS)
    mab_fl.run()

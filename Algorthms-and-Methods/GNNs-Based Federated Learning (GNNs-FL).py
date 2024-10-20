import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, DataLoader
import numpy as np
import time  # For measuring communication latency
import psutil  # For measuring resource consumption
import random  # For simulating network failures

# GNN-Based Federated Learning for Neighbor Selection and Resource Allocation

# Graph Attention Network (GAT) Model with multi-head attention
class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=4):
        super(GAT, self).__init__()
        # First graph attention layer with multiple heads
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, concat=True)
        # Second graph attention layer to reduce dimension to output
        self.gat2 = GATConv(hidden_dim * heads, output_dim, heads=1, concat=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gat1(x, edge_index)
        x = torch.relu(x)
        x = self.gat2(x, edge_index)
        return x

# GNN-FL Neighbor Selection & Resource Allocation class
class GNNFL:
    def __init__(self, num_clients, input_dim, hidden_dim, output_dim, lr=0.001):
        self.num_clients = num_clients
        self.gnn_model = GAT(input_dim, hidden_dim, output_dim).to(device)
        self.optimizer = optim.Adam(self.gnn_model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()  # Loss function for resource allocation optimization

    def train_gnn(self, data_loader, epochs=100):
        """
        Train the GNN model to select neighbors and optimize resource allocation.
        """
        self.gnn_model.train()
        for epoch in range(epochs):
            total_loss = 0
            for data in data_loader:
                data = data.to(device)
                self.optimizer.zero_grad()
                output = self.gnn_model(data)
                loss = self.loss_fn(output, data.y)  # Target values for power/CPU frequency
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(data_loader)}")

    def evaluate(self, data_loader):
        """
        Evaluate the GNN model's performance on a validation/test set.
        """
        self.gnn_model.eval()
        total_loss = 0
        with torch.no_grad():
            for data in data_loader:
                data = data.to(device)
                output = self.gnn_model(data)
                loss = self.loss_fn(output, data.y)
                total_loss += loss.item()
        print(f"Evaluation Loss: {total_loss / len(data_loader)}")
        return total_loss / len(data_loader)

    def neighbor_selection(self, data):
        """
        Perform neighbor selection using the trained GNN model.
        """
        self.gnn_model.eval()
        with torch.no_grad():
            output = self.gnn_model(data.to(device))
        selected_neighbors = output.topk(k=5, dim=0).indices  # Select top-k neighbors based on attention scores
        return selected_neighbors.cpu().numpy()

    def resource_allocation(self, data):
        """
        Perform resource allocation based on GNN output.
        """
        self.gnn_model.eval()
        with torch.no_grad():
            output = self.gnn_model(data.to(device))
        return output.cpu().numpy()  # Return resource allocation decision (e.g., power, CPU freq)

    def update_global_model(self, client_weights):
        """
        Updates the global model with the aggregated weights from clients.
        Also measures communication latency.
        """
        start_time = time.time()  # Start time for measuring latency
        aggregated_weights = self.aggregate_weights(client_weights)
        self.gnn_model.load_state_dict(aggregated_weights)
        end_time = time.time()  # End time for measuring latency
        
        latency = end_time - start_time  # Calculate latency
        print(f"Communication Latency: {latency:.4f} seconds")

# Generating sample data for simulation (clients and edges between them)
def generate_synthetic_data(num_clients, input_dim):
    """
    Generate synthetic graph data for training the GNN model.
    """
    x = torch.randn((num_clients, input_dim))  # Random features for each client
    edge_index = torch.randint(0, num_clients, (2, num_clients * 2))  # Random edges between clients
    y = torch.randn((num_clients, 1))  # Targets
    data = Data(x=x, edge_index=edge_index, y=y)
    return data

# Decentralized Federated Learning Simulation with Network Failures
def run_simulation(num_clients=10, input_dim=5, hidden_dim=16, output_dim=1, epochs=100):
    """
    Simulate the decentralized federated learning process.
    """
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize GNN-based Federated Learning instance
    gnn_fl = GNNFL(num_clients, input_dim, hidden_dim, output_dim)

    # Create a synthetic dataset of clients for training the GNN model
    train_data = generate_synthetic_data(num_clients, input_dim)
    train_loader = DataLoader([train_data], batch_size=1)

    # Train the GNN model
    print("Training GNN model for neighbor selection and resource allocation...")
    gnn_fl.train_gnn(train_loader, epochs)

    # Simulate federated learning with client failures
    print("Simulating Federated Learning with Network Failures...")
    client_weights = []
    failure_rate = 0.2  # 20% failure rate

    for client_id in range(num_clients):
        # Simulate network failure for some clients
        if random.random() > failure_rate:
            # Each client would train its model here (dummy training as a placeholder)
            client_weights.append(train_data.y)  # Append dummy client weights
            print(f"Client {client_id} trained successfully.")
        else:
            print(f"Client {client_id} failed to communicate.")

    if client_weights:
        gnn_fl.update_global_model(client_weights)

    # Evaluate the GNN model
    print("Evaluating global model...")
    eval_loss = gnn_fl.evaluate(train_loader)

if __name__ == "__main__":
    run_simulation()

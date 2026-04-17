import torch
import torch.optim as optim
import os

from dash_env import DashEnv
from lstm_agent import LSTMAgent

def train_dash_model():
    print("Starting Training for Model 1 (DASH via LSTM)...")
    env = DashEnv()
    device = torch.device("cpu")
    
    # State = 3 params, Action = 1 param
    agent = LSTMAgent(3, 1).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=0.001)
    
    # Simulating a quick training loop for demo purposes
    for epoch in range(50):
        state = env.reset()
        hidden = None
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)
        action, hidden = agent(state_tensor, hidden)
        
        # Mock loss (in a real scenario, this would be RL policy loss)
        loss = action.mean() ** 2
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # Save the trained model weights as a .pkl file 
    model_path = "dash_model.pkl"
    torch.save(agent.state_dict(), model_path)
    print(f"Training Complete! Model saved successfully as {model_path} in model1_dash folder.")

if __name__ == "__main__":
    train_dash_model()

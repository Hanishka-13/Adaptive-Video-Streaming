import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from sac.actor import Actor
from sac.critic import Critic


class SACAgent:
    def __init__(self, state_dim, action_dim, device):

        self.device = device
        self.gamma = 0.99
        self.tau = 0.005
        self.target_entropy = -float(action_dim)

        # --- Auto-tuning Alpha (Temperature) ---
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp().item()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)

        # Actor
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)

        # Critics (Twin Q)
        self.critic1 = Critic(state_dim, action_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim).to(device)

        self.critic1_target = Critic(state_dim, action_dim).to(device)
        self.critic2_target = Critic(state_dim, action_dim).to(device)

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=3e-4)

    # --------------------------------------------------
    # Action Sampling
    # --------------------------------------------------
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mean, log_std = self.actor(state)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)

        return action.detach().cpu().numpy()[0]

    # --------------------------------------------------
    # Update Step
    # --------------------------------------------------
    def update(self, replay_buffer, batch_size=64):

        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        state      = torch.FloatTensor(state).to(self.device)
        action     = torch.FloatTensor(action).to(self.device)
        reward     = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done       = torch.FloatTensor(done).to(self.device)

        # ---------------------------
        # Compute Target Q
        # ---------------------------
        with torch.no_grad():
            next_mean, next_log_std = self.actor(next_state)
            next_std = next_log_std.exp()
            normal = torch.distributions.Normal(next_mean, next_std)
            z = normal.rsample()
            next_action = torch.tanh(z)

            log_prob = normal.log_prob(z) - torch.log(1 - next_action.pow(2) + 1e-7)
            log_prob = log_prob.sum(1, keepdim=True)

            q1_target = self.critic1_target(next_state, next_action)
            q2_target = self.critic2_target(next_state, next_action)
            min_q_target = torch.min(q1_target, q2_target)

            target_q = reward + (1 - done) * self.gamma * (min_q_target - self.alpha * log_prob)

        # ---------------------------
        # Update Critics
        # ---------------------------
        q1 = self.critic1(state, action)
        q2 = self.critic2(state, action)

        critic1_loss = nn.MSELoss()(q1, target_q)
        critic2_loss = nn.MSELoss()(q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # ---------------------------
        # Update Actor
        # ---------------------------
        mean, log_std = self.actor(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action_new = torch.tanh(z)

        log_prob = normal.log_prob(z) - torch.log(1 - action_new.pow(2) + 1e-7)
        log_prob = log_prob.sum(1, keepdim=True)

        q1_new = self.critic1(state, action_new)
        q2_new = self.critic2(state, action_new)
        min_q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha * log_prob - min_q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ---------------------------
        # Update Alpha (Auto-Tuning)
        # ---------------------------
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp().item()

        # ---------------------------
        # Soft Update Targets
        # ---------------------------
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
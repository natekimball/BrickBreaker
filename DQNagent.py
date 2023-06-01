import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
import random

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(210, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Agent:
    def __init__(self, gamma=.9, epsilon_decay=.999, learning_rate=.001, model_dir=None, replay=True, batch_size=32, memory_size=1024):
        self.gamma = gamma
        self.batch_size = batch_size
        self.actions = [[0,0], [1,0], [0,1]]
        if model_dir:
            self.model = torch.load(model_dir)
        else:
            self.model = QNetwork()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.epsilon = 1
        self.epsilon_decay = epsilon_decay
        self.use_replay_buffer = replay
        self.memory = []
        self.memory_size = memory_size

    def get_action(self, state):
        Q_vals = self.Q(state)
        print("Q_vals =", Q_vals)
        if np.random.random() < self.epsilon:
            return Q_vals, self.actions[random.randint(0,len(self.actions)-1)]
        return Q_vals, self.actions[np.argmax(Q_vals)]    

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.tensor([state for state, _, _, _ in minibatch], dtype=torch.float32)
        old_Q_vals = self.model(states)
        target_Q = torch.clone(old_Q_vals)
        next_states = [next_state if next_state is not None else np.zeros(210) for _, _, _, next_state in minibatch]
        next_state_Q_vals = self.model(torch.tensor(next_states, dtype=torch.float32))
        for i, (_, action, reward, next_state) in enumerate(minibatch):
            target_Q[i,self.actions.index(action)] = reward
            if next_state is not None:
                target_Q[i, self.actions.index(action)] += self.gamma * torch.max(next_state_Q_vals[i])
        self.optimizer.zero_grad()
        loss = self.criterion(old_Q_vals, target_Q)
        loss.backward()
        self.optimizer.step()

    def update(self, state, action, reward, new_state, Q_vals):
        print("ε =", self.epsilon)
        if np.isclose(self.epsilon,0):
            self.epsilon = 0
        else:
            self.epsilon *= self.epsilon_decay
        if self.use_replay_buffer:
            self.remember(state, action, reward, new_state)
            self.replay()
        else:
            new_Q = reward
            if new_state is not None:
                new_Q += self.gamma * max(self.Q(new_state))
            Q_vals = self.Q(state).detach().numpy()
            new_Q_vals = copy.deepcopy(Q_vals)
            new_Q_vals[self.actions.index(action)] = new_Q
            self.optimizer.zero_grad()
            loss = self.criterion(Q_vals, new_Q_vals)
            loss.backward()
            self.optimizer.step()
    
    def lost(self, state, action, Q_vals):
        self.update(state, action, -100, None, Q_vals)
    
    def Q(self, state):
        return self.model(torch.tensor(state, dtype=torch.float32)).detach().numpy()
    
    def shape(self, state):
        paddle, bricks, ball, ball_dx, ball_dy = state
        data = np.zeros(210)
        data[0:4] = paddle.left, paddle.right, paddle.top, paddle.bottom
        data[4:8] = ball.left, ball.right, ball.top, ball.bottom
        data[8:10] = ball_dx, ball_dy
        for i, (brick, _) in enumerate(bricks):
            data[i+10:i+14] = brick.left, brick.right, brick.top, brick.bottom
        return data

    def save(self, path):
        with open(path, 'w') as f:
            torch.save(self.model.state_dict(), path)

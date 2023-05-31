import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam
import random

class Agent:
    def __init__(self, gamma=.9, epsilon_decay=.999, learning_rate=.001, model_dir=None, replay=True, batch_size=32, memory_size=1024):
        self.gamma = gamma
        self.batch_size = batch_size
        self.actions = [[0,0], [1,0], [0,1]]
        if model_dir:
            self.model = tf.keras.models.load_model(model_dir)
        else:
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu', input_shape=(210,)),
                tf.keras.layers.Dense(len(self.actions), activation='linear')
            ])
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=10,
            decay_rate=.999,
            staircase=True
        )
        self.model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='mse')
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
        states = np.array([state for state, _, _, _ in minibatch])
        print(states.shape)
        Q_vals = self.model.predict(states, batch_size=self.batch_size)
        next_state_Q_vals = self.model.predict(np.array([next_state if next_state is not None else np.zeros(210) for _, _, _, next_state in minibatch]), batch_size=self.batch_size)
        for i, (_, action, reward, next_state) in enumerate(minibatch):
            Q_vals[i,self.actions.index(action)] = reward
            if next_state is not None:
                Q_vals[i][self.actions.index(action)] += self.gamma * np.max(next_state_Q_vals[i])
        self.model.fit(states, np.array(Q_vals), batch_size=self.batch_size)

    # other methods remain the same

    # def update(self, Q_vals, state, action, reward, new_state):
    #     print("ε =", self.epsilon)
    #     self.epsilon *= self.epsilon_decay
    #     Q_vals[self.actions.index(action)] = reward + self.gamma * np.max(self.Q(new_state))
    #     self.model.fit(np.array([state]), np.array([Q_vals]))
    
    def update(self, state, action, reward, new_state, Q_vals):
        print("ε =", self.epsilon)
        self.epsilon *= self.epsilon_decay
        if self.use_replay_buffer:
            self.remember(state, action, reward, new_state)
            self.replay()
        else:
            new_Q = reward
            if new_state is not None:
                new_Q += self.gamma * np.max(self.Q(new_state))
            Q_vals[self.actions.index(action)] = reward
            self.model.fit(np.array([state]), np.array([Q_vals]))
    
    # def lost(self, Q_vals, state, action):
    #     Q_vals[self.actions.index(action)] = -100
    #     self.model.fit(np.array([state]), np.array([Q_vals]))
    
    def lost(self, state, action, Q_vals):
        self.update(state, action, -100, None, Q_vals)
    
    def Q(self, state):
        return np.array(np.squeeze(self.model.predict(np.array([state]))))
    
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
        self.model.save(path)
        
    
        
    # def get_action(self, state):
    #     Q_vals = self.Q(state)
    #     print("Q_vals =", Q_vals)
    #     if np.random.random() < self.epsilon:
    #         return Q_vals, self.actions[random.randint(0,len(self.actions)-1)]
    #     return Q_vals, self.actions[np.argmax(Q_vals)] 
    
    # def update(self, Q_vals, state, action, reward, new_state):
    #     print("ε =", self.epsilon)
    #     self.epsilon *= self.epsilon_decay
    #     Q_vals[self.actions.index(action)] = reward + self.gamma * np.max(self.Q(new_state))
    #     self.model.fit(np.array([state]), np.array([Q_vals]))
        
    # def lost(self, Q_vals, state, action):
    #     Q_vals[self.actions.index(action)] = -100
    #     self.model.fit(np.array([state]), np.array([Q_vals]))





    # def get_action(self, state):
    #     Q_vals = np.array([self.Q(state, action) for action in self.actions])
    #     print(Q_vals)
    #     print(Q_vals.shape)
    #     i = np.argmax(Q_vals)
    #     print(i)
    #     return self.actions[i]
    
    # def update(self, state, action, reward, new_state):
    #     new_q = reward + self.gamma * np.max([self.Q(new_state, action) for action in self.actions])
    #     self.model.fit(self.shape(state, action), new_q)
    
    # def lost(self, state, action):
    #     self.model.fit(self.shape(state, action), 0)
    
    # def Q(self, state, action):
    #     data = np.array([self.shape(state, action)])
    #     return np.squeeze(self.model.predict(data))
    
    # def shape(self, state, action):
    #     paddle, bricks, ball, ball_dx, ball_dy = state
    #     left, right = action
    #     data = np.zeros(self.input_shape)
    #     data[0] = [paddle.left, paddle.right, paddle.top, paddle.bottom]
    #     data[1] = [ball.left, ball.right, ball.top, ball.bottom]
    #     data[2] = [ball_dx, ball_dy, left, right]
    #     for i, (brick, _) in enumerate(bricks):
    #         data[i+3] = [brick.left, brick.right, brick.top, brick.bottom]
    #     return data
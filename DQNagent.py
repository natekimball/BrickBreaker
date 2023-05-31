import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam
import random

class Agent:
    def __init__(self, gamma=.9, epsilon_decay=.999, learning_rate=.001, model_dir=None):
        # 50 bricks
        self.gamma = gamma
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
    
    def get_action(self, state):
        Q_vals = self.Q(state)
        print("Q_vals =", Q_vals)
        if np.random.random() < self.epsilon:
            return Q_vals, self.actions[random.randint(0,len(self.actions)-1)]
        return Q_vals, self.actions[np.argmax(Q_vals)]    

    def update(self, Q_vals, state, action, reward, new_state):
        print("ε =", self.epsilon)
        self.epsilon *= self.epsilon_decay
        Q_vals[self.actions.index(action)] = reward + self.gamma * np.max(self.Q(new_state))
        self.model.fit(np.array([state]), np.array([Q_vals]))
        
    def lost(self, Q_vals, state, action):
        Q_vals[self.actions.index(action)] = -100
        self.model.fit(np.array([state]), np.array([Q_vals]))
    
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
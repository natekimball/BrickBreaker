import tensorflow as tf
import numpy as np
import math


class Agent:
    def __init__(self):
        # 50 bricks
        self.gamma = .9
        self.input_shape = (53,4)
        self.actions = [[0,0], [1,0], [0,1]]
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=self.input_shape),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    def get_action(self, state):
        i = np.argmax([self.Q(state, action) for action in self.actions])
        return actions[i]
    
    def update(self, state, action, reward, new_state):
        new_q = reward + self.gamma * np.max([self.Q(new_state, aaction) for action in self.actions])
        self.model.fit(self.shape(state, action), new_q)
        
    def lost(self, state, action):
        self.model.fit(self.shape(state, action), 0)
    
    def Q(self, state, action):
        data = self.shape(state, action)
        return self.model.predict(data)
    
    def update(self, state, action, reward):
        data = self.shape(state, action)
        self.model.fit(data, self.Q(state, action) + reward)
    
    def shape(self, state, action):
        paddle, bricks, ball, ball_dx, ball_dy = state
        left, right = action
        data = np.zeros(self.input_shape)
        data[0] = [paddle.left, paddle.right, paddle.top, paddle.bottom]
        data[1] = [ball.left, ball.right, ball.top, ball.bottom]
        data[2] = [ball_dx, ball_dy, left, right]
        for i, (brick, _) in enumerate(bricks):
            data[i+3] = [brick.left, brick.right, brick.top, brick.bottom]
        return data
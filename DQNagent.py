import tensorflow as tf
import numpy as np
import math


class Agent:
    def __init__(self):
        # 50 bricks
        self.input_shape = (53,4)
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(5,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    def get_action(self, state):
        actions = [[0,0], [1,0], [0,1]]
        i = np.argmax([self.Q(state, action) for action in actions])
        return actions[i]
    
    def Q(self, state, action):
        data = self.shape(state, action)
        return self.model.predict(data.reshape(1,54))[0]
    
    def update(self, state, action, reward):
        data = self.shape(state, action)
        self.model.fit(data.reshape(1,54), self.Q(state, action) + reward)
    
    def shape(self, state, action):
        paddle, bricks, ball, ball_dx, ball_dy = state
        left, right = action
        data = np.zeros(self.input_shape)
        data[0] = [paddle.left, paddle.right, paddle.top, paddle.bottom]
        data[1] = [ball.left, ball.right, ball.top, ball.bottom]
        data[2] = [ball_dx, ball_dy, left, right]
        for i, (brick, _) in enumerate(bricks):
            data[i+3] = [brick.left, brick.right, brick.top, brick.bottom]
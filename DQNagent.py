import tensorflow as tf
import numpy as np
import random

class Agent:
    def __init__(self):
        # 50 bricks
        self.gamma = .9
        self.input_shape = (53,4)
        self.actions = [[0,0], [1,0], [0,1]]
        self.model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=self.input_shape),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(len(self.actions), activation='linear')
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.epsilon = 1
        self.epsilon_decay = .9999
    
    def get_action(self, state):
        Q_vals = self.Q(state)
        if np.random.random() < self.epsilon:
            # return random.choice(self.actions)
            return Q_vals, self.actions[random.randint(0,len(self.actions)-1)]
        i = np.argmax(Q_vals)
        return Q_vals, self.actions[i]    

    def update(self, Q_vals, state, action, reward, new_state):
        self.epsilon *= self.epsilon_decay
        print("epsilon: ", self.epsilon)
        # new_qs = self.Q(state)
        Q_vals[self.actions.index(action)] = reward + self.gamma * np.max(self.Q(new_state))
        self.model.fit(np.array([state]), np.array([Q_vals]))
        
    def lost(self, Q_vals, state, action):
        # new_qs = self.Q(state)
        Q_vals[self.actions.index(action)] = -100
        self.model.fit(np.array([state]), np.array([Q_vals]))
    
    def Q(self, state):
        return np.array(np.squeeze(self.model.predict(np.array([state]))))
    
    def shape(self, state):
        paddle, bricks, ball, ball_dx, ball_dy = state
        data = np.zeros(self.input_shape)
        data[0] = [paddle.left, paddle.right, paddle.top, paddle.bottom]
        data[1] = [ball.left, ball.right, ball.top, ball.bottom]
        data[2] = [ball_dx, ball_dy, 0, 0]
        for i, (brick, _) in enumerate(bricks):
            data[i+3] = [brick.left, brick.right, brick.top, brick.bottom]
        return data
    
    def save(self, path):
        self.model.save(path)

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
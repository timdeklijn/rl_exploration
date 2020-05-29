import numpy as np
import tensorflow as tf
from collections import deque
import random


class DQN:
    def __init__(self, state_size, action_space):

        # === DQN run control parameters ===
        self.memory_size = 1000
        self.learning_rate = 0.01
        self.batch_size = 32
        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.action_space = [0, 1, 2]

        # === Initialize class variables ===
        self.memory = deque(maxlen=self.memory_size)
        self.model = self.create_model(state_size, action_space)
        self.target_model = self.create_model(state_size, action_space)

    def create_model(self, input_shape, output_shape):
        """Create and compule a simple sequential model

        Returns
        -------
        tf.keras.Sequential
            model
        """
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=input_shape))
        model.add(tf.keras.layers.Dense(10, activation="relu"))
        model.add(tf.keras.layers.Dense(output_shape, activation="relu"))
        model.compile(
            loss="mean_squared_error",
            optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate),
        )
        return model

    def remember(self, obs, action, reward, new_obs, done):
        """Save a time step in the models memory

        Parameters
        ----------
        obs : np.array
            observation
        action : int
            action
        reward : int
            reward
        new_obs : np.array
            observation after action
        done : bool
            game over or not
        """
        self.memory.append([obs, action, reward, new_obs, done])

    def replay(self):
        """
        Sample steps from memory, calculate their Q-values and update
        self.model
        """
        if len(self.memory) < self.batch_size:
            return
        samples = random.sample(self.memory, self.batch_size)
        for sample in samples:
            obs, action, reward, new_obs, done = sample
            target = self.target_model.predict(obs)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_obs)[0])
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(obs, target, epochs=1, verbose=0)

    def target_train(self):
        """
        Copy weights form self.model to self.target_model
        """
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.target_model.set_weights(target_weights)

    def act(self, obs):
        """Choose an action by doing inference on self.model or randomly
        picking an action.

        Parameters
        ----------
        obs : np.array()
            observation

        Returns
        -------
        int
            action
        """
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        return np.argmax(self.model.predict(obs)[0])

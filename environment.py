import numpy as np
import pandas as pd


class Environment:
    def __init__(self, fname, label):
        """Environment class for reinforcement learning classification

        Parameters
        ----------
        fname : str
            File name of data file
        label : str
            Name of y column in data file
        """
        # === Load and set data ===
        self.label = label  # Column to classify
        self.data = pd.read_csv(fname)  # Load data set, this not be altered
        self.actions = self.data[label].unique()
        self.state_size = np.shape(
            np.array([i for i in self.data.columns if i != self.label])
        )

        # === Initiate class variables ===
        self.state = None  # state represents a time point in an environment
        self.episode_obs = None  # np.array with observations (fraction of the data)
        self.episode_labels = None  # np.array with labels for self.game_obs
        self.done = False  # Wether or not the observations are empty

        # === run control ===
        self.fraction = 0.5  # Fraction of data used for an episode

    def step(self, action):
        """Set a step based on the action

        Should return reward, observation, done

        Parameters
        ----------
        action : int
            Action for to take in the environment

        Returns
        -------
        reward: int
            Reward for action
        obs: np.array
            Next observation
        done: bool
            If the environment is over
        """
        # step in environment and call reward function
        reward = self._reward(action)
        obs, done = self._next_observation()
        return reward, obs, done

    def _next_observation(self):
        """Get the new observation

        Returns
        -------
        self.current_obs: np.array
            observation
        self.done: bool
            If the environment is over
        """
        # Take the first observation and label form the episode lists
        self.current_obs = self.episode_obs[0]
        self.current_label = self.episode_labels[0]
        # If episode lists are empty, return done
        if len(self.episode_obs) == 1:
            self.done = True
            return self.current_obs, self.done
        # Remove first elements from episode lists
        self.episode_obs = self.episode_obs[1:]
        self.episode_labels = self.episode_labels[1:]
        return self.current_obs, self.done

    def _reward(self, action):
        """Compare action with result and return reward

        Parameters
        ----------
        action : int
            action to take

        Returns
        -------
        int
            reward
        """
        if self.actions[action] == self.current_label:
            return 1
        return 0

    def reset(self):
        """Reset the environment

        Returns
        -------
        np.array
            first observation of the episode
        """
        self.episode_obs, self.episode_labels = self.get_environment_data()
        obs, self.done = self._next_observation()
        self.done = False
        return obs

    def get_environment_data(self):
        """Sample the data and split into observations and labels

        Returns
        -------
        x: np.array
            observations
        y: np.array
            labels
        """
        tmp = self.data.sample(frac=self.fraction)
        y = tmp.variety.values
        X = tmp.drop(labels=[self.label], axis=1).values
        return X, y

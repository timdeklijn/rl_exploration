import pandas as pd


def load_data():
    """Load data

    Returns
    -------
    pd.DataFrame
        Data, full dataframe
    """
    df = pd.read_csv("iris.csv")
    return df


def get_environment_data(df, fraction):
    """Sample the data and split into observations and labels

    Parameters
    ----------
    df : pd.DataFrame
        Full data
    fraction: float
        fraction of the dataset to sample from

    Returns
    -------
    x: np.array
        observations
    y: np.array
        labels
    """
    tmp = df.sample(frac=fraction)
    y = tmp.variety.values
    X = tmp.drop(labels=["variety"], axis=1).values
    return X, y


class Environment:
    def __init__(self):
        self.state = None  # state represents a time point in an environment
        self.actions = ["Setosa", "Versicolor", "Virginica"]  # Currently hardcoded
        self.fraction = 0.5  # Fraction of data used for an episode
        self.data = load_data()  # Load data set, this not be altered
        self.episode_obs = None  # np.array with observations (fraction of the data)
        self.episode_labels = None  # np.array with labels for self.game_obs
        self.done = False  # Wether or not the observations are empty

    def step(self, action):
        """
        Set a step based on the action

        Should return reward, observation, done
        """
        # step in environment and call reward function
        reward = self._reward(action)
        obs, done = self._next_observation()
        return reward, obs, done

    def _next_observation(self):
        """
        Get the new observation
        """
        self.current_obs = self.episode_obs[0]
        self.current_label = self.episode_labels[0]
        if len(self.episode_obs) == 1:
            self.done = True
            return self.current_obs, self.done
        self.episode_obs = self.episode_obs[1:]
        self.episode_labels = self.episode_labels[1:]
        return self.current_obs, self.done

    def _reward(self, action):
        if self.actions[action] == self.current_label:
            return 1
        return 0

    def reset(self):
        """
        Reset the environment
        """
        self.episode_obs, self.episode_labels = get_environment_data(
            self.data, self.fraction
        )
        obs, self.done = self._next_observation()
        self.done = False
        return obs

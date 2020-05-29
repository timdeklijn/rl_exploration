from environment import Environment
from model import DQN
import logging
import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %I:%M:%S",
)

N_EPISODES = 500
MODEL_UPDATE_FREQ = 50
TARGET_MODEL_UPDATE_FREQ = 500


def take_action(action_space):
    return np.random.randint(0, action_space)


if __name__ == "__main__":
    logging.info("Reinforcement Learning")
    env = Environment()
    action_space = len(env.actions)  # get actions to choose from

    dqn_agent = DQN()

    # Basic rl loop
    observation = env.reset().reshape(1, 4)
    steps = 0
    for ep in range(N_EPISODES):
        tot_reward = 0
        while True:
            action = dqn_agent.act(observation)
            reward, new_observation, done = env.step(action)
            new_observation = new_observation.reshape(1, 4)
            dqn_agent.remember(observation, action, reward, new_observation, done)

            if steps % MODEL_UPDATE_FREQ == 0:
                dqn_agent.replay()
            if steps % TARGET_MODEL_UPDATE_FREQ == 0:
                dqn_agent.target_train()

            observation = new_observation
            tot_reward += reward
            steps += 1
            if done:
                break
        logging.info(
            f"reward: {tot_reward}, episode: {ep}, epsilon: {dqn_agent.epsilon}"
        )
        env.reset()

    df = env.data
    y = df.variety.values
    X = df.drop(labels=["variety"], axis=1).values

    y_pred = dqn_agent.model.predict(X)
    converter = {"Setosa": 0, "Versicolor": 1, "Virginica": 0}
    s = 0
    for i in range(len(y)):
        if converter[y[i]] == np.argmax(y_pred[i]):
            s += 1
    logging.info(f"Accuracy: {s/len(y)}")

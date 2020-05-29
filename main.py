from environment import Environment
import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %I:%M:%S",
)

N_PLAYS = 100


def take_action(action_space):
    return np.random.randint(0, action_space)


if __name__ == "__main__":
    logging.info("Reinforcement Learning")
    env = Environment()
    action_space = len(env.actions)  # get actions to choose from
    observation = env.reset()
    logging.info(f"First observation: {observation}")

    # Basic rl loop
    for _ in range(N_PLAYS):
        tot_reward = 0
        while True:
            action = take_action(action_space)
            reward, observation, done = env.step(action)
            tot_reward += reward
            if done:
                break
        logging.info(f"DONE: reward: {tot_reward}")
        env.reset()

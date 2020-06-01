import logging
from operator import attrgetter
import requests
import multiprocessing

from model import DQN

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %I:%M:%S",
)


class Ace:
    def __init__(self, fname: str, label: str, n_agents: int):

        # === Globals ===
        # Total number of episodes
        self.training_runs = 10
        # Update dqn.model weights every `model_update_freq` steps
        self.model_update_freq = 50
        # Update dqn dqb.target_model weights every `target_model_update_freq` steps
        self.target_model_update_freq = 500
        # Set the weights of all models to the best model every `pick_best_weights_freq`
        # episodes
        self.episodes_per_run = 20

        # === Create Agents ===
        self.agents = [DQN(i, fname, label) for i in range(n_agents)]

        # === Done setting up ACE ===
        logging.info(f"Ace is set up with {n_agents} agents")

    def run_agents(self):
        # Train agents for n runs
        for run in range(self.training_runs):
            # Train all agents for `episodes_per_run` episodes
            for agent in self.agents:
                agent.run_agent(
                    self.episodes_per_run,
                    self.model_update_freq,
                    self.target_model_update_freq,
                )
            self.set_weights()

    def set_weights(self):
        logging.info("Setting best weights to model.")
        self.agents.sort(key=attrgetter("score"), reverse=True)
        top_weights = self.agents[0].model.get_weights()
        for agent in self.agents[1:]:
            agent_model_weights = agent.model.get_weights()
            agent_target_model_weights = agent.target_model.get_weights()
            for i in range(len(top_weights)):
                agent_model_weights[i] = top_weights[i]
                agent_target_model_weights[i] = top_weights[i]
            agent.model.set_weights(agent_model_weights)
            agent.target_model.set_weights(agent_target_model_weights)

import logging
import numpy as np

from ace import Ace

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %I:%M:%S",
)


def assess_final_model(agent):
    # === Test accuracy ===
    df = agent.env.data
    y = df.variety.values
    X = df.drop(labels=["variety"], axis=1).values
    y_pred = agent.model.predict(X)
    converter = {"Setosa": 0, "Versicolor": 1, "Virginica": 2}
    s = 0
    for i in range(len(y)):
        if converter[y[i]] == np.argmax(y_pred[i]):
            s += 1
    logging.info(f"Accuracy: {s/len(y)}")


if __name__ == "__main__":
    logging.info("Multi Agent Reinforcement Learning")

    # === Create Ace class and perform RL ===
    ace = Ace("iris.csv", "variety", 8)
    ace.run_agents()

    final_agent = ace.agents[0]

    assess_final_model(final_agent)
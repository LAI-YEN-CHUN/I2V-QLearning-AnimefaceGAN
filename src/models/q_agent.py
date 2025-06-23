import random
from collections import defaultdict
from itertools import product

from configs import BaseConfig


class QAgent:
    def __init__(self, cfg: BaseConfig):
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.visited = set()
        self.epsilon = cfg.EPSILON
        self.alpha = cfg.ALPHA
        self.gamma = cfg.GAMMA

        self.action_space = list(
            product(cfg.LEARNING_RATE_LIST, cfg.MOMENTUM_LIST, cfg.WEIGHT_DECAY_LIST))

    def get_state(self):
        return "STATE_LESS"

    def select_action(self, state):
        available = [a for a in self.action_space if (
            state, a) not in self.visited]

        if not available:
            # No available actions
            return None

        if random.random() < self.epsilon:
            # With probability epsilon, select a random action
            action = random.choice(available)
        else:
            # With probability 1 - epsilon, select the best action based on Q-values
            q_actions = self.q_table[state]
            sorted_q = sorted(q_actions.items(),
                              key=lambda x: x[1], reverse=True)
            action = None
            for a, _ in sorted_q:
                if (state, a) not in self.visited:
                    action = a
                    break
            if action is None:
                action = random.choice(available)
        self.visited.add((state, action))

        return action

    def update_q(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values(), default=0)
        self.q_table[state][action] += self.alpha * \
            (reward + self.gamma * max_next_q - current_q)

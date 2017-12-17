import math
from project_utils.utilities import *
import pickle


class Environment:

    def __init__(self, rewards):

        self.actions = get_actions()
        self.states = get_actions()
        self.reward = rewards
        self.prev_state = np.random.choice(list(self.states.keys()))
        self.prev_state_id = self.states[self.prev_state]
        self.terminal_states = [self.states['.']]
        self.t = 0


    def getNextState(self, action):
        self.t += 1
        self.next_state = action
        reward = self.reward[self.prev_state][self.next_state]
        #reward = self.reward
        self.prev_state = self.next_state
        self.next_state_id = self.states[self.next_state]
        if self.next_state_id in self.terminal_states:
            run = 'terminate'
        elif self.t == 10:
            run = 'terminate'
        else:
            run = 'run'
        return run, reward





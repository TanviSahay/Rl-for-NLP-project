import pickle
import json
import numpy as np
from datetime import datetime
from project_utils.utilities import *
from Codes import env as ENV
from Codes import senagent as SGA


class model:
    def __init__(self, params, reward_func):

        with open(params,'r') as f:
            self.params = json.load(f)

        self.epsilon = self.params['epsilon']
        self.alpha  = self.params['alpha']
        self.gamma = self.params['gamma']
        self.hyperparams = [self.epsilon, self.alpha, self.gamma]
        self.rewards = getReward(reward_func)
        self.reward_func = reward_func


    def fit(self, neps, tri, perm):

        start = datetime.now()
        self.epsilon, self.alpha, self.gamma = perm

        returns = np.zeros((neps, tri))
        for t in range(tri):
            self.agent = SGA.sentGenAgent(self.epsilon, self.alpha, self.gamma)
            print('Running trial: ', t)
            for n in range(neps):
                #if n % 100 == 0:
                #    print('Running episode: ', n)
                env = ENV.Environment(self.rewards, self.reward_func)
                state = env.prev_state_id
                action = self.agent.getAction(state)
                run = 'run'
                l = 0
                while run == 'run':
                    qsa, phisa = self.agent.qValue(state, action)
                    run, reward = env.getNextState(action)
                    returns[n,t] += (self.gamma ** l) * reward
                    l += 1
                    next_state = env.next_state_id
                    if run == 'terminate':
                        qsa_prime = 0.0
                        next_action = action
                    else:
                        next_action = self.agent.getAction(next_state)
                        qsa_prime, _ = self.agent.qValue(next_state, next_action)
                    self.agent.updateWeights(reward, qsa_prime, qsa, phisa)
                    state = next_state
                    action = next_action
        end = datetime.now()
        return returns, end-start


    def predict(self):
        sentences = []
        for n in range(1000):
            sent = []
            env = ENV.Environment(self.rewards, self.reward_func)
            state = env.prev_state_id
            sent.append(env.prev_state)
            run = 'run'
            while run == 'run':
                action = self.agent.getAction(state)
                run, reward = env.getNextState(action)
                next_state = env.next_state
                sent.append(next_state)
                state = env.next_state_id
            sentences.append(sent)

        return self.score(sentences)


    def score(self, sentences):
        average_score = 0.0
        for sent in sentences:
            average_score += self.bigram_probability(sent)

        return average_score / float(len(sentences))


    def bigram_probability(self,sent):
        with open('./Data/bigram_probability.pkl','rb') as f:
            bigram_occurrence = pickle.load(f)
        bigram_prob = 0.0
        for i in range(len(sent) - 1):
            bigram_prob += np.log(bigram_occurrence[sent[i]][sent[i+1]])
        return np.exp(bigram_prob)




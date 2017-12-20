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
        self.lmbda = self.params['lmbda']
        self.hyperparams = [self.epsilon, self.alpha, self.gamma, self.lmbda]
        self.reward_func = reward_func
        self.rewards = getReward(reward_func)



    def fit(self, neps, tri, perm):

        start = datetime.now()

        self.epsilon, self.alpha, self.gamma, self.lmbda = perm

        returns = np.zeros((neps, tri))
        for t in range(tri):
            self.agent = SGA.sentGenAgent(self.epsilon, self.alpha, self.gamma)
            self.etrace = np.zeros((self.agent.no_actions, self.agent.no_actions))

            print('Running trial: ', t)
            for n in range(neps):
                #if n % 100 == 0:
                #    print('Running episode: ', n)
                env = ENV.Environment(self.rewards, self.reward_func)
                state = env.prev_state_id
                run = 'run'
                l = 0
                while run == 'run':
                    action = self.agent.getAction(state)
                    run, reward = env.getNextState(action)
                    returns[n,t] += (self.gamma ** l) * reward
                    l += 1
                    qsa, phisa = self.agent.qValue(state, action)
                    next_state = env.next_state_id
                    max_qsa_prime = []
                    for act in self.agent.actions.keys():
                        if run == 'terminate':
                            qsa_prime = 0.0
                        else:
                            qsa_prime, _ = self.agent.qValue(next_state, act)
                        max_qsa_prime.append(qsa_prime)
                    qsa_prime = max(max_qsa_prime)
                    self.qlambda_update(reward, qsa_prime, qsa, phisa)
                    state = next_state

        end = datetime.now()
        return returns, end-start


    def qlambda_update(self, reward, qsa_prime, qsa, phisa):
            tderror = reward + self.gamma * qsa_prime - qsa_prime
            self.etrace[phisa[0], phisa[1]] = self.gamma * self.lmbda * self.etrace[phisa[0],phisa[1]] + 1
            self.agent.weights[phisa[0], phisa[1]] = self.agent.weights[phisa[0], phisa[1]] + self.alpha * tderror * self.etrace[phisa[0], phisa[1]]


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




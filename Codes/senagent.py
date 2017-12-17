import math
from project_utils.utilities import *


class sentGenAgent:

    def __init__(self, epsilon, alpha, gamma):


        self.actions = get_actions()
        self.no_actions = len(self.actions)
        self.weights = np.zeros((self.no_actions, self.no_actions))       #(n+1)^d*3
        
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma



    def getStateActionFeatures(self,state,action):
        """Only values for the input action will be non-zero"""
        return [state, self.actions[action]]


    def qValue(self, state, action):
        phisa = self.getStateActionFeatures(state, action)
        qSA = self.weights[phisa[0], phisa[1]]
        #print 'weights: ', self.weights
        #print 'qvalue: ', qSA            #to check that the qvalue is being computed properly
        return qSA, phisa


    def epsilonGreedyPolicy(self, optimal_actions):

        random_number = np.random.rand()

        if random_number <= self.epsilon:
            action = np.random.choice(list(self.actions.keys()), 1)[0]
        else:
            action = np.random.choice(optimal_actions, 1)[0]

        return action


    def getAction(self, states):
        qvalActions = {}
        for key, val in self.actions.items():
            qvalActions[key] = self.qValue(states, key)[0]
        maximum_q = max(qvalActions.values())
        optimal_actions = [k for k,v in qvalActions.items() if v == maximum_q]
        #print optimal_actions
        action = self.epsilonGreedyPolicy(optimal_actions)
        #print action
        return action


    def updateWeights(self,reward,newqSA,qSA,phisa):
        self.weights[phisa[0], phisa[1]] = self.weights[phisa[0], phisa[1]] + self.alpha * (reward + self.gamma * newqSA - qSA)
        pass

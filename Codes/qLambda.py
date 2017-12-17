import numpy as np
import argparse
import pickle
#from joblib import Parallel, delayed
from datetime import datetime
#import tempfile
import environment as ENV
import sentGenAgent as SGA
from project_utils.utilities import *



"""
qlambda algorithm:
1. for each trial:
	initialize agent
	i. for each episode:
		initialize environment
		a. initialize e to 0 for all states and actions (e will be the same shape as w)
		b. initialize initial state
		c. until reaching terminal state or completing time steps
			- choose action a from state
			- take action a and get reward and next state
			- compute the td error - Rt + gamma * max(next_a) q(next_state, next_a) - q(state, a)
			- update e as - gamma * lambda * e + phisa
			- weight = weight + alpha * td * e

"""



def qlambda(args, neps, returns, j):
	agent = SGA.sentGenAgent(epsilon=float(args.epsilon), alpha=float(args.alpha), gamma=float(args.gamma))
	etrace = np.zeros((agent.no_actions, agent.no_actions))
	for eps in range(neps):
		environment = ENV.Enviroment()
		state = environment.prev_state_id
		run = 'run'
		l = 0
		while run == 'run':
			action = agent.getAction(state)
			run, reward = environment.getNextState(action, )
			returns[eps, j] += (float(args.gamma)**l)*reward
			l += 1
			next_state = environment.next_state_id
			max_qsa = []
			qsa, phisa = agent.qValue(state, action)
			for act in agent.actions.keys():
				if run == 'terminate':
					qsa_prime = 0.0
				else:
					qsa_prime,_ = agent.qValue(next_state, act)
				max_qsa.append(qsa_prime)
			max_qsa_prime = max(max_qsa)
			#policy update
			tderror = reward + float(args.gamma) * max_qsa_prime - qsa_prime
			etrace[phisa[0], phisa[1]] = float(args.gamma) * float(args.lmbda) * etrace[phisa[0],phisa[1]] + 1
			agent.weights[phisa[0], phisa[1]] = agent.weights[phisa[0], phisa[1]] + float(args.alpha) * tderror * etrace[phisa[0], phisa[1]]

		state = next_state

	return returns, agent





"""Q Learning Update Algorithm"""
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epsilon','-e',help="suggested value for epsilon")
    parser.add_argument('--alpha','-a',help="suggested value for learning rate alpha")
    parser.add_argument('--gamma','-g',help="suggested value for gamma")
    parser.add_argument('--lmbda','-l',help="suggested value for lambda")
    parser.add_argument('--trials','-tri',help="suggested number of trials to run")
    parser.add_argument('--neps', '-n', default=60, help="suggested number of episodes to run")
    parser.add_argument('--plot', '-p', action='store_true', help="Plot the expected return of rewards versus episodes")
    parser.add_argument('--save','-s',action='store_true',help="save the expected return of rewards for each trial ans episode")
    args = parser.parse_args()


    NEPS = int(args.neps)
    returns = np.zeros((NEPS, int(args.trials)))
    for j in range(int(args.trials)):
        returns, agent = qlambda(args, NEPS, returns, j)

    if args.save:
        with open('%s.pickle' % (args.save),'wb') as f:
            pickle.dump(returns, f)
        
    if args.plot:
        plot(returns, 'QLambda', args, NEPS)
    
import numpy as np
import argparse
from datetime import datetime
import environment as ENV
import sentGenAgent as SGA
from project_utils.utilities import *
import pickle
from collections import defaultdict

"""
Implement the SARSA algorithm

1. Initialize q(s,a) arbitrarily. In our case, initialize weights to 0 and compute the q value.
                                                            (optimistic estimate) - Done in MCA
2. For n episodes, repeat the following:
      i. Choose an initial state s0 at random - Done with initializing Environment ENV
     ii. Choose action a0 from state s0 using q(s,a) initialized above.
    iii. Repeat until reaching the terminal state:
            a. Take the action a0 and get the reward r1 and next state s1 values.
            b. Choose a1 from the new state using the same q(s,a)
            c. Compute q(s1,a1) using the old weights. Also get the intermediate phi(s,a)
            d. Update the weights using the equation : w = w + alpha * (r1 + gamma*q(s1,a1) - q(s,a)) * phi(s,a)
            e. set a0 = a1 and s0 = s1
"""

parser = argparse.ArgumentParser()
parser.add_argument('--epsilon','-e',help="suggested value for epsilon")
parser.add_argument('--alpha','-a',help="suggested value for learning rate alpha")
parser.add_argument('--gamma','-g',help="suggested value for gamma")
parser.add_argument('--neps', '-n', default=200, help="suggested number of episodes to run")
parser.add_argument('--trials','-tri',help="number of trials to run")
parser.add_argument('--plot', '-p', action='store_true', help="Plot the expected return of rewards versus episodes")
parser.add_argument('--save','-s',action='store_true',help="save the expected return of rewards for each trial ans episode")
args = parser.parse_args()

NEPS = int(args.neps)

#log(args)

all_returns = np.zeros((NEPS, int(args.trials)), dtype=float)
start = datetime.now()

"""SARSA Update Algorithm"""
for j in range(int(args.trials)):
    agent = SGA.sentGenAgent(epsilon=float(args.epsilon),alpha=float(args.alpha),gamma=float(args.gamma),fourier_order=1)
    print 'initial weights: ', agent.weights
    print 'starting condition: ', ENV.Enviroment().x, ENV.Enviroment().v
    for i in range(NEPS):
        #print agent.weights
        environment = ENV.Enviroment()
        #print 'Running episode %d/%d' % (i+1, NEPS)
        state = environment.prev_state_id
        action = agent.getAction(state) 
        #print 'action: ', action       
        run = 'run'
        l = 0
        while run == 'run':
            
            qsa, phisa = agent.qValue(state, action)

            run, reward = environment.getNextState(action)
            all_returns[i,j] += (float(args.gamma)**l) *reward
            l += 1
            next_state = environment.next_state_id
            if run == 'terminate':
                qsa_prime = 0
                next_action = action
            else:
                next_action = agent.getAction(next_state)
                qsa_prime, _ = agent.qValue(next_state, next_action)

            agent.updateWeights(reward, qsa_prime, qsa, phisa)
            state = next_state
            action = next_action
            
        if i % 1 == 0:
            print 'Total expected return after episode %d for trial %d:%f ' % (i+1,j,all_returns[i,j])
            print 'Total time steps run: ', environment.t


end = datetime.now()

if args.save:
    with open('all_returns_sarsa_%d.pickle' % (int(args.trials)),'wb') as f:
        pickle.dump(all_returns, f)


if args.plot:
    plot(all_returns, 'sarsa', args, NEPS)




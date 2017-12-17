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
Implement the Q Learning algorithm

1. Initialize q(s,a) arbitrarily. In our case, initialize weights to 0 (optimistic estimate).
2. For n episodes, repeat the following:
      i. Choose an initial state s0 at random - Done with initializing Environment ENV
     ii. Repeat until reaching the terminal state:
            a. Choose action a from state s0.
            b. Take the action a and get reward r1 and next state s1.
            b. Compute q(s1,a') using the old weights for all actions a'. Also get the intermediate phi(s0,a)
            c. Update the weights using the equation : w = w + alpha * (r1 + gamma*max_a'(q(s1,a')) - q(s0,a)) * phi(s0,a)
	    d. s0 = s1
"""



def qlearning(args, NEPS, returns, j):
    agent = SGA.sentGenAgent(epsilon=float(args.epsilon), alpha=float(args.alpha), gamma=float(args.gamma))
    print ('running trial %d' % j)
    np.random.seed()
    for i in range(NEPS):
        #print agent.weights
        environment = ENV.Enviroment()
        #print 'Running episode %d/%d' % (i+1, NEPS)
        state = environment.prev_state_id
        if i % 100 == 0:
            sentence = [environment.prev_state]
        run = 'run'
        l = 0
        while run == 'run':
            action = agent.getAction(state)
            run, reward = environment.getNextState(action)
            #print('reward: ', reward)
            returns[i, j] += (float(args.gamma)**l)*reward
            l += 1
            next_state = environment.next_state_id
            qsa, phisa = agent.qValue(state, action)
            max_qsa_prime = []
            for act in agent.actions.keys():
                if run == 'terminate':
                    qsa_prime = 0.0
                else:
                    qsa_prime,_ = agent.qValue(next_state, act)
                max_qsa_prime.append(qsa_prime)
            max_qsa_prime = max(max_qsa_prime)
            agent.updateWeights(reward, max_qsa_prime, qsa, phisa)
            state = next_state
            if i % 100 == 0:
                sentence.append(action.encode('utf8'))
        if i % 100 == 0:
            try:
                print('\nSample sentence after {} epochs: {}'.format(i, ' '.join(sentence)))
            except:
                pass
    return returns, agent

   
def sample(agent):
    sample = []
    environment = ENV.Enviroment()
    state = environment.prev_state_id
    sample.append(environment.prev_state)
    run = 'run'
    l = 0
    while run == 'run':
        action = agent.getAction(state)
        sample.append(action)
        run, reward = environment.getNextState(action)
        l += 1
        next_state = environment.next_state_id
        qsa, phisa = agent.qValue(state, action)
        max_qsa_prime = []
        for act in agent.actions.keys():
            if run == 'terminate':
                qsa_prime = 0.0
            else:
                qsa_prime,_ = agent.qValue(next_state, act)
            max_qsa_prime.append(qsa_prime)
        max_qsa_prime = max(max_qsa_prime)
        state = next_state
    return returns, sample






"""Q Learning Update Algorithm"""
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epsilon','-e',help="suggested value for epsilon")
    parser.add_argument('--alpha','-a',help="suggested value for learning rate alpha")
    parser.add_argument('--gamma','-g',help="suggested value for gamma")
    parser.add_argument('--trials','-tri',help="suggested number of trials to run")
    parser.add_argument('--neps', '-n', default=60, help="suggested number of episodes to run")
    parser.add_argument('--plot', '-p', action='store_true', help="Plot the expected return of rewards versus episodes")
    parser.add_argument('--save','-s',action='store_true',help="save the expected return of rewards for each trial ans episode")
    args = parser.parse_args()
    
    start = datetime.now()
    NEPS = int(args.neps)
    #folder = tempfile.mkdtemp()
    #returns_name = os.path.join(folder, 'returns')
    #returns = np.memmap(returns_name, dtype=np.float32, shape=(NEPS, int(args.trials)), mode='w+')

    #Parallel(n_jobs=4)(delayed(qlearning)(args, NEPS, returns, i) for i in range(int(args.trials)))
    returns = np.zeros((NEPS, int(args.trials)))
    for j in range(int(args.trials)):
        returns, agent = qlearning(args, NEPS, returns, j)
    end = datetime.now()
    print('Total time take: ', end-start)
    print(returns)
    if args.save:
        with open('%s.pickle' % (args.save),'wb') as f:
            pickle.dump(returns, f)
        
    if args.plot:
        plot(returns, 'QLearning', args, NEPS)

    sample_sent = sample(agent)
    print('sample sentence: ', sample_sent)
    print('Bigram probability of the sentence: ', evaluate_sentence(sample_sent))



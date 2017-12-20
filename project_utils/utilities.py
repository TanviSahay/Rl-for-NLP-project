import numpy as np
import os
import matplotlib.pyplot as plt
import warnings
import pickle
from collections import defaultdict
from nltk import pos_tag, word_tokenize
warnings.simplefilter("ignore")


def dd():
    return defaultdict(int)

def get_actions():
    with open('./Data/vocab.pkl','rb') as f:
        actions = pickle.load(f)
    actions = {k:i for i,k in enumerate(actions)}
    return actions

def getReward(reward_func):
    if reward_func == 1:
        #print('Reward will be: word-word co-occurrence')
        return word_cooc_reward()
    if reward_func == 2:
        #print('Reward will be: pos-pos co-occurrence')
        return pos_cooc_reward()
    if reward_func == 3:
        #print('Reward will be: product of word-word and pos-pos cooccurrence')
        return word_pos_reward('prod')
    if reward_func == 4:
        #print('reward will be: average of word-word and pos-pos cooccurrence')
        return word_pos_reward('avg')

def word_cooc_reward():
    with open('./Data/word_cooccurrence.pkl','rb') as f:
        return pickle.load(f)

def pos_cooc_reward():
    with open('./Data/pos_cooccurrence.pkl','rb') as f:
        return pickle.load(f)


def word_pos_reward(combine):
    if os.path.exists('./Data/word_pos_%s'%combine):
        with open('./Data/word_pos_%s'%combine,'rb') as f:
            rewards = pickle.load(f)
    else:
        with open('./Data/pos_cooccurrence.pkl','rb') as f:
            pos_cooc = pickle.load(f)
        with open('./Data/word_cooccurrence.pkl','rb') as f:
            word_cooc = pickle.load(f)
        rewards = defaultdict(dd)
        for key, val in word_cooc.items():
            for word, score in val.items():
                bigram = [key, word]
                tagged_bigram = pos_tag(bigram)
                if combine == 'prod':
                    rewards[key][word] = pos_cooc[tagged_bigram[0][1]][tagged_bigram[1][1]] * score
                if combine == 'avg':
                    rewards[key][word] = (pos_cooc[tagged_bigram[0][1]][tagged_bigram[1][1]] + score) / 2
        with open('./Data/word_pos_%s.pickle'%combine, 'wb') as f:
            pickle.dump(rewards, f)
    return rewards


#def scale(val, old_min, old_max, new_min, new_max):
#    new_val = (val - old_min)/(old_max - old_min)
#    return new_val


#def count(number, base, shape):
#    c = np.zeros(shape=shape)

#    i = c.shape[0] - 1
#    while number >= base:
#        remainder = number % base
#        c[i] = remainder
#        i -= 1
#        number = number / base
#    if number != 0 and number < base:
#        c[i] = number
#    return c


def plot(data, method, trials, NEPS,eps,alp,g):	
    mean = np.mean(data, axis=1)
    #print mean.shape
    variance = np.mean(np.square(data.T-mean).T, axis=1)
    #print variance
    std = np.sqrt(variance)
    #print std
    x = list(np.arange(0,NEPS,1))
    y = list(mean)
    print 'Length of x: {}   length of y: {}'.format(len(x), len(y))
    err = list(std)
    plt.axis((0,NEPS,0,15))
    plt.errorbar(x, y, yerr=err, fmt='-ro')
    #plt.plot(y)
    plt.xlabel('Episode')
    plt.ylabel('Expected return of reward')
    plt.title('%s for %d trials, epsilon: %.4f, alpha: %.2f, gamma: %.2f' % (method, trials, float(eps), float(alp), float(g)))
    plt.savefig('Expected_Return_%s_%d_unclipped.jpg' % (method, trials))
    plt.show()
    return mean[-1]


def log(method, trials, eps, gamma, alpha, maxima=None, time=0):
    if os.path.exists('log'):
        with open('log','r') as f:
            data = f.readlines()
        data.append('method: {0}, trials: {1}, epsilon: {2}, gamma: {3}, alpha: {4}, maximum value: {5}, time taken: {6}\n'.format(method, trials, eps, gamma, alpha, maxima, time))
    else:
        data = 'method: {0}, trials: {1}, epsilon: {2}, gamma: {3}, alpha: {4}, maximum value: {5}, time taken: {6}\n'.format(method, trials, eps, gamma, alpha, maxima, time)

    with open('log','w') as f:
        for line in data:
            f.write(line)

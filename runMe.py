import argparse
import imp
import itertools
from project_utils.utilities import  *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model','-m',help='rl algorithm to run')
    parser.add_argument('--rewards','-r',help="reward function to choose")
    parser.add_argument('--trials','-tri',help="suggested number of trials to run")
    parser.add_argument('--neps', '-n', default=60, help="suggested number of episodes to run")
    parser.add_argument('--plot', '-p', action='store_true', help="Plot the expected return of rewards versus episodes")
    parser.add_argument('--save', '-s', action='store_true',
                        help="save the expected return of rewards for each trial ans episode")
    parser.add_argument('--predict','-pr',action='store_true',help='predict the scores for the trained model')
    args = parser.parse_args()

    reward = int(args.rewards)
    neps = int(args.neps)
    trials = int(args.trials)

    print('Initializing model: ', args.model)
    model_source = imp.load_source(args.model, './Models/%s/%s.py' % (args.model, args.model))
    params = './Models/%s/%s-params.json' % (args.model,args.model)
    Model = model_source.model(params, reward)

    hyperparameters = Model.hyperparams
    permutations = itertools.product(*hyperparameters)
    best_params = None
    best_score = 0.0
    for i, perm in enumerate(permutations):
        print('Beginning training')
        print('Running permutation {}'.format(i))
        returns, time = Model.fit(neps, trials, perm)
        print 'model has been fit, time taken: ', time
        if args.plot:
            plot(returns, args.model, trials, neps, perm[0], perm[1], 1.0)
        print returns

        reward_id = {1: 'word_cooc', 2: 'pos_cooc', 3:'word_pos_prod', 4: 'word_pos_avg'}

        if args.predict:
            print('Computing sentence scores for sentences generated using the final trained model')
            score = Model.predict()
            print('Final Bigram Probability score averaged over 1000 sentences: ', score)
        #score = 100
        if score > best_score:
            best_score = score
            best_params = perm
            if args.save:
                print 'Saving permutation'
                name = '{}-{}-n{}-t{}-perm'.format(args.model, reward_id[reward], args.neps, args.trials)
                for val in perm:
                    name += '_'
                    name += str(val)
                with open(os.path.join('./saved_models/%s' % (args.model), '%s.pickle' % name), 'wb') as f:
                    pickle.dump(returns, f)

    print 'Best score: ', best_score
    print 'Best Parameters: ', best_params



#, 0.001, 0.0001


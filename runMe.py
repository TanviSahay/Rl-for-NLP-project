import argparse
import imp
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
    print('Beginning training')
    returns = Model.fit(neps, trials)

    if args.plot:
        plot(returns, args.model, trials, neps)

    reward_id = {1: 'word_cooc', 2: 'pos_cooc', 3:'word_pos_prod', 4: 'word_pos_avg'}
    if args.save:
        name = '{}-{}-n{}-t{}'.format(args.model, reward_id[reward], args.neps, args.trials)
        with open(os.path.join('./saved_models/%s' % (args.model), '%s.pickle' % name), 'wb') as f:
            pickle.dump(returns, f)

    if args.predict:
        print('Computing sentence scores for sentences generated using the final trained model')
        score = Model.predict()
        print('Final Bigram Probability score averaged over 1000 sentences: ', score)







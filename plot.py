from project_utils.utilities import *
import pickle
import sys
import os
import re


if __name__ == '__main__':
	datapath = sys.argv[1]
	with open(datapath, 'rb') as f:
		data = pickle.load(f)
	print 'Data shape: {}'.format(data.shape)
	name = os.path.basename(datapath)
	model = re.findall('(.*)\.pickle',name)[0]
	neps, trials, eps, alp, g = re.findall('.*-.*-n(.*)-t(.*)-perm_(.*)_(.*)_(.*)_?.*?\.pickle',name)[0]
	print 'Number of episodes: {}, number of trials: {}'.format(neps, trials)
	plot(data, model, int(trials), int(neps),eps,alp,g)



from project_utils.utilities import *
import pickle
import sys
import os
import re


if __name__ == '__main__':
	datapath = sys.args[1]
	with open(datapath, 'rb') as f:
		data = pickle.load(f)

	name = os.path.basename(datapath)
	model = re.findall('(.*)\.pickle',name)[0]
	neps, trials = re.findall('.*-.*-n(.*)-t(.*)\.pickle',name)[0]
	plot(data, model, int(neps), int(trials))



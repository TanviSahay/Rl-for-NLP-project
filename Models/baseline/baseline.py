import pickle
import numpy as np
import json


class model:

	def __init__(self, params, reward):

		with open(params,'r') as f:
			self.params = json.load(f)

		input_file = self.params['inputfile']
		with open(input_file,'r') as f:
			self.text = f.read().strip().decode('utf-8').lower()


	def cleandata(self, text):
		clean_text = []
		for char in text:
			if char in [',','-','_','@','#','$','%','&',';','"',"'",':',')','(','\u201d','\u2018','\u2019','\u201c']:
				pass
			elif char in ['!','?']:
				clean_text.extend(['.'])
			else:
				#print char
				clean_text.extend([char])

		return ''.join(clean_text)	


	def get_sentences(self, text):
		tokenized_sentences = []
		sentences = text.split('.')
		#print sentences
		for sent in sentences:
			#print sent.split()
			if sent != '':
				tokenized_sentences.append(sent.split() + ['.'])
  		return tokenized_sentences


	def fit(self, neps, tri):
		#No need for neps and tri
		clean_text = self.cleandata(self.text)
		self.sentences = self.get_sentences(clean_text)
		#print self.sentences
		return 0


	def predict(self):
		sentences = np.random.choice(self.sentences, 1000)
		return self.score(sentences)


	def score(self, sentences):
		average_score = 0.0
		for sent in sentences:
			score = self.bigram_probability(sent[0:10])
			average_score += score
			print('bigram probability of sentence: ', score)

		return average_score / float(len(sentences))


	def bigram_probability(self,sent):
		with open('./Data/bigram_probability.pkl','rb') as f:
			bigram_occurrence = pickle.load(f)
		bigram_prob = 1.0
		for i in range(len(sent) - 1):
			bigram_prob *= bigram_occurrence[sent[i]][sent[i+1]]
		return bigram_prob









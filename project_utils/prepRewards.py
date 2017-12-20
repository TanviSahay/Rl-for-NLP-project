import os
import pickle
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
import json
import copy
import math
import argparse
from collections import defaultdict, Counter




def dd():
    return defaultdict(float)


def gettext(inpath):
    text = ''
    files = os.listdir(inpath)
    for file in files:
        with open(os.path.join(inpath, file), 'r') as f:
            text += f.read().decode("utf-8").strip().lower()
            text += ' '
    return text


def clean_data(text):
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



def limit_vocab(text, limit):
    tokens, _ = get_tokenized_text(text)
    vocab = dict(Counter(tokens))
    sorted_vocab = sorted(vocab, key=vocab.__getitem__, reverse = True)[0:limit]

    limited_text = []
    for token in tokens:
        if token in sorted_vocab or token == '.':
            limited_text.append(token)
        else:
            limited_text.append('unk')
    return ' '.join(limited_text)



def get_tokenized_text(text):
    tokenized = []
    tokenized_sentences = []
    sentences = text.split('.')
    for sent in sentences:
        if sent != '':
            tokenized.extend(sent.split() + ['.'])
            tokenized_sentences.append(sent.split() + ['.'])
    return tokenized, tokenized_sentences


def get_unigram_counts(tokens):
    i = 0
    unigram_counts = defaultdict(int)
    while i < len(tokens):
        unigram_counts[tokens[i]] += 1
        if i % 1000 == 0:
            try:
                print 'Token: {} Count: {}'.format(tokens[i], unigram_counts[tokens[i].encode("utf-8")])
            except:
                pass
        i += 1
    return unigram_counts


def get_bigram_counts(tokens):
    i = 0
    bigram_counts = defaultdict(dd)
    while i < len(tokens) - 1:
        bigram_counts[tokens[i]][tokens[i+1]] += 1
        i += 1
    return bigram_counts


def get_pos_unigram_counts(sentences):
    tag_unigram_counts = defaultdict(int)
    for i,sent in enumerate(sentences):
        tagged_sent = pos_tag(sent)
        if i % 1000 == 0:
            print('\n\tNumber of sentences done for unigrams: {}/{}'.format(i, len(sentences)))
            print('\tSample tagging: {}'.format(tagged_sent))
        for word, tag in tagged_sent:
            tag_unigram_counts[tag] += 1
    return tag_unigram_counts


def get_pos_bigram_counts(sentences):
    tag_bigram_counts = defaultdict(dd)
    for j,sent in enumerate(sentences):
        tagged_sent = pos_tag(sent)
        if j % 1000 == 0:
            print('\n\tNumber of sentences done for bigrams: {}/{}'.format(j, len(sentences)))
            print('\tSample tagging: {}'.format(tagged_sent))
        i = 0
        while i < len(tagged_sent) - 1:
            tag_bigram_counts[tagged_sent[i][1]][tagged_sent[i+1][1]] += 1
            i += 1
    return tag_bigram_counts



def getreward(counts, bigram_counts):
    rewards = copy.deepcopy(bigram_counts)
    for key, val in bigram_counts.items():
        for word, score in val.items():
            rewards[key][word] = math.ceil((score / float(counts[key])) * 10)
            print score, counts[key]
    return rewards


def get_bigram_prob(unigram_counts, bigram_counts):
    for key, val in bigram_counts.items():
        for word, score in val.items():
            #print score, unigram_counts[key]
            bigram_counts[key][word] =  score / float(unigram_counts[key])
    return bigram_counts


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab',help='limit to vocabulary')
    args = parser.parse_args()


    vocab_limit = int(args.vocab)

    text = gettext('./Data/Text_Corpus')
    print('Text data loaded.\nTokenizing Sentences')

    #clean the data
    clean_text = clean_data(text)

     #limit vocab
    limited_text = limit_vocab(clean_text, vocab_limit)

    #tokenize the data
    tokens, tokenized_sentences = get_tokenized_text(limited_text)

    #get vocab
    vocab = list(set(tokens))


    print('Extracting unigram and bigram counts of words')
    #get unigram counts of words
    unigram_counts = get_unigram_counts(tokens)
    #get bigram counts of words
    bigram_counts = get_bigram_counts(tokens)


    #print bigram_counts['trims']

    #get word-cooccurrence based rewards
    word_cooc_scores = getreward(unigram_counts, bigram_counts)


    #print bigram_counts['trims']


    print('Extracting word cooccurrence probabilities')
    #get probability of word cooccurrence
    word_cooc_prob = get_bigram_prob(unigram_counts, bigram_counts)

    print('Extracting unigram and bigram counts of pos tags')
    #get unigram counts of pos tags
    tag_unigram_counts = get_pos_unigram_counts(tokenized_sentences)
    #get bigram counts of pos tags
    tag_bigram_counts = get_pos_bigram_counts(tokenized_sentences)
    #get pos tag - cooccurrence based rewards
    pos_cooc_scores = getreward(tag_unigram_counts, tag_bigram_counts)

    with open('./Data/word_cooccurrence.pkl','wb') as f:
        pickle.dump(word_cooc_scores, f)

    with open('./Data/pos_cooccurrence.pkl','wb') as f:
        pickle.dump(pos_cooc_scores, f)

    with open('./Data/bigram_probability.pkl','wb') as f:
        pickle.dump(word_cooc_prob, f)

    with open('./Data/vocab.pkl','wb') as f:
        pickle.dump(vocab,f)

    print('Limiting vocabulary of input to {} words'.format(vocab_limit))
    print('Total number of tokens: {}'.format(len(tokens)))

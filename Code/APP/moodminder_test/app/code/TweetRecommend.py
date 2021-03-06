#!/usr/local/bin/python3

"""
Read tweet and figure out which category in database is most similar
PROCEDURE
1. Extract keyword from tweet
2. Figure out which category matched closely with keywords
"""

import sys
import nltk
from nltk import word_tokenize as wt 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

def tweet_filter(tweet):
	# Tokenize tweet
	tweet_token = wt(tweet)
	#print(tweet_token)

	# Remove stop words
	tweet_token = [word for word in tweet_token if word not in stopwords.words('english')]
	#print(tweet_token)

	# POS tagging
	tweet_POS = nltk.pos_tag(tweet_token)
	#print(tweet_POS)

	# Extract nouns, adjectives and adverbs
	tweet_NN = [word for word,pos in tweet_POS if 'NN' in pos or 'JJ' in pos or 'RB' in pos]
	#print(tweet_NN)

	return tweet_NN

# Compute similarity scores with current category names and choose category name with max score
sim_scores_c = {}
def get_category(words, categories):
	sim = wn.path_similarity
	max_cw = (0, '') # (similarity score, category)
	for c in categories:
		sim_value_c = 0
		sim_scores_c[c] = 0
		for w in words:
			synsets1 = wn.synsets(c, pos='n')
			synsets2 = wn.synsets(w, pos='n')
			sim_scores = []
			for synset1 in synsets1:
				for synset2 in synsets2:
					sim_value = sim(synset1, synset2)
					if sim_value:
						sim_scores.append(sim_value)
						sim_value_c += sim_value
		
			#if len(sim_scores):
			#	max_sim_score = max(sim_scores)
			#	if max_sim_score > max_cw[0]:
			#			max_cw = (max_sim_score, c)

		# Add scores for each category
		sim_scores_c[c] = sim_value_c
		if sim_value_c > max_cw[0]:
				max_cw = (sim_value_c, c)

	return (max_cw[0], max_cw[1]) # (similarity score, category)

if __name__ == '__main__':
	categories = ['music', 'food']

	# Get tweet
	tweets = ['I hate food']
	for tweet in tweets:
		for s in [wn.path_similarity,
				  #wn.lch_similarity,
				  #wn.wup_similarity,
				  #wn.res_similarity,
				  #wn.jcn_similarity,
				  #wn.lin_similarity
				  ]:

				#get_category(tweet_filter(tweet), categories, s).upper()
				print tweet, get_category(tweet_filter(tweet), categories, s).upper()




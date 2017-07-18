"""
Feature extractor from tweets
Author: Yamin Tun
"""

import nltk as nk
import re
import pickle

class Feat_extractor:

    """
    ngramType- integer value
    1 for unigram
    2 for bigram

    data- dataframe
    """
    
    def __init__(self, ngramType=1, train_feat=None):
        self.ngramType=ngramType
        self.train_feat=train_feat

    """Process one tweet:
    -strips of URL, twitter usernames and punctuations
    -if unigrams, remove words with less than 3 characters
    -if bigrams, extract bigrams and append the unigram list
    """
    def process_onetweet(self,tweet):
        
        words= re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweet).split()

        raw_feat= [e.lower() for e in words if len(e)>=3]
        
        if self.ngramType==2:
            bigrams=[' '.join(biTuple) for biTuple in nk.bigrams(words)]
            raw_feat=raw_feat+bigrams            

        return raw_feat

    """Process multiple tweets"""
    def process_many(self,tweets_list):
        print ('    Processing...')

        tweets=[];
        
        for(tweet,sentiment) in tweets_list:
            words_filt=self.process_onetweet(tweet)
            tweets.append((words_filt, sentiment))

        print('     done')
        return tweets
    
    """Extract features in a form of a binary dictionary of whether contains words"""
    def get_containWords(self,tweet):        
        tweet_words = set(tweet)
        
        features={}
        for word in self.train_feat:
            features['contains(%s)' % word]= (word in tweet_words)

        return features

    """(Training)
       Extract list of distict words 
       This step is required only for training process
        -tweets: [(words, sentiment),...]
    """
    def get_WordsSet(self,tweets_words):
        wordlist=[]

        for(words, sentiment) in tweets_words:
            wordlist.extend(words)           

        #group words by frequency and extract unique words (take key)
        wordlist= nk.FreqDist(wordlist)
        word_features=wordlist.keys()
        return word_features

    # ---- last min addition ---- #
    """Extract and apply features and save features in a pickle file (for train data only)"""
    def extractApplyFeatTRAIN_many(self,raw_tweets,featDir=''):
        tweets_words=self.process_many(raw_tweets)
        
        #extract features
        
        self.train_feat=self.get_WordsSet(tweets_words)

        if len(featDir)>0: #save features
            pickle.dump(self.train_feat, open(featDir, 'wb'))

        print ('    Extracting/Applying Features...')

        train_set= nk.classify.apply_features(self.get_containWords,tweets_words)
 
        print('     done')
        return train_set


    """Extract features from many tweets (for test data only)"""
    def extractFeatTEST_many(self,raw_tweets):

        print ('    Extracting Features...')
        tweets_words=self.process_many(raw_tweets)

        tweetsOnly=[tweet[0] for tweet in tweets_words]
            
        test_feat=[self.get_containWords(tweet) for tweet in tweetsOnly]
        print('     done')
        return test_feat
        
        


        




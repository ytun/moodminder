import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split


"""
Util class for emotion classification
Author: Yamin Tun

"""

pos_label=4
neg_label=0

"""Load the data"""
def load_data(fileDir, subSize=0,SEED=42, useC=[0,5],filtNeutral=True):
    print ("Loading Data...")
    dataFrame = pd.read_csv(fileDir,  sep=",",usecols=useC, names=['label','tweet'], index_col=False,dtype={'label': np.int32, 'tweet': np.str_})

    if filtNeutral:
        dataFrame=dataFrame[(dataFrame.label==pos_label) | (dataFrame.label==neg_label)]
    
    tweets_list =list(dataFrame.itertuples(index=False))

    tweets_list=zip([tweet_pair[1] for tweet_pair in tweets_list],[tweet_pair[0] for tweet_pair in tweets_list])


    if subSize>0:
        tweets_list=subSample(tweets_list,subSize,SEED)
    
    print ("done")
    return tweets_list

"""Split the data into train and test
    return: X_train, X_test, y_train, y_test"""
def split(data,train_size=0.8,seed=42):
    print ('Splitting into train and test')

    
    X=[tweet_pair[0] for tweet_pair in data]
    y=[tweet_pair[1] for tweet_pair in data]
    
    X_train, X_test, y_train, y_test= train_test_split(X, y, train_size=train_size, random_state=seed)
    
    train_pair_list=zip(X_train, y_train)
    test_pair_list=zip(X_test, y_test)

    print('done')   
    return train_pair_list,test_pair_list,y_test,X_test

"""Get subsize of the data with equal number of positive and negative tweets"""
def subSample(tweets_list,subSize,SEED=42):
    print('Subsampling the data...')

    import random
    random.seed(SEED)
    random.shuffle(tweets_list)

    if subSize<=1:
        labelSet_size=int((len(tweets_list)*subSize)/2) #equal size of two classes
        subSizeLen=int(len(tweets_list)*subSize)
    else:
        labelSet_size=int(subSize*0.5)
        subSizeLen=subSize

    
    pos_tweets=filter(lambda x: x[1] == 4, tweets_list)[:labelSet_size]
    neg_tweets=filter(lambda x: x[1] == 0, tweets_list)[:labelSet_size]

##    print len(tweets_list), len(pos_tweets), len(neg_tweets), labelSet_size, subSizeLen

    assert len(pos_tweets)==labelSet_size and len(neg_tweets)==labelSet_size
    assert (all(emo== 0 for tweet, emo in neg_tweets))
    assert (all(emo== 4 for tweet, emo in pos_tweets))

    print "Subsize: %i with pos and neg size: %i"%(subSizeLen, labelSet_size)

    print('done')
    return (pos_tweets + neg_tweets)



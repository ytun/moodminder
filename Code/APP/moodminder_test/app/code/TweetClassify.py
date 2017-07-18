import sys, os, time
import nltk as nk
from Feat_extractor import *
from Util_data import *
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_score, accuracy_score, recall_score
from sklearn.svm import SVC, LinearSVC


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import precision_score, accuracy_score, recall_score
from sklearn import cross_validation

import nltk
"""
Emotion Classifier

Author: Yamin Tun

NOTE: Even though we are aware that the if-statement in the constructor violates open-close
principles, we didn't use any design pattern here for different classifiers
because:
1) we are using a third-party library for classification
2) we are trying no more than 2-3 classifiers. We don't want to overengineer
for a simple experiment 
"""


from nltk.classify import maxent
nk.classify.MaxentClassifier.ALGORITHMS
# ['GIS','IIS','CG','BFGS','Powell','LBFGSB','Nelder-Mead','MEGAM','TADM']

# MEGAM or TADM are not rec'd for text classification

import scipy

print(__doc__)


class TweetClassify:    
                
    def __init__(self,classifier=None,ngramType=-1,train_feat=None,classifierType=''):

        self.classifier=classifier

        if len(classifierType)>0:
            self.classifierType=classifierType
        
        self.ngramType=ngramType
        self.f=Feat_extractor(ngramType=ngramType,train_feat=train_feat)

    def setClassifier(self,classifier):
        self.classifier=classifier
        
    def getDataFeat(self,dataDir,featDir,train_size,subSize=0):
        #create an empty folder for features if not exists yet
        if len(featDir)>0:
            allDir=os.path.dirname(featDir)
            if not os.path.exists(allDir):
                print 'making dir @'
                print allDir
                os.makedirs(allDir)
            
        data=load_data(dataDir,subSize=subSize)
        
        train_tweets, test_tweets, test_gold,raw_test_tweets = split(data, train_size=train_size)

        print ("    Training set:")
        train_set=self.f.extractApplyFeatTRAIN_many(train_tweets,featDir=featDir)
        print ("    Test set:")        
        test_set=self.f.extractFeatTEST_many(test_tweets)

        return train_set, test_set, test_gold, raw_test_tweets

    def train(self, train_set,allDir='',classifierFileName='',timeit=True):
            
        if timeit:
            t0 = time.time()

        print "\nTraining..."

        if self.classifierType=='maxen':
            classifier= self.classifier.train(train_set)#,'GIS', trace=0, max_iter=100)
        else:
            classifier= self.classifier.train(train_set) #add here
        
        if timeit:
            elapsed_time = time.time() - t0
            print "elapsed_time: ",elapsed_time
##            np.savetxt(allDir+'/time_'+classifierFileName, np.array([ elapsed_time]), fmt='%f')

        #SAVE classifier
        if len(classifierFileName)>0:
            pickle.dump(classifier, open(allDir+classifierFileName, 'wb'))

        self.classifier=classifier

        print "done"
        return classifier,elapsed_time

        
    def predict_many(self, test_set,gold_list,raw_test_set,train_time,allDir, testresultFileName="", target_names=['negative','positive']):
    
        pred_list=[]
       
        pred_list= self.classifier.classify_many(test_set)
    
        if len(testresultFileName)>0:
            result= self.saveResult(pred_list, gold_list,raw_test_set,allDir+testresultFileName)

        
        accuracy=accuracy_score(gold_list, pred_list)
        print accuracy
        
        return pred_list,accuracy

    def predict(self,tweet):

        tweet=self.f.process_onetweet(tweet)

        #print "predicting..."
        pred= self.classifier.classify(self.f.get_containWords(tweet))

        return pred

    def saveResult(self,pred_list,gold_list,raw_test_set,testresultDir):

        combo_result= {'gold':pd.Series(gold_list) ,\
                       'pred':pd.Series(pred_list), \
                       'tweet':pd.Series([('"'+tweet+'"') for tweet in raw_test_set]) }

       
        result=pd.DataFrame(combo_result)
        np.savetxt(testresultDir, result, fmt='%i,%i,%s')

        if(len(result)>=3):
            print "\nresult_frame: \n", result.iloc[:2]
        else:
            print "\nresult_frame: \n", result
            
        return result
    
    

    def cross_val(self,train_set):
    
        print "\nCross-validating..."

        print "train size: ",len(train_set)
        cv = cross_validation.KFold(len(train_set), n_folds=5, shuffle=True, random_state=None)

        acc_list=[]
        
        for traincv, evalcv in cv:
            classifier = self.classifier.train(train_set[traincv[0]:traincv[len(traincv)-1]])
            acc= nk.classify.util.accuracy(classifier, train_set[evalcv[0]:evalcv[len(evalcv)-1]])
            print 'accuracy: %.3f' % acc
            acc_list=acc_list+[acc]

        avgAcc=np.average(acc_list)

        print 'AVERAGE accuracy: %.3f' % avgAcc
        
        print "done"
        return avgAcc

    
    """Cross-validation"""
    def cross_valid(self,Dir,subSize,num_folds=5):

        print "cross validing..",num_folds, " fold"
        tweets=load_data(Dir,subSize=subSize)
        import random
        random.shuffle(tweets)

        trainSize=len(tweets)
        subset_size = int(trainSize/num_folds)

        acc_list=[]
        assert len(tweets)==trainSize

        for i in range(num_folds):
            test_set = tweets[i*subset_size:][:subset_size]
            train_set = tweets[:i*subset_size] + tweets[(i+1)*subset_size:]

            gold_list= [ t[1] for t in test_set]
            raw_test_set=[ t[0] for t in test_set]
            
            print ("    Training set:")
            train=self.f.extractApplyFeatTRAIN_many(train_set)
            print ("    Test set:")        
            test=self.f.extractFeatTEST_many(test_set)
           
            ##TESTING   
            print "training round: ",i,"..."

            print len(train), " test: ",len(test)
            
            assert len(train)==subSize-subset_size
            assert len(test)==subset_size
            
            self.train(train)
            
            print "testing..."
            pred_list, acc=self.predict_many(test,gold_list,raw_test_set,0,allDir='',testresultFileName='')

            print "accuracy: ", acc, "\n"
            acc_list.append(acc)
            


        avgAcc=np.average(acc_list)
        print "Finished!"

        print "Mean Accuracy after CV: ", avgAcc #reduce(lambda x, y: x + y, acc_list) / len(acc_list)
        


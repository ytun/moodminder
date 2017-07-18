"""Experiment 1: Cross-Validation for NB and SVC with unigrams only and unigram+bigram"""

import sys
sys.path.insert(0, './../')

import nltk
from Classifier import *
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import LinearSVC

import nltk
from nltk.classify import maxent
nk.classify.MaxentClassifier.ALGORITHMS

def main():

    #Set parameters for the experiment
    subSize=10
    train_size=1
    ngramType=2


    s='crossVal'#'GIS100'          

    #####
    featDir=""

    model_list=['nb','svc']#,'maxen']
    
    for classifierType in model_list:

        experimentName=s+"_%s_%i_%.2f_%igram"%(classifierType,subSize,train_size,ngramType)
     #   experimentName="stanford_%i_%.2f_%igram"%(subSize,train_size,ngramType)
        
        print experimentName
        dataDir="./../stanford_data/train.csv"
        allDir='./../RESULTS/'+experimentName+"/"
        

        if ngramType==1:
            featDir=allDir+"feature_"+`train_size`+'.pickle'
            classifierFileName=classifierType+'_'+`train_size`+'.pickle'
            testresultFileName='testResult_'+`train_size`+'.txt'
            reportFileName='report_'+`train_size`+'.txt'
        elif ngramType==2:
            featDir=allDir+"bi_feature_"+`train_size`+'.pickle'
            classifierFileName="bi_"+classifierType+'_'+`train_size`+'.pickle'
            testresultFileName='bi_testResult_'+`train_size`+'.txt'
            reportFileName='bi_report_'+`train_size`+'.txt'
            
        if classifierType=='nb':
            print "Using Naive Bayes"
            classifier=nltk.NaiveBayesClassifier
        elif classifierType=='svc':
            print "Using SVC"
            classifier=SklearnClassifier(LinearSVC())
        elif classifierType=='maxen':
            print "Using Max Entropy"
            classifier = nltk.classify.MaxentClassifier

            
        #Run the experiment
        c=Classifier(classifier=classifier,ngramType=ngramType,classifierType=classifierType) 

        c.cross_valid(dataDir,subSize=subSize,num_folds=3)

 
if __name__=='__main__':
    main()

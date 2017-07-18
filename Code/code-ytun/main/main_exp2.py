"""Experiment 2: comparison among svc and nb in terms of training time, precision, accuracy, recall"""

import sys
sys.path.insert(0, './../')

from Util_plot import *

def main():    
    
    #Set parameters for the experiment
    subSize=2000
    train_size=0.8
    ngramType=2

 #   classifierType='maxen'#'maxen' ##CHANGE HERE

    s='dummy_graph'#'GIS100'
    
 #   experimentName="stanford_%i_%.2f_%igram"%(subSize,train_size,ngramType)
    
    
    dataDir="./../stanford_data/train.csv"
       

    #####

    model_list=['nb','svc']#,'maxen']

    trainT_mat=np.zeros(len(model_list))
    accuracy_mat=np.zeros(len(model_list))
    precision_mat=np.zeros(len(model_list))
    recall_mat=np.zeros(len(model_list))

    
    i=0
    
    for classifierType in model_list:
        experimentName=s+"_%s_%i_%.2f_%igram"%(classifierType,subSize,train_size,ngramType)
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

        train_set , test_set , test_gold,raw_test_set= c.getDataFeat( dataDir,featDir,train_size,subSize=subSize)      
        cl, train_time=c.train(train_set,allDir,classifierFileName,timeit=True)
        trainT_mat[i]=train_time
        pred_list,accuracy=c.predict_many(test_set, test_gold,raw_test_set,train_time,allDir,testresultFileName)

        if classifierType=='nb':
            precision,accuracy,recall=report(test_gold,pred_list,train_time,allDir+reportFileName,classifierType=classifierType,nbclassifier=cl)
        else:
            precision,accuracy,recall=report(test_gold,pred_list,train_time,allDir+reportFileName,classifierType)

        precision_mat[i]=precision 
        accuracy_mat[i]=accuracy 
        recall_mat[i]=recall 

        i=i+1
        
    result_mat= trainT_mat# np.vstack((trainT_mat,testT_mat))
    param_list=['trainT']
    title="Comparison between classifiers in terms of training times"
    ylabel='seconds'
    plot_bar(model_list,result_mat,title,param_list,'upper left',ylabel)

    result_mat=np.vstack((accuracy_mat,precision_mat,recall_mat))
    param_list=['accuracy','precision','recall']
    title="Comparison between classifiers in terms of accuracy, precision, recall"
    log=0
    ylabel='score'
    plot_bar(model_list,result_mat,title,param_list,'upper left',ylabel,[0,1])

            ###############################################################################
            # Quantitative evaluation of the model quality on the test set

    plt.show()

if __name__=='__main__':
    main()

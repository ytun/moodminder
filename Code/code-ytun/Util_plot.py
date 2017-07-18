
"""
Util for Plotting for Emotion Classifer
Author: Yamin Tun
"""

import sys
sys.path.insert(0, './../')

import nltk
from Classifier import *
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import LinearSVC

import nltk
from nltk.classify import maxent
nk.classify.MaxentClassifier.ALGORITHMS

import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, target_names, title, cmap=plt.cm.Blues):
    plt.imshow(cm.T, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('Predicted label')
    plt.xlabel('True label')


def report(gold_list,pred_list,train_time,reportDir,classifierType='',nbclassifier=None,target_names=['negative','positive']):
    precision=precision_score(gold_list, pred_list,pos_label=4)
    accuracy=accuracy_score(gold_list, pred_list)
    recall=recall_score(gold_list, pred_list,pos_label=4)

    print("Accuracy: ",accuracy)
    
    orig_stdout = sys.stdout
    f = file(reportDir, 'w')
    sys.stdout = f

    print "train time: ",train_time
    
    print("Precision: ", precision)
    print("Accuracy: ",accuracy)
    print("Recall: ",recall)
    
    print classification_report(gold_list, pred_list, target_names=target_names)  # or f.write('...\n')

    # Compute confusion matrix
    cm = confusion_matrix(gold_list, pred_list)
    np.set_printoptions(precision=2)
    print('Confusion matrix, without normalization')
    print(cm)
    
    if classifierType=='nb':
        print "\nImportant features: \n"
        nbclassifier.show_most_informative_features(30)

    sys.stdout = orig_stdout
    f.close()
    
    print('Confusion matrix, without normalization')
    print(cm)
    
    plt.figure()
    plot_confusion_matrix(cm,target_names,title=classifierType+ ": Confusion matrix")

    return precision,accuracy,recall

def plot_bar(model_list,result_mat,title,param_list,loc,ylabel,max_ylim=[]):

    color_list=['r','g','b']

    
    N = len(param_list)
    

    legd=()
    label_tup=()
    
   
    ind_list = np.arange(N)  # the x locations for the groups
    width = 0.15       # the width of the bars

    fig, ax = plt.subplots()

    ind=0

    print result_mat
    
    for i in range(len(model_list)):
        print result_mat.ndim
        if (result_mat.ndim)==1:
            rect = ax.bar(ind_list+(i+1)*width, result_mat[i], width, color=color_list[i])
        elif (result_mat.ndim)==2:
            rect = ax.bar(ind_list+(i+1)*width, result_mat[:,i], width, color=color_list[i])
   
  
        legd=legd+(rect,)

    if max_ylim!=[]:
        ax.set_ylim(max_ylim)
        
    ax.set_ylabel(ylabel)

    ax.set_title(title)
    ax.set_xticks(ind_list+(i+1)*width)
    ax.set_xticklabels(param_list)

    ax.legend(legd, model_list,loc=loc)



# -- Main script for emotion classifer results-- #       
def main():    
    
    #Set parameters for the experiment
    subSize=2000
    train_size=0.8
    ngramType=2

    s='dummy_graph'#'GIS100'
    
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

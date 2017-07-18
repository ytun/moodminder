
"""
Util functions that are not used in app
Author: Yamin Tun

"""

"""
def print_results(yPred_test,y_test,target_names,model_name):

    print (y_test)
    print (yPred_test)


    precision=precision_score(y_test, yPred_test, average='binary')
    accuracy=accuracy_score(y_test, yPred_test)
    recall=recall_score(y_test, yPred_test)

    print("Precision: ", precision)
    print("Accuracy: ",accuracy)
    print("Recall: ",recall)

    print("Classification Report: \n", classification_report(y_test, yPred_test, target_names=target_names))
    print("Confusion Mat: \n", confusion_matrix( y_test.tolist(), yPred_test.tolist(), labels=range(len(target_names))))

    title_model=model_name

    # Compute confusion matrix
    cm = confusion_matrix(y_test, yPred_test)
    np.set_printoptions(precision=2)
    print('Confusion matrix, without normalization')
    print(cm)
    plt.figure()
    plot_confusion_matrix(cm,target_names,title=title_model+ ": Confusion matrix")

    return precision,accuracy,recall
"""
"""
def autolabel(rects,ax):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

"""

"""
def experiment(feat_train,y_train,feat_test,y_test, best_est_model,target_names,model_name):
#    clf_train=SVC(C=1000.0, gamma=0.0001, class_weight='auto' ).fit(feat_train,y_train)

#    clf_train=SVC(C=best_estimator_list[i].C, gamma=best_estimator_list[i].gamma, class_weight='auto' )
    clf_train=best_est_model.fit(feat_train,y_train)
    yPred_test = clf_train.predict(feat_test)

    
    precision,accuracy,recall=print_results(yPred_test,y_test,target_names,model_name)

    return precision,accuracy,recall
"""
 


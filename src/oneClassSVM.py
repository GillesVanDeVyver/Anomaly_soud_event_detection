import os
import numpy as np
from sklearn.svm import OneClassSVM
import ast_based_novelty
import pandas as pd
import common









#combine individual tensors stored in .pt to one pandas dataframe and save as pkl file => not enough RAM
skip_count=0
embedding_base_directory="../dev_data_embeddings/"
for machine in os.listdir(embedding_base_directory):
    if os.path.isdir(embedding_base_directory+"/"+machine):
        print(machine)
        nu=0.05
        source_test_FPR_TPRs=[]
        target_test_FPR_TPRs=[]
        while(nu<=1):
            print("nu="+str(nu))
            one_class_SVM_model = OneClassSVM(nu=nu)

            machine_dir=embedding_base_directory + machine

            train_pickle_location=machine_dir+"/train/"+"dataframe.pkl"
            lables_train,X= ast_based_novelty.generate_lables_and_pd_dataframe(machine_dir + "/train/")
            X=pd.read_pickle(train_pickle_location)

            one_class_SVM_model.fit(X)
            train_prediction=one_class_SVM_model.predict(X)

            train_prediction=one_class_SVM_model.predict(X)
            predicted_anomaly=(train_prediction==-1).sum()
            true_pos, true_neg, false_pos, false_neg = common.generate_confusion_matrix(train_prediction,lables_train)
            print("TRAIN")
            print("True pos: " + str(true_pos) +", false pos: "+str(false_pos))
            print("False neg: " + str(false_neg) +", True neg: "+str(true_neg))



            labels_source_test,X_source_test=ast_based_novelty.generate_lables_and_pd_dataframe(machine_dir+"/source_test")
            source_prediction=one_class_SVM_model.predict(X_source_test)
            true_pos, true_neg, false_pos, false_neg = common.generate_confusion_matrix(source_prediction,labels_source_test)
            print("SOURCE TEST")
            print("True pos: " + str(true_pos) +", false pos: "+str(false_pos))
            print("False neg: " + str(false_neg) +", True neg: "+str(true_neg))
            FPR=false_pos/(false_pos+true_neg)
            TPR=true_pos/(true_pos+false_neg)
            source_test_FPR_TPRs.append((FPR,TPR))


            labels_target_test,X_target_test=ast_based_novelty.generate_lables_and_pd_dataframe(machine_dir+"/target_test")
            target_prediction=one_class_SVM_model.predict(X_target_test)
            true_pos, true_neg, false_pos, false_neg = common.generate_confusion_matrix(target_prediction,labels_target_test)
            print("Target TEST")
            print("True pos: " + str(true_pos) +", false pos: "+str(false_pos))
            print("False neg: " + str(false_neg) +", True neg: "+str(true_neg))
            FPR=false_pos/(false_pos+true_neg)
            TPR=true_pos/(true_pos+false_neg)
            target_test_FPR_TPRs.append((FPR,TPR))



            nu+=0.25
    print(machine + " done")
    print(source_test_FPR_TPRs)
    auc_approx_source_test=common.calculate_pAUC_approximation(source_test_FPR_TPRs,p=1)
    print("AUC_approx_source_test: "+str(auc_approx_source_test))
    pauc_approx_source_test=common.calculate_pAUC_approximation(source_test_FPR_TPRs,p=0.1)
    print("pAUC_approx_source_test: "+str(pauc_approx_source_test))

    print()
    print(target_test_FPR_TPRs)
    auc_approx_target_test=common.calculate_pAUC_approximation(target_test_FPR_TPRs,p=1)
    print("AUC_approx_target_test: "+str(auc_approx_target_test))
    pauc_approx_target_test=common.calculate_pAUC_approximation(target_test_FPR_TPRs,p=0.1)
    print("pAUC_approx_target_test: "+str(pauc_approx_target_test))



    break
        #skip_count+=1










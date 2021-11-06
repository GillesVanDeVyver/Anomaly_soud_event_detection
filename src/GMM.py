import os
import numpy as np
import yaml as yaml
import ast_based_novelty
import pandas as pd
import common
import csv
from sklearn import metrics
from pathlib import Path
from sklearn import mixture


with open("GMM.yaml") as stream:
    param = yaml.safe_load(stream)


result_dir="./results/GMM/nb_comp="+str(param["fit"]["nb_comp"])+"_n_init="+\
                             str(param["fit"]["n_init"])+"/"

Path(result_dir).mkdir(parents=True, exist_ok=True)

f = open(result_dir+"AUC_scores_GMM_nb_comp="+str(param["fit"]["nb_comp"])+"_n_init="+\
                             str(param["fit"]["n_init"])+".txt", 'w')
f_csv = open(result_dir+"AUC_scores_GMM_nb_comp="+str(param["fit"]["nb_comp"])+"_n_init="+\
                             str(param["fit"]["n_init"])+".csv", 'w')

csv_writer = csv.writer(f_csv)
header=["machine","AUC_source","pAUC_source","AUC_target","pAUC_target"]
csv_writer.writerow(header)



embedding_base_directory="../dev_data_embeddings/"
for machine in os.listdir(embedding_base_directory):
    if os.path.isdir(embedding_base_directory+"/"+machine):
        machine_dir = embedding_base_directory + machine
        train_pickle_location = machine_dir + "/train/" + "dataframe.pkl"
        lables_train, X = ast_based_novelty.generate_lables_and_pd_dataframe(
            machine_dir + "/train/",format="autoencoder")
        X = pd.read_pickle(train_pickle_location)

        model = mixture.GaussianMixture(n_components= param["fit"]["nb_comp"],n_init=param["fit"]["n_init"])


        model.fit(X)
        # source prediction
        labels_source_test, X_source_test = ast_based_novelty.generate_lables_and_pd_dataframe(
            machine_dir + "/source_test",format="autoencoder")
        source_prediction = -model.score_samples(X_source_test) # higher value means more normal => - for anomaly score
        auc_source = metrics.roc_auc_score(labels_source_test, source_prediction)
        print(auc_source)
        pauc_source= metrics.roc_auc_score(labels_source_test, source_prediction,max_fpr=param["max_fpr"])
        print(pauc_source)

        fpr_source, tpr_source, thresholds_source = metrics.roc_curve(labels_source_test, source_prediction)

        ROC_location_source=result_dir+"ROC_source_"+str(machine)+"_nb_comp="+str(param["fit"]["nb_comp"])+"_n_init="+\
                             str(param["fit"]["n_init"])+".png"
        common.generate_ROC_curve(fpr_source,tpr_source,ROC_location_source)

        # target prediction
        labels_target_test, X_target_test = ast_based_novelty.generate_lables_and_pd_dataframe(
            machine_dir + "/target_test",format="autoencoder")
        target_prediction = -model.score_samples(X_target_test)
        auc_target= metrics.roc_auc_score(labels_target_test, target_prediction)
        print(auc_target)
        pauc_target= metrics.roc_auc_score(labels_target_test, target_prediction,max_fpr=param["max_fpr"])
        print(pauc_target)

        fpr_target, tpr_target, thresholds_target = metrics.roc_curve(labels_target_test, target_prediction)

        ROC_location_target=result_dir+"ROC_target_"+str(machine)+"_nb_comp="+str(param["fit"]["nb_comp"])+"_n_init="+\
                             str(param["fit"]["n_init"])+".png"
        common.generate_ROC_curve(fpr_target,tpr_target,ROC_location_target)

        f.write(str(machine) + ":\n"+
                "AUC source test="  + str(auc_source) + ", pAUC source test=" + str(pauc_source) + "\n"+
                "AUC target test=" + str(auc_target) + ", pAUC target test=" + str(pauc_target) +  "\n")

        csv_row = [machine, auc_source, pauc_source, auc_target,pauc_target]
        csv_writer.writerow(csv_row)
    print(str(machine)+" done")

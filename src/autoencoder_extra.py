import os
import numpy as np
import yaml as yaml

import ast_based_novelty
import pandas as pd
import common
import csv
import keras_model_ae
from tensorflow import keras
from sklearn import metrics
from pathlib import Path


########################################################################
# visualizer copied from autoencoder baseline
########################################################################
class visualizer(object):
    def __init__(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.fig = self.plt.figure(figsize=(7, 5))
        self.plt.subplots_adjust(wspace=0.3, hspace=0.3)

    def loss_plot(self, loss, val_loss):
        """
        Plot loss curve.

        loss : list [ float ]
            training loss time series.
        val_loss : list [ float ]
            validation loss time series.

        return   : None
        """
        ax = self.fig.add_subplot(1, 1, 1)
        ax.cla()
        ax.plot(loss)
        ax.plot(val_loss)
        ax.set_title("Model loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(["Train", "Validation"], loc="upper right")

    def save_figure(self, name):
        """
        Save figure.

        name : str
            save png file path.

        return : None
        """
        self.plt.savefig(name)
        self.plt.close()


########################################################################
# Autoencoder
########################################################################

with open("autoencoder_extra.yaml") as stream:
    param = yaml.safe_load(stream)


result_dir="./results/autoencoder_extra/lr="+str(param["fit"]["lr"])+"_nbEpochs="+\
                             str(param["fit"]["epochs"])+"/"

Path(result_dir).mkdir(parents=True, exist_ok=True)

f = open(result_dir+"AUC_scores_AE_lr="+str(param["fit"]["lr"])+"_nbEpochs="+\
                             str(param["fit"]["epochs"])+".txt", 'w')
f_csv = open(result_dir+"AUC_scores_AE_lr="+str(param["fit"]["lr"])+"_nbEpochs="+\
                             str(param["fit"]["epochs"])+".csv", 'w')

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

        model = keras_model_ae.get_model(param["ast_embedding_size"],
                                      param["fit"]["lr"])


        model.summary()

        history = model.fit(x=X,
                            y=X,
                            epochs=param["fit"]["epochs"],
                            shuffle=param["fit"]["shuffle"],
                            validation_split=param["fit"]["validation_split"],
                            verbose=param["fit"]["verbose"])
        # source prediction
        labels_source_test, X_source_test = ast_based_novelty.generate_lables_and_pd_dataframe(
            machine_dir + "/source_test",format="autoencoder")
        source_prediction = model.predict(X_source_test)
        y_pred_source = np.mean(np.square(X_source_test - model.predict(source_prediction)),axis=1)
        auc_source = metrics.roc_auc_score(labels_source_test, y_pred_source)
        print(auc_source)
        pauc_source= metrics.roc_auc_score(labels_source_test, y_pred_source,max_fpr=param["max_fpr"])
        print(pauc_source)

        fpr_source, tpr_source, thresholds_source = metrics.roc_curve(labels_source_test, y_pred_source)

        ROC_location_source=result_dir+"ROC_source_"+str(machine)+"_lr="+str(param["fit"]["lr"])+"_nbEpochs="+\
                             str(param["fit"]["epochs"])+".png"
        common.generate_ROC_curve(fpr_source,tpr_source,ROC_location_source)

        # target prediction
        labels_target_test, X_target_test = ast_based_novelty.generate_lables_and_pd_dataframe(
            machine_dir + "/target_test",format="autoencoder")
        target_prediction = model.predict(X_target_test)
        y_pred_target= np.mean(np.square(X_target_test - model.predict(target_prediction)),axis=1)
        auc_target = metrics.roc_auc_score(labels_target_test, y_pred_target)
        print(auc_target)
        pauc_target= metrics.roc_auc_score(labels_target_test, y_pred_target,max_fpr=param["max_fpr"])
        print(pauc_target)

        fpr_target, tpr_target, thresholds_target = metrics.roc_curve(labels_target_test, y_pred_target)

        ROC_location_target=result_dir+"ROC_target_"+str(machine)+"_lr="+str(param["fit"]["lr"])+"_nbEpochs="+\
                             str(param["fit"]["epochs"])+".png"
        common.generate_ROC_curve(fpr_target,tpr_target,ROC_location_target)

        f.write(str(machine) + ":\n"+
                "AUC source test="  + str(auc_source) + ", pAUC source test=" + str(pauc_source) + "\n"+
                "AUC target test=" + str(auc_target) + ", pAUC target test=" + str(pauc_target) +  "\n")

        csv_row = [machine, auc_source, pauc_source, auc_target,pauc_target]
        csv_writer.writerow(csv_row)




        history_img_location=result_dir+"history_"+str(machine)+"_lr="+str(param["fit"]["lr"])+"_nbEpochs="+\
                             str(param["fit"]["epochs"])+".png"
        new_visualizer = visualizer()

        new_visualizer.loss_plot(history.history["loss"], history.history["val_loss"])
        new_visualizer.save_figure(history_img_location)
    print(str(machine)+" done")

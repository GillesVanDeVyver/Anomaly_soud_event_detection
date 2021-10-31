import os
import numpy as np
from sklearn.svm import OneClassSVM
import ast_based_novelty
import pandas as pd








one_class_SVM_model = OneClassSVM()

#combine individual tensors stored in .pt to one pandas dataframe and save as pkl file => not enough RAM
skip_count=0
embedding_base_directory="../dev_data_embeddings/"
for machine in os.listdir(embedding_base_directory):
    if os.path.isdir(embedding_base_directory+"/"+machine):
        print(machine)
        if skip_count>=0:
            machine_dir=embedding_base_directory + machine
            train_directory = machine_dir + "/" + "train"
            pickle_location=machine_dir+"/"+"dataframe.pkl"
            X=pd.read_pickle(pickle_location)
            print(X.shape)
            #one_class_SVM_model.fit(X)


            #print(machine + " done")
        skip_count+=1








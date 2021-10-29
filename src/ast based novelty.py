import os, sys
parentdir = str(os.path.abspath(os.path.join(__file__, "../../../"))) + '/src'
print(parentdir)
sys.path.append(parentdir)
import models
import torch
import preprocessing
import pandas as  pd


class adast(): # anomaly detection ast
    def __init__(self,input_tdim = 1024,num_mel_bins=128,embedding_dimension=768):
        # audioset input sequence length is 1024
        pretrained_mdl_path = '../../pretrained_models/audioset_10_10_0.4593.pth'
        # get the frequency and time stride of the pretrained model from its name
        fstride, tstride = int(pretrained_mdl_path.split('/')[-1].split('_')[1]), int(pretrained_mdl_path.split('/')[-1].split('_')[2].split('.')[0])
        # The input of audioset pretrained model is 1024 frames.
        self.input_tdim = input_tdim
        # initialize an AST model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #device = torch.device("cuda")
        torch.cuda.empty_cache()
        print("cache cleared")
        sd = torch.load(pretrained_mdl_path, map_location=device)
        audio_model_ast = models.ASTModel(input_tdim=input_tdim, fstride=fstride, tstride=tstride)
        self.audio_model = torch.nn.DataParallel(audio_model_ast)
        self.audio_model.load_state_dict(sd, strict=False)
        self.num_mel_bins=num_mel_bins
        self.embedding_dimension=embedding_dimension


    def get_ast_embedding_single_file(self,file_location):
        log_mel = preprocessing.convert_to_log_mel(file_location,num_mel_bins=self.num_mel_bins,target_length=self.input_tdim)
        input = torch.unsqueeze(log_mel, dim=0)
        #input = torch.rand([1, self.input_tdim, 128])
        output = self.audio_model(input)
        return output


    def generate_and_save_embeddings(self,input_directory,output_directory):
        #tensors=None
        for filename in os.listdir(input_directory):
            if filename.endswith(".wav"):
                file_location=os.path.join(input_directory, filename)
                output = self.get_ast_embedding_single_file(file_location)
                sample_name=os.path.splitext(file_location[len(input_directory):])[0]
                print(sample_name)
                output_location=output_directory+sample_name+".pt"
                torch.save(output,output_location)

                """
                if tensors==None:
                    tensors=output
                else:
                    tensors=torch.cat((tensors,output))
                """


# generate intermediate tensors and store as .pt files

adast_mdl = adast()
base_directory="../../dev_data/"
output_base_directory="../dev_data_embeddings/"
for machine in os.listdir(base_directory):
    for domain in os.listdir(base_directory+"/"+machine):
        input_directory=base_directory+machine+"/"+domain
        output_directory=output_base_directory+domain[len(base_directory):]
        adast_mdl.generate_and_save_embeddings(input_directory, output_directory)
    print(machine+" "+domain+" done")



#loaded_sample = torch.load("../dev_data_embeddings/section_00_source_test_anomaly_0093.pt")


""" 
# load resulting vectors in pandas dataframe
embedding_base_directory="../dev_data_embeddings/"
for machine in os.listdir(embedding_base_directory):
    if os.path.isdir(embedding_base_directory+"/"+machine):
        for domain in os.listdir(embedding_base_directory+"/"+machine):
            input_directory = embedding_base_directory + machine + "/" + domain
            tensors_in_domain=None
            for filename in os.listdir(input_directory):
                if filename.endswith(".pt"):
                    file_location=input_directory+"/"+filename
                    loaded_tensor=torch.load(file_location)
                    if tensors_in_domain==None:
                        tensors_in_domain=loaded_tensor
                    else:
                        tensors_in_domain = torch.cat((tensors_in_domain,loaded_tensor))
                    px = pd.DataFrame(tensors_in_domain)
                    print(tensors_in_domain.size())
                    print(px.shape)
                    #print(tensors_in_domain)
        print(machine+" "+domain+" done")
"""





















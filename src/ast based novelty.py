import os, sys
parentdir = str(os.path.abspath(os.path.join(__file__, "../../../"))) + '/src'
print(parentdir)
sys.path.append(parentdir)
import models
import torch
import preprocessing

class ast_model():
    def __init__(self,input_tdim = 1024,num_mel_bins=128,embedding_dimension=768):
        # audioset input sequence length is 1024
        pretrained_mdl_path = '../../pretrained_models/audioset_10_10_0.4593.pth'
        # get the frequency and time stride of the pretrained model from its name
        fstride, tstride = int(pretrained_mdl_path.split('/')[-1].split('_')[1]), int(pretrained_mdl_path.split('/')[-1].split('_')[2].split('.')[0])
        # The input of audioset pretrained model is 1024 frames.
        self.input_tdim = input_tdim
        # initialize an AST model
        device = torch.device("cpu")
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

ast_mdl = ast_model()

directory = "../../dev_data/fan/source_test/"
tensors=None
for filename in os.listdir(directory):
    if filename.endswith(".wav"):
        file_location=os.path.join(directory, filename)
        output = ast_mdl.get_ast_embedding_single_file(file_location)
        if tensors==None:
            tensors=output
        else:
            tensors=torch.cat((tensors,output))
        print(tensors.size())

#output=ast_mdl.get_ast_embedding_single_file("../../dev_data/fan/source_test/section_00_source_test_anomaly_0000.wav")
print(output.shape)
import torch
import torch.nn as nn
import torch.nn.parallel
from miscc.config import cfg
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import models
import torch.utils.model_zoo as model_zoo

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
from torchvision import models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from miscc.config import cfg


#######################################################################################################
# DO NOT CHANGE 
class RNN_ENCODER(nn.Module):
#######################################################################################################
    def __init__(self, ntoken, ninput=256, drop_prob=0.5, nhidden=128, nlayers=1, bidirectional=True):
        super(RNN_ENCODER, self).__init__()
        self.n_steps = cfg.TEXT.WORDS_NUM
        self.ntoken = ntoken        # size of the dictionary
        self.ninput = ninput        # size of each embedding vector
        self.drop_prob = drop_prob  # probability of an element to be zeroed
        self.nlayers = nlayers      # Number of recurrent layers
        self.bidirectional = bidirectional
        self.rnn_type = cfg.RNN.TYPE
        
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
            
        # number of features in the hidden state
        self.nhidden = nhidden // self.num_directions

        self.define_module()


    def define_module(self):
        '''
        e.g., nn.Embedding, nn.Dropout, nn.LSTM
        '''

    def forward(self, captions, cap_lens, hidden, mask=None):
        '''
        1. caption -> embedding
        2. pack_padded_sequence (embedding, cap_lens)
        3. for rnn, hidden is used
        4. sentence embedding should be returned
        '''
        
        return sent_emb

#######################################################################################################
# DO NOT CHANGE 
class CNN_ENCODER(nn.Module):
#######################################################################################################
    def __init__(self, nef):
        super(CNN_ENCODER, self).__init__()
        '''
        nef: size of image feature
        '''

        self.define_module()

    def define_module(self):
        '''
        '''
        
    def forward(self, x):
        '''
        '''
        return x

#######################################################################################################
# DO NOT CHANGE  
class GENERATOR(nn.Module):
#######################################################################################################
    def __init__(self):
        super(GENERATOR, self).__init__()
        '''
        '''
        
    def forward(self, z_code, sent_emb):
        """
        z_code: batch x cfg.GAN.Z_DIM
        sent_emb: batch x cfg.TEXT.EMBEDDING_DIM
        imgs / img_emb: batch x ...
        return: generated image
        """
        return fake_imgs


#######################################################################################################
# DO NOT CHANGE 
class DISCRIMINATOR(nn.Module):
#######################################################################################################
    def __init__(self, b_jcu=True):
        super(DISCRIMINATOR, self).__init__()
        '''
        '''

    def forward(self, x_var):
        '''
        '''
        return x_var

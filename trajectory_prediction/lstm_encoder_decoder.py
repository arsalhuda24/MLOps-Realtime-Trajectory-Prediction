import matplotlib.pyplot as plt
import numpy as np
import random
import os, errno
import sys
from tqdm import trange
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch
import time

os.environ["CUDA_VISIBLE_DEVICES"]="0"

class lstm_encoder(nn.Module):
    ''' Encodes time-series sequence '''

    def __init__(self, input_size, hidden_size,emb_dim,num_layers = 1):
        
        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''
        
        super(lstm_encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.emb_dim = emb_dim

        # define LSTM layer
        
        
        self.embedding = nn.Linear(input_size, emb_dim)
        self.lstm = nn.LSTM(input_size = emb_dim, hidden_size = hidden_size,
                             num_layers = num_layers)
        
#         self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
#                             num_layers = num_layers)

    def forward(self, x_input):
        
        '''
        : param x_input:               input of shape (seq_len, # in batch, input_size)
        : return lstm_out, hidden:     lstm_out gives all the hidden states in the sequence;
        :                              hidden gives the hidden state and cell state for the last
        :                              element in the sequence 
        '''
        
        x_input = self.embedding(x_input)
#         print("CHECK INPUT SHAPE", x_input.shape)
        
#         lstm_out, self.hidden = self.lstm(x_input.view(x_input.shape[0], x_input.shape[1], self.input_size))
        lstm_out, self.hidden = self.lstm(x_input.view(x_input.shape[0], x_input.shape[1], self.emb_dim))

        
        return lstm_out, self.hidden     
    
    def init_hidden(self, batch_size):
        
        '''
        initialize hidden state
        : param batch_size:    x_input.shape[1]
        : return:              zeroed hidden state and cell state 
        '''
        
        eturn (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))


class lstm_decoder(nn.Module):
    ''' Decodes hidden state output by encoder '''
    
    def __init__(self, input_size, hidden_size, num_layers = 1):

        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''
        
        super(lstm_decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                            num_layers = num_layers)
        self.linear = nn.Linear(hidden_size, input_size)           

    def forward(self, x_input, encoder_hidden_states):
        
        '''        
        : param x_input:                    should be 2D (batch_size, input_size)
        : param encoder_hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence 
 
        '''
        
        lstm_out, self.hidden = self.lstm(x_input.unsqueeze(0), encoder_hidden_states)
        output = self.linear(lstm_out.squeeze(0))     
        
        return output, self.hidden

class lstm_seq2seq(nn.Module):
    ''' train LSTM encoder-decoder and make predictions '''
    
#     def __init__(self, input_size, emb_dim ,hidden_size):
    def __init__(self, input_size,hidden_size, emb_dim):


        '''
        : param input_size:     the number of expected features in the input X
        : param hidden_size:    the number of features in the hidden state h
        '''

        super(lstm_seq2seq, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.emb_dim = emb_dim
        self.encoder = lstm_encoder(input_size = input_size, hidden_size = hidden_size, emb_dim = emb_dim)
        self.decoder = lstm_decoder(input_size = input_size, hidden_size = hidden_size)
           
    def forward(self, x_input, target_len):
        batch_size = x_input.shape[1]
        outputs = torch.zeros(target_len, batch_size, x_input.shape[2])        
        encoder_output, encoder_hidden = self.encoder(x_input)
        decoder_input = x_input[-1,:,:]
        decoder_hidden = encoder_hidden
        for t in range(target_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[t] = decoder_output
            decoder_input = decoder_output           
        return outputs

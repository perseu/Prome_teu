# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 16:08:09 2024

@author: JMSA
"""

# The usual imports...
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# This lib downloads the data from Yahoo Finance.
import yfinance as yf

# The PyTorch libs
import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# Sklearn libs to normalize the data and present information on the quality of the model.
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Global variables
ticker = ['GLD', 'SLV', 'CORN', 'WEAT']
period = '10y'
interval = '1d'
stockdata = {}
ndays = 60


# The Dataset class.
class Data(Dataset):
    # The initialization 
    def __init__(self, df, target, ndays):
        
        feat = []
        target = []
        
        for ii in range(0,len(df)-ndays):
            feat.append(df.iloc[ii:ii + ndays].values)
            target.append(df['GLD Close'].iloc[ii + ndays])
            
        self.x = torch.from_numpy(np.array(feat)).type(torch.FloatTensor)
        self.y = torch.from_numpy(np.array(target)).type(torch.FloatTensor)
        
    
    # get size of the sample
    def __len__(self):
        return len(self.y)
    
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
            


# The recurrent neural network class.
class LSTMmodel(nn.Module):
    # The neural network creation. Let it be light! The a RNN Spawns from nothing.
    def __init__(self, input_size, num_hidden, size_hidden, output_size):
        super(LSTMmodel, self).__init__()  
        
        # Storing some parameter values in the class.
        self.input_size = input_size
        self.num_hidden = num_hidden
        self.size_hidden = size_hidden
        self.output_size = output_size
        
        # The LSTM layer
        self.lstm = nn.LSTM(self.input_size, self.num_hidden, self.size_hidden, batch_first=True)

        # The last layer which is a fully connected layer.
        self.fc = nn.Linear(self.size_hidden, self.output_size)
        
    
    def forward(self, x):
        # Initialization of the LSTM parameters
        h0 = torch.zeros(self.num_hidden, x.size(0), self.size_hidden)
        c0 = torch.zeros(self.num_hidden, x.size(0), self.size_hidden)
        
        # Forward propagation of the LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Propagation through the fully connected layer, which connects to the output
        out = self.fc(out[:,-1,:])
        
        return out
    


if __name__ == '__main__':
    
    print('Wellcome to Prometeu. \n\nThis is an exercise to practice with the creation and training of RNNs.\nLets start by downloading the data.\n')
    
    # Downloading the data to a dictonary of DataFrames
    for tic in ticker:
        df = yf.download(tickers=tic, period=period,interval=interval, multi_level_index=False)
        stockdata[tic]=df
    
    # Verifying the existance of NaN in the Dataframes. For this version to work there can be no NaN.
    for tic in ticker:
        if stockdata[tic].isna().sum().sum() != 0:
            print('Missing values!!!')
            print(stockdata[tic].isna().sum())
            print('\n\nExiting script!!!')
            sys.exit()
            
    print('\n\nAll Dataframes have their values, proceeding to the next phase!')
    
    # Merging data.
    dfnames = []
    df_cols = []
    
    for tic in stockdata.keys():
        for coln in stockdata[tic]:
            df_cols.append(tic+' '+coln)
        stockdata[tic].columns = df_cols
        df_cols = []
        dfnames.append(stockdata[tic])  

    df = pd.concat(dfnames, axis=1, join='outer')
    
    
    # Rescalling data.
    scaler = MinMaxScaler()
    scaler.fit(df)
    df_normalized = pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)
    
    
    # Creating the Dataset that will feed the model for training and validation.
    data = Data(df_normalized, 'GLD Close', ndays)
    
    # Splitting the Dataset into a training, and validation datasets.
    train_length = int(0.8 * len(data))
    valid_length = len(data) - train_length
    
    trainDataset , validDataset = random_split(data, [train_length, valid_length])
    
    trainLoader = DataLoader(trainDataset, batch_size=32, shuffle=True)
    validLoader = DataLoader(validDataset, batch_size=32, shuffle=False)
    
    
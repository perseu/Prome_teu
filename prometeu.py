# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 16:08:09 2024

@author: JMSA
"""

# The usual imports...
import time
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

################################################################################
# Classes                                                                      #
################################################################################

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
    def __init__(self, input_size, num_hidden, size_hidden, output_size):
        super(LSTMmodel, self).__init__()

        self.input_size = input_size
        self.num_hidden = num_hidden
        self.size_hidden = size_hidden
        self.output_size = output_size

        # The LSTM layer
        self.lstm = nn.LSTM(self.input_size, self.size_hidden, self.num_hidden, batch_first=True)

        # The fully connected layer
        self.fc = nn.Linear(self.size_hidden, self.output_size)
        
    def forward(self, x):
        # Initialize hidden and cell states (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.num_hidden, x.size(0), self.size_hidden).to(x.device)  
        c0 = torch.zeros(self.num_hidden, x.size(0), self.size_hidden).to(x.device)  

        # Forward pass through the LSTM
        out, (h_n, c_n) = self.lstm(x, (h0, c0))

        # Use the last hidden state from the last time step
        out = out[:, -1, :]  

        # Feed the output into the fully connected layer
        out = self.fc(out)

        return out
    

################################################################################
# Functions                                                                    #
################################################################################

# Training and validation of the model
def train_validate(model, trainloader, validloader, criterion, optimizer, epochs, device='cpu'):
    # setting up the device that is going to process the model
    model.to(device)
    
    # Registration of loss
    trainLoss = []
    validLoss = []
    lossAVGEpochTrain = []
    lossSTDEpochTrain = []
    lossAVGEpochValid = []
    lossSTDEpochValid = []
    
    
    start_time = time.time()
    for epoch in range(epochs):
        for x, y in trainloader:
            x, y = x.float(), y.float()
            model.train()
            optimizer.zero_grad()
            yhat = model(x)
            loss = criterion(yhat, y.view(-1, 1))
            loss.backward()
            optimizer.step()
            trainLoss.append(loss.item())
        
        lossAVGEpochTrain.append(np.array(trainLoss).mean())
        lossSTDEpochTrain.append(np.array(trainLoss).std())
        
        with torch.no_grad():
            for x, y in validloader:
                x, y = x.float(), y.float()
                model.eval()
                yhat = model(x)
                loss = criterion(yhat, y.view(-1, 1))
                validLoss.append(loss.item())
                
        lossAVGEpochValid.append(np.array(validLoss).mean())
        lossSTDEpochValid.append(np.array(validLoss).std())
        
        if epoch%10 == 0:                          # This cycle prints out the epoch number every 10 epochs.
            epoch_time = time.time()
            epoch_time = (epoch_time - start_time)/60             # Converts from seconds to minutes.
            print(f'Passing epoch {epoch:.0f} after {epoch_time:.2f} minutes.')
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTraining time: {total_time:.2f} minutes")    
    print('\n\nThe training as ended... You\'re now a Jedi!!!\n')
    
    TrainValidStatistics = pd.DataFrame({'Train Loss Average':lossAVGEpochTrain,
                                         'Train Loss STD':lossSTDEpochTrain,
                                         'Validation Loss Average':lossAVGEpochValid,
                                         'Validation Loss STD':lossSTDEpochValid})
    
    return TrainValidStatistics


################################################################################
# The biginning of everything... Let there be the "Main"!!!                    #
################################################################################

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
    
    # Creating the model.
    n_hidden = 3
    hidden_size = 64
    out_size = 1
    
    lstmmodel = LSTMmodel(df.shape[1], n_hidden, hidden_size, out_size)
    
    # Training and validation parameters, criterion, optimizer, and number of epochs.
    lr = 0.001
    epochs = 100
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(lstmmodel.parameters(), lr=lr)
    
    # training and validation
    TrainValidStatistics = train_validate(lstmmodel, trainLoader, validLoader, criterion, optimizer, epochs)
    
    # Plotting the evolution of the Loss throughout the training and validation epochs
    x = range(len(TrainValidStatistics))
    plt.figure(figsize=(14,10))
    sns.lineplot(data=TrainValidStatistics, x=x, y='Train Loss Average', label='Train Loss Average')
    plt.fill_between(x, TrainValidStatistics['Train Loss Average'] - TrainValidStatistics['Train Loss STD'], TrainValidStatistics['Train Loss Average'] + TrainValidStatistics['Train Loss STD'], alpha=0.3) #, label='Train Loss STD')
    sns.lineplot(data=TrainValidStatistics, x=x, y='Validation Loss Average', label='Validation Loss Average')
    plt.fill_between(x, TrainValidStatistics['Validation Loss Average'] - TrainValidStatistics['Validation Loss STD'], TrainValidStatistics['Validation Loss Average'] + TrainValidStatistics['Validation Loss STD'], alpha=0.3) #, label='Validation Loss STD')    
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.show()

    
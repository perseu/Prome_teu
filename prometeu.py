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


# The Dataset class.
class Data(Dataset):
    # The initialization 
    def __init__(self, df, target):
        pass


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
    
    
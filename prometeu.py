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

import yfinance as yf

# Global variables
ticker = ['GLD', 'SLV', 'CORN', 'WEAT']
period = '10y'
interval = '1d'
stockdata = {}




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
    print('All Dataframes have their values, proceeding to the next phase!')
    
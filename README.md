This is another excercise that I did. This script takes stock information from Yahoo Finance from the stocks of Gold, Silver, Corn and Wheat, and tries to estimate the Value of Gold at the Close. For this it uses 60 days, and trys to perdict day 61.
As this is just an exercise, it is not a production script, it will download the required data from yahoo, creates a Dataset, trains and validates the RNN, and at the end it predicts the end it makes a prediction for tomorrow taking into account the last 60 days, counting from today.

It uses: time, sys, os, numpy, pandas, matplotlib, seaborn, yfinance, pytorch, sklearn

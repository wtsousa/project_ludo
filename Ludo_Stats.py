# -*- coding: utf-8 -*-
"""
Ludo Simulator for Statistical Analysis based on Logistic Regression
DATA ANALYSIS SCRIPT
Created on Sat Dec 12 23:34:30 2020
@author: William Sousa - williamsousa@gmail.com
"""

###############
## LIBRARIES ##
###############

import glob
from Ludo_Functions import *
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols, glm
import seaborn as sns
import matplotlib.pyplot as plt

########################
## READING INPUT DATA ##
########################

#Initializing Data Frame
LUDO_GAMES_DF = pd.DataFrame()

#Input Data Directory - Other users need to change this line of code
DF_DIR = 'C:\\Users\\Will\\Documents\\Data Science\\Projeto Ludo' 

#Reads all existing files that were created by the Ludo.py script
for filename in glob.glob(DF_DIR + '\\*_CompleteSample.csv'):
    print(filename)
    SAMPLE_DF = pd.read_csv(filename, index_col=False)
    SAMPLE_DF = SAMPLE_DF.drop('Unnamed: 0',axis=1)
    #Appending data to the Data Frame
    LUDO_GAMES_DF = LUDO_GAMES_DF.append(SAMPLE_DF,ignore_index=True)

######################
##  DATA WRANGLING  ##
######################

#Dropping "Game" columns because it contains no useful information
LUDO_GAMES_DF = LUDO_GAMES_DF.drop('Game',axis=1)

#Check for duplicated rows that could affect modelling
if sum(LUDO_GAMES_DF.duplicated()) > 0: print("Warning: Duplicated Values")

#Save consolidated data on a single csv file
LUDO_GAMES_DF.to_csv(DF_DIR + '\\Ludo_Games_Samples.csv')

####################################
##  PREPARING DATA FOR MODELLING  ##
####################################

#The original DataFrame differentiates the player order. However, for modelling
#purpose, it is a better idea to "explode" the data frame to reflect one row
#per player.

#Transforms original data (one game per row) in to new data frame (one player per row)
LUDO_RESULTS_DF = ludo_explode(LUDO_GAMES_DF)
LUDO_RESULTS_DF.to_csv(DF_DIR + '\\Ludo_Players_Samples.csv')

#Optional code to read existing Players Samples csv file
#filename = DF_DIR + '\\Ludo_Players_Samples.csv'
#LUDO_RESULTS_DF = pd.read_csv(filename, index_col=False).drop('Unnamed: 0',axis=1)

#Adding the PlayerID to the Data Frame
LUDO_RESULTS_DF['PlayerID'] = np.mod(LUDO_RESULTS_DF.index,4)

###############
## MODELLING ##
###############


#MODELLING - IS THE 'RANDOM' STRATEGY REALLY RANDOM?

# Define model formula
formula = 'Winner ~ C(PlayerID)' #PlayerID as the categorical explanatory variable
family_GLM = sm.families.Binomial() #Preparing for Logistic Regression
model_GLM = glm(formula = formula, data = LUDO_RESULTS_DF, family = family_GLM).fit()
print(model_GLM.summary())
#High p-values mean that PlayerID has no statistical significance 
#to explain if a player will win

#Plotting a Bar Chart displaying the number of victories per player
is_winner = (LUDO_RESULTS_DF['Winner']==1)
plot_data = LUDO_RESULTS_DF['PlayerID'][is_winner].value_counts()
plt.bar(plot_data.index,plot_data)
plt.ylabel('Vitórias')
plt.xlabel('Jogador')
plt.xticks(np.arange(4), ('A', 'B', 'C','D'))
plt.axhline(linewidth=1, color='r',y=8100) 
#8100 is the mean number of victories per player considering 32400 game samples
plt.ylim(8000, 8300)
plt.show()

#MODELLING - DOES IT MATTER IF YOU ARE THE FIRST PLAYER?

#Updating the previous Bar Chart to include the number of victories the first player had
is_first = (LUDO_RESULTS_DF['FirstP']==1)
new_value = LUDO_RESULTS_DF['Winner'][is_first].value_counts()[1]
plot_data = plot_data.append(pd.Series(new_value,index=["P"]))
plt.bar(plot_data.index,plot_data,color=['blue', 'blue', 'blue', 'blue', 'yellow'])
plt.ylabel('Vitórias')
plt.xlabel('Jogador')
plt.xticks(np.arange(5), ('A', 'B', 'C','D', 'Primeiro'))
plt.axhline(linewidth=1, color='r',y=8100)
#plt.ylim(8000, 8300)
plt.show()

# Define model formula
formula = 'Winner ~ FirstP'
family_GLM = sm.families.Binomial()
model_GLM = glm(formula = formula, data = LUDO_RESULTS_DF, family = family_GLM).fit()
print(model_GLM.summary())

# Extract model coefficients
print('Model coefficients: \n', model_GLM.params) 
#Positive FirstP coefficient with statistical significance (p-value < 0.05) means that
#being the first player increases the odds of winning

# Extract coefficients
intercept, slope = model_GLM.params

# Compute the multiplicative effect on the odds (logit function)
odds = np.exp(model_GLM.params)
print('Odds: \n', odds)
odds = odds - 1
print('Probabilities: \n', odds/(1+odds))

# Estimated covariance matrix: model_cov
model_cov = model_GLM.cov_params()
print(model_cov)

# Compute standard error (SE): std_error
std_error = np.sqrt(model_cov.loc['FirstP', 'FirstP'])
print('SE: ', round(std_error, 4))

# Compute Wald statistic (z-statistic)
wald_stat = slope/std_error
print('Wald statistic: ', round(wald_stat,4))
#This indicator is also available in the .summary() method data

# Compute confidence intervals for the odds
odds = np.exp(model_GLM.conf_int())
print('Odds: \n', odds)
odds = odds - 1
print('Probabilities: \n', odds/(1+odds))


#MODELLING - DOES THE STRATEGY MATTER?

# Define model formula
formula = "Winner ~ C(Strategy, Treatment(reference='RANDOM'))" 
#The formula considers the 'RANDOM' strategy as reference to compare coefficients
family_GLM = sm.families.Binomial()
model_GLM = glm(formula = formula, data = LUDO_RESULTS_DF, family = family_GLM).fit()
print(model_GLM.summary())

# Compute confidence intervals for the odds
print(np.exp(model_GLM.conf_int()))

# Compute the multiplicative effect on the odds
odds = np.exp(model_GLM.params)
print('Odds: \n', odds)
#Besides the "Always Activate" strategy, all others increase the odds of winning in
#comparison to the reference 'Random' strategy
odds = odds - 1
print('Probabilities: \n', odds/(1+odds))

# Compute confidence intervals for the odds
odds = np.exp(model_GLM.conf_int())
print('Odds: \n', odds)
odds = odds - 1
print('Probabilities: \n', odds/(1+odds))

#Visualizing the number of victories per player strategy
plot_data = LUDO_RESULTS_DF['Strategy'][is_winner].value_counts()
plt.bar(plot_data.index,plot_data,color=['green', 'blue', 'orange', 'yellow', 'purple', 'black'])
plt.ylabel('Vitórias')
plt.xlabel('Estratégia')
plt.xticks(np.arange(6), ('Fugir', 'Andar', 'Caçar','Atacar', 'Aleatória', 'Ativar'))
plt.axhline(linewidth=1, color='r',y=5400) 
#5400 is the mean of victories considering a sample of 32400 games
plt.show()
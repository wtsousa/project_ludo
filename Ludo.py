# -*- coding: utf-8 -*-
"""
Ludo Simulator for Statistical Analysis based on Logistic Regression
SAMPLES CREATION SCRIPT
Created on Sat Dec 12 23:34:30 2020
@author: William Sousa - williamsousa@gmail.com
"""
###############
## LIBRARIES ##
###############

#Ludo_Functions.py contains a set of functions designed to make the following
#code clearer
from Ludo_Functions import *
import numpy as np

############################
## INITIALIZING VARIABLES ##
############################

#STRATEGY_SAMPLES indicate how many sets of ludo games will be processed
STRATEGY_SAMPLES = 1

#LIMIT_RUNS limits the 'while' loop which processes each ludo game
LIMIT_RUNS = 1296*STRATEGY_SAMPLES 
#1296 is the total number of different strategies combination

NUM_OF_PLAYERS = 4 #The script does not support a different number of players

#Initializing the LUDO_STATS Data Frame
LUDO_STATS = init_stats() 

#Simple code that creates a list of all possible strategy combinations
STRATEGIES = ['RANDOM','ALWAYS_MOVE','ALWAYS_ACTIVATE','PRIORITIZE_ATTACK','PRIORITIZE_HUNT','PRIORITIZE_ESCAPE']
possible_strategies = []
for strat1 in STRATEGIES:
    for strat2 in STRATEGIES:
        for strat3 in STRATEGIES:
            for strat4 in STRATEGIES:
                possible_strategies.append([strat1,strat2,strat3,strat4])

#Initializing game counter
game_run = 1

###############
## MAIN CODE ##
###############

#'while' loop runs games until LIMIT_RUNS is reached
while (game_run <= LIMIT_RUNS):
    
    #Initializing arrays of counters
    total_rolls = np.zeros(NUM_OF_PLAYERS) #Number of Dice Rolls per Player (pp)
    total_attacks = np.zeros(NUM_OF_PLAYERS) #Number of Attacks per Player
    total_pass = np.zeros(NUM_OF_PLAYERS) #Number of turns without actions per Player
    total_moves = np.zeros(NUM_OF_PLAYERS) #Number of movements per Player
    total_activations = np.zeros(NUM_OF_PLAYERS)  #Number of activated tokens per Player
    
    #Randomly selecting the First Player of the game
    first_playerID = get_playerIndex(who_is_first_player(NUM_OF_PLAYERS))
    
    #Current Player indicator
    current_playerID = first_playerID
    
    #Initializing the Ludo Board Data Frame
    ludo_board = reset_board(NUM_OF_PLAYERS)

    #Variable that indicates when the game is won
    GAME_WON = False
    
    #Additional counter
    roll_times = 0 #Number of dice rolled
    round_number = 0 #Round number
    rolled_six = 0 #Number of times '6' is rolled in a player's turn
    
    #Assigning a strategy to the current player
    players_strategy = possible_strategies[np.mod(game_run-1,1296)]
    
    #Loop for each player's actions    
    while not(GAME_WON):
        
        #Random roll of a 6-sides die
        die_result = die_roll()
        
        #Retrieve all possible actions for the current player
        possible_actions = get_possible_actions(current_playerID,die_result,ludo_board)
        
        #Retrieve the current player strategy
        strategy = players_strategy[current_playerID]
        
        #Assign an action based on the player's strategy
        chosen_action = choose_action_ai(current_playerID,ludo_board,possible_actions, strategy)
        
        #Execute the player's action
        next_playerID,ludo_board,attacked_token = execute_action(current_playerID,ludo_board,chosen_action,die_result)
    
        #Assess the result of the player's action
        turn_summary,GAME_WON = assess_result(current_playerID,ludo_board,die_result)
        
        #Check if game was won
        if GAME_WON: winner_playerID = current_playerID
        
        #Check if the current player rolled a '6' (which means he/she can continue to play)
        #If current_playerID is the same as the next_playerID, it means the player had rolled a '6'
        if not(current_playerID == next_playerID): 
            round_number = round_number + 1 #Incrementing the Round counter
        else: #current player rolled a '6'
            #Limiting the number of times (3) the current player can continue rolling dice
            if (rolled_six==3): 
                #Limit has been reached
                #Assigning another player for the next turn
                next_playerID = get_next_playerID(current_playerID,NUM_OF_PLAYERS)
                #Resetting the counter to control how many times a player has rolled a '6' in the same turn
                rolled_six = 0
            else:
                #Incrementing the counter to control the number of times a player rolls a '6' in the same turn
                rolled_six = rolled_six + 1
                
        #Updating arrays of counters
        total_rolls[current_playerID] = total_rolls[current_playerID] + 1
        total_attacks[current_playerID] = total_attacks[current_playerID] + attacked_token
        if (chosen_action == 'PASS'): total_pass[current_playerID] = total_pass[current_playerID] + 1
        if (chosen_action == 'MOVE_TOKEN'): total_moves[current_playerID] = total_moves[current_playerID] + 1
        if (chosen_action == 'ACTIVATE_TOKEN'): total_activations[current_playerID] = total_activations[current_playerID] + 1
        current_playerID = next_playerID
        roll_times = roll_times + 1

    #Printing game information on the console
    #print (ludo_board[ludo_board==62].count()) #Final Score
        
    #Calculate Game Statistics
    GAME_STATS = fill_stats(game_run,first_playerID,winner_playerID,ludo_board,total_rolls,round_number,
                            players_strategy,total_attacks,total_pass,total_moves,total_activations)
    
    #Update Ludo Statistics Data Frame
    LUDO_STATS = LUDO_STATS.append(GAME_STATS,ignore_index=True)
    
    #Incrementing game counter
    game_run = game_run + 1
    
    #'while' loop for game ends

############################
## SAVING SIMULATION DATA ##
############################

# Users need to change the following directory before running this code
DF_DIR = 'C:\\Users\\Will\\Documents\\Data Science\\Projeto Ludo'    
LUDO_STATS.to_csv(DF_DIR + '\\Ludo_Games_CompleteSample.csv')

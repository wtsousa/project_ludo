# -*- coding: utf-8 -*-
"""
Ludo Simulator for Statistical Analysis based on Logistic Regression
SUPPORTING FUNCTIONS
Created on Sat Dec 12 23:34:30 2020
@author: William Sousa - williamsousa@gmail.com
"""
import random
import numpy as np
import pandas as pd

def die_roll(n_sides = 6, n_dice = 1):
    return np.random.randint(1,n_sides+1,size=n_dice)

def who_is_first_player(NUM_OF_PLAYERS, die_sides = 6):
    dice_roll = die_roll(6,4)
    dice_result = (dice_roll == max(dice_roll))
    dice_draw = sum(dice_result)
    if (dice_draw != 1):
        dice_result = who_is_first_player(NUM_OF_PLAYERS, die_sides)
    return dice_result

def get_playerIndex(first_player):
    return np.where(first_player==True)[0][0]

def get_next_playerID(current_playerID,NUM_OF_PLAYERS):
    return np.mod(current_playerID+1,NUM_OF_PLAYERS)

def reset_board(NUM_OF_PLAYERS):
    ludo_board = pd.DataFrame()
    for col_index in range(0,NUM_OF_PLAYERS):
        new_column = np.zeros(4)-1
        col_name = "player" + str(col_index+1)
        ludo_board[col_name] = new_column
        ludo_board.index = ["token1","token2","token3","token4"]
    return ludo_board

def reset_history(NUM_OF_PLAYERS):
    ludo_history = pd.DataFrame()
    ludo_history['turn'] = pd.Series(dtype='object')
    ludo_history['player'] = pd.Series(dtype='object')
    ludo_history['roll'] = pd.Series(dtype='object')
    ludo_history['board'] = str(reset_board(NUM_OF_PLAYERS).to_dict())
    return ludo_history
    
def get_possible_actions(playerID,die_result,ludo_board):
    
    possibleActions = []
    #Check if tokens can be activated
    inactive_tokens = sum(ludo_board.iloc[:,playerID]==-1)
    tokens_in_goal = sum(ludo_board.iloc[:,playerID]==62)
    active_tokens = 4 - inactive_tokens - tokens_in_goal
    if ((inactive_tokens > 0) and (die_result[0] == 6)):
        possibleActions.append("ACTIVATE_TOKEN")
        
    if (active_tokens > 0):
        possibleActions.append("MOVE_TOKEN")
        
    #Check if this is the first turn
    if ((active_tokens == 0) and (die_result[0] != 6)):
        possibleActions.append("PASS")
        
    return possibleActions

def get_initialPosition(playerID):
    INITIAL_POSITIONS = [1,15,29,43]
    initialPosition = INITIAL_POSITIONS[playerID]
    return initialPosition
            
def choose_action(current_playerID,ludo_board,possible_actions):
    #random choice
    chosen_index = np.random.randint(0,len(possible_actions))
    chosen_action = possible_actions[chosen_index]
    return chosen_action

def choose_action_ai(current_playerID,ludo_board,possible_actions,player_strategy):
    if (player_strategy == 'RANDOM'):
        chosen_index = np.random.randint(0,len(possible_actions))
        chosen_action = possible_actions[chosen_index]
        
    if (player_strategy == 'ALWAYS_ACTIVATE'):
        if ("ACTIVATE_TOKEN" in possible_actions):
            chosen_action = "ACTIVATE_TOKEN"
        else: #Random Choice
            chosen_index = np.random.randint(0,len(possible_actions))
            chosen_action = possible_actions[chosen_index]
            
    if (player_strategy == 'ALWAYS_MOVE'):
        if ("MOVE_TOKEN" in possible_actions):
            chosen_action = "MOVE_TOKEN"
        else: #Random Choice
            chosen_index = np.random.randint(0,len(possible_actions))
            chosen_action = possible_actions[chosen_index]
            
    if (player_strategy == 'PRIORITIZE_ATTACK'):
        if ("MOVE_TOKEN" in possible_actions):
            #Check if enemy can be attacked
            enemy_player,enemy_token,enemy_distance,player_token = nearest_target(current_playerID,ludo_board)
            if (enemy_distance<=6):
                #print("ATTACK!")
                chosen_action = "MOVE_TOKEN"
            else: #Random Choice
                chosen_index = np.random.randint(0,len(possible_actions))
                chosen_action = possible_actions[chosen_index]
        else: #Random Choice
            chosen_index = np.random.randint(0,len(possible_actions))
            chosen_action = possible_actions[chosen_index]
            
    if (player_strategy == 'PRIORITIZE_HUNT'):
        if ("MOVE_TOKEN" in possible_actions):
            #Check if enemy can be attacked
            enemy_player,enemy_token,enemy_distance,player_token = nearest_target(current_playerID,ludo_board)
            if (enemy_distance<=13):
                #print("HUNT!")
                chosen_action = "MOVE_TOKEN"
            else: #Random Choice
                chosen_index = np.random.randint(0,len(possible_actions))
                chosen_action = possible_actions[chosen_index]
        else: #Random Choice
            chosen_index = np.random.randint(0,len(possible_actions))
            chosen_action = possible_actions[chosen_index]

    if (player_strategy == 'PRIORITIZE_ESCAPE'):
        if ("MOVE_TOKEN" in possible_actions):
            #Check if enemy can be attacked
            enemy_player,enemy_token,enemy_distance,player_token = nearest_foe(current_playerID,ludo_board)
            if (enemy_distance>=(56-13)):
                #print("ESCAPE!")
                chosen_action = "MOVE_TOKEN"
            else: #Random Choice
                chosen_index = np.random.randint(0,len(possible_actions))
                chosen_action = possible_actions[chosen_index]
        else: #Random Choice
            chosen_index = np.random.randint(0,len(possible_actions))
            chosen_action = possible_actions[chosen_index]  
            
    return chosen_action

def nearest_target(current_playerID,ludo_board):
    
    ingame_tokens = (ludo_board >= 0) & (ludo_board<57) #True for active tokens that are not yet in home column
    enemy_distance = 62
    player_token = -1
    enemy_player = -1
    enemy_token = -1
    transformed_board = transform_board(ludo_board) #Enables position comparison
    for playerTokenIndex in range(0,4):
        player_position = transformed_board.iloc[playerTokenIndex,current_playerID]
        playerTokenIsInGame = ingame_tokens.iloc[playerTokenIndex,current_playerID]
        if playerTokenIsInGame:
            for enemyIndex in range(0,4):
                if (enemyIndex != current_playerID):
                    for enemyTokenIndex in range(0,4):
                        enemyTokenIsInGame = ingame_tokens.iloc[enemyTokenIndex,enemyIndex]
                        if enemyTokenIsInGame:
                            enemy_position = transformed_board.iloc[enemyTokenIndex,enemyIndex]
                            distance = enemy_position - player_position
                            if (distance < 0): 
                                distance = distance + 56
                            enemy_distance = min(distance,enemy_distance)
                            if (distance == enemy_distance):
                                enemy_token = enemyTokenIndex
                                enemy_player = enemyIndex
                                player_token = playerTokenIndex
                                
    restored_board = restore_board(transformed_board) #Restore board index

    return enemy_player,enemy_token,enemy_distance,player_token

def nearest_foe(current_playerID,ludo_board):
    
    ingame_tokens = (ludo_board >= 0) & (ludo_board<57) #True for active tokens that are not yet in home column
    enemy_distance = 0
    player_token = -1
    enemy_player = -1
    enemy_token = -1
    transformed_board = transform_board(ludo_board) #Enables position comparison
    for playerTokenIndex in range(0,4):
        player_position = transformed_board.iloc[playerTokenIndex,current_playerID]
        playerTokenIsInGame = ingame_tokens.iloc[playerTokenIndex,current_playerID]
        if playerTokenIsInGame:
            for enemyIndex in range(0,4):
                if (enemyIndex != current_playerID):
                    for enemyTokenIndex in range(0,4):
                        enemyTokenIsInGame = ingame_tokens.iloc[enemyTokenIndex,enemyIndex]
                        if enemyTokenIsInGame:
                            enemy_position = transformed_board.iloc[enemyTokenIndex,enemyIndex]
                            distance = enemy_position - player_position
                            if (distance < 0): 
                                distance = distance + 56
                            enemy_distance = max(distance,enemy_distance)
                            if (distance == enemy_distance):
                                enemy_token = enemyTokenIndex
                                enemy_player = enemyIndex
                                player_token = playerTokenIndex
                                
    restored_board = restore_board(transformed_board) #Restore board index

    return enemy_player,enemy_token,enemy_distance,player_token

def choose_token(playerID,ludo_board,chosen_action):
    if (chosen_action == "ACTIVATE_TOKEN"):
        tokens = ludo_board.iloc[:,playerID]
        inactive_tokens = np.where(tokens==-1)[0]
        
        #random choice
        chosen_index = np.random.randint(0,len(inactive_tokens))
        chosen_token = inactive_tokens[chosen_index]
        
    if (chosen_action == "MOVE_TOKEN"):
        tokens = ludo_board.iloc[:,playerID]
        active_tokens = np.where((tokens>0)&(tokens<62))[0]
        
        #random choice
        chosen_index = np.random.randint(0,len(active_tokens))
        chosen_token = active_tokens[chosen_index]
        
    return chosen_token

def choose_token_ai(playerID,ludo_board,chosen_action,player_strategy="RANDOM"):
    if (chosen_action == "ACTIVATE_TOKEN"):
        tokens = ludo_board.iloc[:,playerID]
        inactive_tokens = np.where(tokens==-1)[0]
        
        #random choice
        chosen_index = np.random.randint(0,len(inactive_tokens))
        chosen_token = inactive_tokens[chosen_index]
        
    if (chosen_action == "MOVE_TOKEN"):
        tokens = ludo_board.iloc[:,playerID]
        active_tokens = np.where((tokens>0)&(tokens<62))[0] #returns the indexes of active tokens as a list
        
        if (player_strategy in ['RANDOM','ALWAYS_MOVE','ALWAYS_ACTIVATE']):
            chosen_index = np.random.randint(0,len(active_tokens))
            chosen_token = active_tokens[chosen_index]

        if (player_strategy in ['PRIORITIZE_ATTACK','PRIORITIZE_HUNT']):
            enemy_player,enemy_token,enemy_distance,player_token = nearest_target(playerID,ludo_board)
            chosen_token = player_token

        if (player_strategy in ['PRIORITIZE_ESCAPE']):
            enemy_player,enemy_token,enemy_distance,player_token = nearest_foe(current_playerID,ludo_board)
            chosen_token = player_token        
    return chosen_token

def execute_action(current_playerID,ludo_board,chosen_action,die_result):
    NUM_OF_PLAYERS = len(ludo_board.columns)
    attacked_enemies = 0
    next_playerID = current_playerID
    if (chosen_action == "PASS"):
        next_playerID = get_next_playerID(current_playerID,NUM_OF_PLAYERS)
        
    if (chosen_action == "ACTIVATE_TOKEN"):
        next_playerID = current_playerID
        chosen_token = choose_token_ai(current_playerID,ludo_board,chosen_action)
        #print(chosen_token)
        ludo_board.iloc[chosen_token,current_playerID] = 1
        
    if (chosen_action == "MOVE_TOKEN"):
        if (die_result[0] == 6):
            next_playerID = current_playerID
        else:
            next_playerID = get_next_playerID(current_playerID,NUM_OF_PLAYERS)
            
        chosen_token = choose_token_ai(current_playerID,ludo_board,chosen_action)
        current_position = ludo_board.iloc[chosen_token,current_playerID]
        #print(chosen_token)
        new_position = current_position + die_result[0]
        
        #Check if the player bounced in the home column
        if (new_position > 62):
            ludo_board.iloc[chosen_token,current_playerID] = 62 - (new_position - 62)
        else:
            #if new_position == 62:
                #print("GOOOOOOAAAAALLLLLL!!!")
            ludo_board.iloc[chosen_token,current_playerID] = new_position
            
            if new_position < 57:
                ingame_tokens = (ludo_board >= 0) & (ludo_board<57) #True for active tokens that are not yet in home column
                unprotected_tokens = (ludo_board != 1) | (ludo_board != 15) | (ludo_board != 29) | (ludo_board != 43) #True for tokens that are not in one of the 4 initial spaces
                transformed_board = transform_board(ludo_board) #Enables position comparison
                transformed_position =  np.mod(new_position+(14*current_playerID)-1,56)+1 #Transformed player position
                attacked_tokens = (transformed_board==transformed_position) #True for all tokens in position
                attacked_tokens.iloc[:,current_playerID] = False #Disregard current player tokens
                restored_board = restore_board(transformed_board) #Restore board index
                restored_board = restored_board.mask(attacked_tokens & unprotected_tokens & ingame_tokens,-1) #Deactivate enemy token
                if ((attacked_tokens & unprotected_tokens & ingame_tokens).any().any() == True): attacked_enemies=1
                ludo_board = restored_board
            #else:
                #print("Just got home")
        
        #print("player{0} moved token{1} from {2} to {3}".format(current_playerID+1,chosen_token+1,current_position,new_position))
        
    return next_playerID,ludo_board,attacked_enemies

def transform_board(ludo_board):
    transformed_board = ludo_board
    transformed_board.iloc[:,1] = np.where(ludo_board.iloc[:,1]<0,-1,
                                           np.where(ludo_board.iloc[:,1]>56,ludo_board.iloc[:,1],
                                                    np.mod(ludo_board.iloc[:,1]+14-1,56)+1))
    transformed_board.iloc[:,2] = np.where(ludo_board.iloc[:,2]<0,-1,
                                           np.where(ludo_board.iloc[:,2]>56,ludo_board.iloc[:,2],
                                                    np.mod(ludo_board.iloc[:,2]+28-1,56)+1))
    transformed_board.iloc[:,3] = np.where(ludo_board.iloc[:,3]<0,-1,
                                           np.where(ludo_board.iloc[:,3]>56,ludo_board.iloc[:,3],
                                                    np.mod(ludo_board.iloc[:,3]+42-1,56)+1))
    
    return transformed_board

def restore_board(ludo_board):
    restored_board = ludo_board
    restored_board.iloc[:,1] = np.where(ludo_board.iloc[:,1]<0,-1,
                                        np.where(ludo_board.iloc[:,1]>56,ludo_board.iloc[:,1],
                                                 np.mod(ludo_board.iloc[:,1]-14-1,56)+1))
    restored_board.iloc[:,2] = np.where(ludo_board.iloc[:,2]<0,-1,
                                        np.where(ludo_board.iloc[:,2]>56,ludo_board.iloc[:,2],
                                                 np.mod(ludo_board.iloc[:,2]-28-1,56)+1))
    restored_board.iloc[:,3] = np.where(ludo_board.iloc[:,3]<0,-1,
                                        np.where(ludo_board.iloc[:,3]>56,ludo_board.iloc[:,3],
                                                 np.mod(ludo_board.iloc[:,3]-42-1,56)+1))
    
    return restored_board

def assess_result(current_playerID,ludo_board,die_result):
    
    turn_summary = "Player " + str(current_playerID + 1) + " rolled a " + str(die_result)
    
    #Check if player has bounced in home column
    if sum(ludo_board.iloc[:,current_playerID]>62) > 0:
        turn_summary = "Error: Position higher than 62"
        
    #Check if player has entered home column
    home_tokens = sum(ludo_board.iloc[:,current_playerID]>56)
    if home_tokens > 0:
        turn_summary = "Player " + str(current_playerID + 1) + " has " + str(home_tokens) + " tokens in home column."
        
    #Check if player has won
    win_tokens = sum(ludo_board.iloc[:,current_playerID]==62)
    if win_tokens > 0:
        turn_summary = "Player " + str(current_playerID + 1) + " has " + str(home_tokens) + " tokens in goal."
        
    if win_tokens == 4:
        GAME_WON = True
    else:
        GAME_WON = False
        
    return turn_summary, GAME_WON

def init_stats():
    LUDO_STATS = pd.DataFrame()
    LUDO_STATS['Game'] = pd.Series(dtype='object')
    LUDO_STATS['FirstP'] = pd.Series(dtype='object')
    LUDO_STATS['Winner'] = pd.Series(dtype='object')
    LUDO_STATS['Score1'] = pd.Series(dtype='object') 
    LUDO_STATS['Score2'] = pd.Series(dtype='object') 
    LUDO_STATS['Score3'] = pd.Series(dtype='object') 
    LUDO_STATS['Score4'] = pd.Series(dtype='object') 
    LUDO_STATS['Rolls1'] = pd.Series(dtype='object')
    LUDO_STATS['Rolls2'] = pd.Series(dtype='object')
    LUDO_STATS['Rolls3'] = pd.Series(dtype='object')
    LUDO_STATS['Rolls4'] = pd.Series(dtype='object')
    LUDO_STATS['Rounds'] = pd.Series(dtype='object')
    LUDO_STATS['Strat1'] = pd.Series(dtype='object')
    LUDO_STATS['Strat2'] = pd.Series(dtype='object')
    LUDO_STATS['Strat3'] = pd.Series(dtype='object')
    LUDO_STATS['Strat4'] = pd.Series(dtype='object')
    LUDO_STATS['Attacks1'] = pd.Series(dtype='object')
    LUDO_STATS['Attacks2'] = pd.Series(dtype='object')
    LUDO_STATS['Attacks3'] = pd.Series(dtype='object')
    LUDO_STATS['Attacks4'] = pd.Series(dtype='object')
    LUDO_STATS['Passes1'] = pd.Series(dtype='object')
    LUDO_STATS['Passes2'] = pd.Series(dtype='object')
    LUDO_STATS['Passes3'] = pd.Series(dtype='object')
    LUDO_STATS['Passes4'] = pd.Series(dtype='object')
    LUDO_STATS['Moves1'] = pd.Series(dtype='object')
    LUDO_STATS['Moves2'] = pd.Series(dtype='object')
    LUDO_STATS['Moves3'] = pd.Series(dtype='object')
    LUDO_STATS['Moves4'] = pd.Series(dtype='object')
    LUDO_STATS['Tokens1'] = pd.Series(dtype='object')
    LUDO_STATS['Tokens2'] = pd.Series(dtype='object')
    LUDO_STATS['Tokens3'] = pd.Series(dtype='object')
    LUDO_STATS['Tokens4'] = pd.Series(dtype='object')
    return LUDO_STATS

def fill_stats(game_run,first_playerID,winner_playerID,ludo_board,total_rolls,round_number,
               players_strategy,total_attacks,total_pass,total_moves,total_activations):
    LUDO_STATS = init_stats()
    LUDO_STATS['Game'] = pd.Series(game_run)
    LUDO_STATS['FirstP'] = pd.Series(first_playerID)
    LUDO_STATS['Winner'] = pd.Series(winner_playerID)
    final_score = ludo_board[ludo_board==62].count()
    LUDO_STATS['Score1'] = pd.Series(final_score['player1'])
    LUDO_STATS['Score2'] = pd.Series(final_score['player2'])
    LUDO_STATS['Score3'] = pd.Series(final_score['player3'])
    LUDO_STATS['Score4'] = pd.Series(final_score['player4'])
    LUDO_STATS['Rolls1'] = pd.Series(total_rolls[0])
    LUDO_STATS['Rolls2'] = pd.Series(total_rolls[1])
    LUDO_STATS['Rolls3'] = pd.Series(total_rolls[2])
    LUDO_STATS['Rolls4'] = pd.Series(total_rolls[3])
    LUDO_STATS['Rounds'] = pd.Series(round_number)
    LUDO_STATS['Strat1'] = pd.Series(players_strategy[0])
    LUDO_STATS['Strat2'] = pd.Series(players_strategy[1])
    LUDO_STATS['Strat3'] = pd.Series(players_strategy[2])
    LUDO_STATS['Strat4'] = pd.Series(players_strategy[3])
    LUDO_STATS['Attacks1'] = pd.Series(total_attacks[0])
    LUDO_STATS['Attacks2'] = pd.Series(total_attacks[1])
    LUDO_STATS['Attacks3'] = pd.Series(total_attacks[2])
    LUDO_STATS['Attacks4'] = pd.Series(total_attacks[3])
    LUDO_STATS['Passes1'] = pd.Series(total_pass[0])
    LUDO_STATS['Passes2'] = pd.Series(total_pass[1])
    LUDO_STATS['Passes3'] = pd.Series(total_pass[2])
    LUDO_STATS['Passes4'] = pd.Series(total_pass[3])
    LUDO_STATS['Moves1'] = pd.Series(total_moves[0])
    LUDO_STATS['Moves2'] = pd.Series(total_moves[1])
    LUDO_STATS['Moves3'] = pd.Series(total_moves[2])
    LUDO_STATS['Moves4'] = pd.Series(total_moves[3])
    LUDO_STATS['Tokens1'] = pd.Series(total_activations[0])
    LUDO_STATS['Tokens2'] = pd.Series(total_activations[1])
    LUDO_STATS['Tokens3'] = pd.Series(total_activations[2])
    LUDO_STATS['Tokens4'] = pd.Series(total_activations[3])
    #print(LUDO_STATS)
    return LUDO_STATS

def ludo_explode(LUDO_DF):
    LUDO_RESULTS = pd.DataFrame()
    countdown = len(LUDO_DF)
    
    for each_key,ludo_game in LUDO_DF.iterrows():

        for playerID  in range(0,4):
            PLAYER_DATA = pd.DataFrame()
            if (ludo_game['Winner'] == playerID): 
                PLAYER_DATA['Winner'] = pd.Series(1)
            else:
                PLAYER_DATA['Winner'] = pd.Series(0)
            if (ludo_game['FirstP'] == playerID): 
                PLAYER_DATA['FirstP'] = pd.Series(1)
            else:
                PLAYER_DATA['FirstP'] = pd.Series(0)
                
            PLAYER_DATA['Score'] = pd.Series(ludo_game['Score'+str(playerID+1)])
            PLAYER_DATA['Rolls'] = pd.Series(ludo_game['Rolls'+str(playerID+1)])
            PLAYER_DATA['Strategy'] = pd.Series(ludo_game['Strat'+str(playerID+1)])
            PLAYER_DATA['Attacks'] = pd.Series(ludo_game['Attacks'+str(playerID+1)])
            PLAYER_DATA['Passes'] = pd.Series(ludo_game['Passes'+str(playerID+1)])
            PLAYER_DATA['Moves'] = pd.Series(ludo_game['Moves'+str(playerID+1)])
            PLAYER_DATA['Tokens'] = pd.Series(ludo_game['Tokens'+str(playerID+1)])
            
            PLAYER_DATA['GameID'] = pd.Series(each_key)
            
            LUDO_RESULTS = LUDO_RESULTS.append(PLAYER_DATA,ignore_index=True)
            #print(PLAYER_DATA)
        #print (str(countdown-each_key))
            
    return LUDO_RESULTS


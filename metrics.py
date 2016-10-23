import data_module
import numpy as np

#calculate the "accuracy" metric of some predictions given lists of the predictions and the true values
def accuracy(predicted_values, true_values):

    #TODO rounding
    #create numpy arrays of true values and rounded predicted values (>= .5 --> 1, 0 otherwise)
    predicted_values = np.array(predicted_values).round()
    true_values = np.array(true_values)

    #calculate and return accuracy = number of correctly predicted values / number of values
    return (predicted_values == true_values).sum() / len(predicted_values)


#calculate the "LogLoss" metric of some predictions given lists of the predictions and the true values
def log_loss(predicted_values, true_values):
    
    #calculate numerator of LogLoss
    ll_n = sum(true_values * np.log(predicted_values) + 
               np.subtract(1, true_values) * np.log(np.subtract(1, predicted_values)))
    
    #calculate and return LogLoss
    return (-ll_n) / len(true_values)


#score a given bracket using ESPN's tournament challenge scoring system
def bracket_score(predicted_bracket, true_bracket):
    
    score = 0

    #iterate over tournament games, ordered by round w/ play in games at the end
    for game_index, game in enumerate(sorted(true_bracket)):

        #correct prediction
        if predicted_bracket[game] in true_bracket[game]:
    
            #give score depending on the round
            if (0 <= game_index <= 31): score += 10         #10 - round 2
            elif (32 <= game_index <= 47): score += 20      #20 - round 3
            elif (48 <= game_index <= 55): score += 40      #40 - round 4
            elif (56 <= game_index <= 59): score += 80      #80 - round 5
            elif (60 <= game_index <= 61): score += 160     #160 - round 6
            elif game_index == 62: score += 320             #320 - championship
    
    #return bracket score
    return score


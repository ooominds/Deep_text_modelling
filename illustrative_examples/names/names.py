

### Predicting gender from names

### Table of Contents

# I. Preliminary steps
# II. Prepare the data
# III. Feed-forward neural network model
# IV. Long short-term memory model
# V. Naive discriminative learning model

#### I. Preliminary steps 

### Import necessary libraries and set up the working directory

### Import necessary packages
import os
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dropout, Dense
from keras.optimizers import Adam, Nadam, RMSprop, SGD
from keras.activations import relu, elu
from keras.losses import binary_crossentropy
from keras import metrics
from keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import warnings

### Set working directory
TOP = '/media/adnane/HDD drive/Adnane/PostDoc_ooominds/Programming/Deep_text_modelling_package_repo/'
#TOP = 'F:/Adnane/PostDoc_ooominds/Programming/Deep_text_modelling_package_repo/'
#TOP = '/media/Deep_text_modelling_package_repo/'
WD = TOP + 'package'
os.chdir(WD)

### Import local packages
import deep_text_modelling.preprocessing as pr
import deep_text_modelling.modelling as md
import deep_text_modelling.evaluation as ev

# Display option for dataframes and matplotlib
pd.set_option('display.max_colwidth', 100) # Max width of columns when dispalying datasets
PREVIOUS_MAX_ROWS = pd.options.display.max_rows
pd.options.display.max_rows = 20
warnings.filterwarnings('ignore') # Hide warnings
warnings.simplefilter('ignore')

### Define file paths
NAMES_FULL_CSV = TOP + "illustrative_examples/names/Data/Names_full.csv"
NAMES_TRAIN_CSV = TOP + "illustrative_examples/names/Data/Names_train.csv"
NAMES_VALID_CSV = TOP + "illustrative_examples/names/Data/Names_valid.csv"
NAMES_TEST_CSV = TOP + "illustrative_examples/names/Data/Names_test.csv"
CUE_INDEX = TOP + "illustrative_examples/names/Data/Cue_index.csv"
OUTCOME_INDEX = TOP + "illustrative_examples/names/Data/Outcome_index.csv"
TEMP_DIR = TOP + "illustrative_examples/names/Data/"

### Parameters to use
N_outcomes = 2  # number of (most frequent) outcomes to keep (here all possible outcomes)
N_cues = 26  # number of (most frequent) cues to keep (here all alphabet letters)
prop_valid = 1/10 # proportion of validation data
prop_test = 1/10 # proportion of test data

### II. Prepare the data

### Load the data
names_full = pd.read_csv(NAMES_FULL_CSV)
print(f'Number of examples: {len(names_full)}')
names_full.head(5)

### Prepare the cues and outcomes
# Create the unigraph cues
names_full['cues'] = names_full['first_name'].apply(lambda s: pr.orthoCoding(s, gram_size = 1))

# Rename the column gender as 'outcomes'
names_full.rename(columns={"gender": "outcomes"}, inplace = True)
names_full = names_full[['cues', 'outcomes']]
names_full.head(5)


### Create index systems for the cues and outcomes
# Create the files containing the index systems
pr.create_index_systems_from_df(data = names_full, 
                                cue_index_path = CUE_INDEX, 
                                outcome_index_path = OUTCOME_INDEX)
# Import the cue index system
cue_to_index = pr.import_index_system(CUE_INDEX, N_tokens = N_cues)
pr.display_dictionary(cue_to_index, start = 0, end = 5)
# Order dictionary alphabetically
cue_to_index = {k:(i+1) for i,k in enumerate(sorted(cue_to_index.keys()))}
pr.display_dictionary(cue_to_index, start = 0, end = 5)

# Import the outcome index system
outcome_to_index = pr.import_index_system(OUTCOME_INDEX)
outcome_to_index
# Reverse the cue dictionary
index_to_cue = pr.reverse_dictionary(cue_to_index)
# Reverse the outcome dictionary
index_to_outcome = pr.reverse_dictionary(outcome_to_index)
index_to_outcome

### Split into training, validation and test sets
# Create train, valid and test set files
pr.df_train_valid_test_split(data = names_full, 
                             train_data_path = NAMES_TRAIN_CSV, 
                             valid_data_path = NAMES_VALID_CSV, 
                             test_data_path = NAMES_TEST_CSV, 
                             p_valid = prop_valid, 
                             p_test = prop_test)

# Load the train, valid and test sets
names_train = pd.read_csv(NAMES_TRAIN_CSV, sep=',', na_filter = False)
names_valid = pd.read_csv(NAMES_VALID_CSV, sep=',', na_filter = False)
names_test = pd.read_csv(NAMES_VALID_CSV, sep=',', na_filter = False)


### III. Feed-forward neural network model

### Tokenisation
# Extract an event
event4 = names_train.iloc[[4]]
event4
# cues (if max_len not specified, all the cues in the sequences will be considered)
cues4 = names_train.loc[names_train.index[4], 'cues']
cues4_onehot_FNN = md.seq_to_onehot_1darray(cues4, index_system = cue_to_index, N_tokens = N_cues, max_len = None)
cues4_onehot_FNN
# cues (if max_len specified, only 'max_len' first cues in a sequence will be considered)
cues4 = names_train.loc[names_train.index[4], 'cues']
cues4_onehot_FNN = md.seq_to_onehot_1darray(cues4, index_system = cue_to_index, N_tokens = N_cues, max_len = 4)
cues4_onehot_FNN

# outcomes
outcomes4 = names_train.loc[names_train.index[4], 'outcomes']
outcomes4_onehot_FNN = md.seq_to_onehot_1darray(outcomes4, index_system = outcome_to_index, N_tokens = N_outcomes)
outcomes4_onehot_FNN

### Build a simple FNN model
# Build a simple FNN with two hidden layers having 64 units 
### Hyperparameters to use
p = {'max_len': None,
     'epochs': 10, # number of iterations on the full set 
     'batch_size': 16, 
     'embedding_input': None,
     'hidden_layers': 2, # number of hidden layers 
     'hidden_neuron':64, # number of neurons in the input layer 
     'lr': 0.0001, # learning rate       
     'dropout': 0.2, 
     'optimizer': Adam, 
     'losses': binary_crossentropy, 
     'activation': relu, 
     'last_activation': 'sigmoid'}

# Model fitting
FNN_hist, FNN_model = md.train(model = 'FNN',
                               data_train = names_train, 
                               data_valid = names_valid, 
                               cue_index = cue_to_index, 
                               outcome_index = outcome_to_index, 
                               verbose = 2,
                               metrics = ['accuracy'],
                               params = p)

FNN_model.summary()

### Hyperparameters to use
p = {'max_len': None,
     'epochs': 10, # number of iterations on the full set 
     'batch_size': 16, 
     'embedding_input': None,
     'hidden_layers': 2, # number of hidden layers 
     'hidden_neuron':64, # number of neurons in the input layer 
     'lr': 0.0001, # learning rate       
     'dropout': 0.2, 
     'optimizer': Adam, 
     'losses': binary_crossentropy, 
     'activation': relu, 
     'last_activation': 'sigmoid'}
# Model fitting
FNN_hist, FNN_model = md.train(model = 'FNN',
                               data_train = names_train, 
                               data_valid = names_valid, 
                               cue_index = cue_to_index, 
                               outcome_index = outcome_to_index, 
                               verbose = 2,
                               metrics = ['accuracy'],
                               params = p)

### Tune the parameters to find a good model
### Parameter values to use in the grid search 
p = {'max_len': [None, 5, 11],
     'epochs': [1, 5, 10, 20, 30], # number of iterations on the full set (x5)
     'batch_size': [8, 16, 32, 64, 128, 256], # (x6)
     'embedding_input': [None],
     'hidden_layers':[0, 1, 2], # number of hidden layers (x3)
     'hidden_neuron':[16, 32, 64, 128], # number of neurons in the input layer (x4)
     'lr': [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01], # learning rate (x7)       
     'dropout': [0, 0.1, 0.2, 0.3, 0.4], # (x5)
     'optimizer': [Adam, RMSprop], # (x2)
     'losses': [binary_crossentropy], # (x1)
     'activation':[relu, elu], # (x2)
     'last_activation': ['sigmoid'] # (x1)
     }
### Grid search 
TUNING_PATH = TOP + 'illustrative_examples/names/Results/grid_search_FNN_names.csv'
md.grid_search(model = 'FNN',
               data_train = names_train, 
               data_valid = names_valid, 
               cue_index = cue_to_index, 
               outcome_index = outcome_to_index,
               params = p,
               prop_grid = 2e-4, 
               tuning_output_file = TUNING_PATH)

### Assessing the grid search results
# Import the grid search file to analyse the results 
gs_results = pd.read_csv(TUNING_PATH, index_col = False)

# get the number of parameter combinations that were processed
len(gs_results)

# Display the dataframe containing the tuning results
gs_results.head()

gs_results.columns

# get the highest result for any metric
print(f"- Highest validation accuracy: {r.high('val_acc')}")
print(f"- Highest validation f1-score: {r.high('f1score')}")

# get the index of the combination with the best result
i_best = gs_results['val_acc'].argmax()
i_best

# Iteration 7 produced the highest validation accuracy, corresponding to the following parameters:

# get the best paramaters
list(gs_results.iloc[i_best, ])

### Retraining with the best parameters

### Hyperparameters to use
p = {'max_len': None,
     'epochs': 1, # number of iterations on the full set 
     'batch_size': 32, 
     'embedding_input': None,
     'hidden_layers': 0, # number of hidden layers 
     'hidden_neuron':16, # number of neurons in the input layer 
     'lr': 0.0002, # learning rate       
     'dropout': 0.2, 
     'optimizer': RMSprop, 
     'losses': binary_crossentropy, 
     'activation': None, 
     'last_activation': 'sigmoid'}

# Model fitting
FNN_hist, FNN_model = md.train_FNN(data_train = names_train, 
                                  data_valid = names_valid, 
                                  cue_index = cue_to_index, 
                                  outcome_index = outcome_to_index, 
                                  verbose = 2,
                                  metrics = ['accuracy'],
                                  params = p)

# Save the model and training history
MODEL_PATH = TOP + 'illustrative_examples/names/Results/FNN_names.h5'
HISTORY_PATH = TOP + 'illustrative_examples/names/Results/FNN_history_dict_names'
md.export_model(model = FNN_model, path = MODEL_PATH)  # creates a HDF5 file 
md.export_history(history_dict = FNN_hist, path = HISTORY_PATH)
del FNN_model, FNN_hist  # deletes the existing model and history dictionary

# Load the model and training history
FNN_model = md.import_model(MODEL_PATH)
FNN_history_dict = md.import_history(path = HISTORY_PATH)

### Evaluate the final model

# Performance on the last epoch of the training set
print(f"- Training loss in the last epoch: {FNN_history_dict['loss'][-1]}")
print(f"- Training accuracy in the last epoch: {FNN_history_dict['acc'][-1]}")

# Performance on the last epoch of the validation set
print(f"- Validation loss in the last epoch: {FNN_history_dict['val_loss'][-1]}")
print(f"- Validation accuracy in the last epoch: {FNN_history_dict['val_acc'][-1]}")

# Generate plots to assess the performance of the NN
ev.plot_learning_curve(history_dict = FNN_history_dict, metric = 'acc', set = 'train_valid')

# Test prediction for a single given cue sequence. Model expect inout as array of shape (1, N_cues) 
cue1_seq = 'y_o_u_s_s_e_f'
outcome1_prob_pred = ev.predict_proba_oneevent_FNN(model = FNN_model, 
                                                   cue_seq = cue1_seq, 
                                                   num_cues = N_cues,  
                                                   cue_index = cue_to_index)
outcome1_prob_pred

# Evaluate the model on the test set
prob_pred = ev.predict_proba_eventfile_FNN(model = FNN_model, 
                                           data_test = names_test, 
                                           num_cues = N_cues, 
                                           num_outcomes = N_outcomes, 
                                           cue_index = cue_to_index, 
                                           outcome_index = outcome_to_index)

# True responses to compare the predictions to
y_test = names_test.replace({'outcomes': outcome_to_index})['outcomes']
y_pred = np.argmax(prob_pred, axis=1)+1

# Overall test accuracy
test_accuracy = accuracy_score(y_test, y_pred)
test_accuracy

# Test accuracy per class
cmat = confusion_matrix(y_test, y_pred) # Confusion matrix
print(cmat.diagonal()/cmat.sum(axis=1)) 


### IV. Long short-term memory model

### Tokenisation

# Extract an event
event1 = names_train.iloc[[1]]
event1

# cues
cues1 = names_train.loc[names_train.index[1], 'cues']
cues1_onehot_LSTM = md.seq_to_onehot_2darray(cues1, index_system = cue_to_index, N_tokens = N_cues, max_len = max_len)
cues1_onehot_LSTM

# outcomes
outcomes1 = names_train.loc[names_train.index[1], 'outcomes']
outcomes1_onehot_LSTM = md.seq_to_onehot_1darray(outcomes1, index_system = outcome_to_index, N_tokens = N_outcomes)
outcomes1_onehot_LSTM

### Build a simple LSTM model

# Build a simple LSTM that has 64 hidden units 

### Build a simple LSTM that has 64 hidden units 
p = {'epochs': 10, # number of iterations on the full set 
    'batch_size': 16, 
    'hidden_neuron': 64, # number of neurons in the input layer 
    'lr': 0.0001, # learning rate       
    'dropout': 0.2, 
    'optimizer': RMSprop, 
    'losses': binary_crossentropy, 
    'last_activation': 'sigmoid'}

# Model fitting
LSTM_out, LSTM_model = md.train_LSTM(data_train = names_train, 
                                     data_valid = names_valid, 
                                     cue_index = cue_to_index, 
                                     outcome_index = outcome_to_index, 
                                     max_len = max_len,
                                     verbose = 2,
                                     metrics = ['accuracy'],
                                     params = p)

LSTM_model.summary()


#### Tune the parameters to find a good model

### Parameter tuning using grid search 
p = {'epochs': [1, 5, 10, 20, 30], # number of iterations on the full set (x5)
     'batch_size': [8, 16, 32, 64, 128, 256], # (x6)
     'hidden_neuron':[16, 32, 64, 128], # number of neurons in the input layer (x4)
     'lr': [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01], # learning rate (x7)       
     'dropout': [0, 0.1, 0.2, 0.3, 0.4], # (x5)
     'optimizer': [Adam, Nadam, RMSprop, SGD], # (x4)
     'losses': [binary_crossentropy], # (x1)
     'last_activation': ['sigmoid'] # (x1)
     }
# => Total number of combinations: 5*6*4*7*5*4*1*1 = 16800

### Grid search 
TUNING_PATH = TOP + 'illustrative_examples/names/Results/grid_search_LSTM_names.csv'
md.grid_search_LSTM(data_train = names_train, 
                    data_valid = names_valid, 
                    cue_index = cue_to_index, 
                    outcome_index = outcome_to_index,
                    max_len = max_len,
                    params = p,
                    prop_grid = 6e-4, 
                    tuning_output_file = TUNING_PATH)


### Assessing the grid search using talos

# Import the results file to analyse the results with talos
r = ta.Reporting(TUNING_PATH)

# get the highest result for any metric
print(f"- Highest validation accuracy: {r.high('val_acc')}")
print(f"- Highest validation f1-score: {r.high('f1score')}")

# get the index of the combination with the best result
r.rounds2high(metric = 'val_acc')

# get the best paramaters
r.data.iloc[69,]

### Retraining with the best parameters

### Hyperparameters to use
p = {'epochs': 29, # number of iterations on the full set 
    'batch_size': 64, 
    'hidden_neuron': 128, # number of neurons in the input layer 
    'lr': 0.01, # learning rate       
    'dropout': 0.4, 
    'optimizer': RMSprop, 
    'losses': binary_crossentropy, 
    'last_activation': 'sigmoid'}

# Model fitting
LSTM_hist, LSTM_model = md.train_LSTM(data_train = names_train, 
                                     data_valid = names_valid, 
                                     cue_index = cue_to_index, 
                                     outcome_index = outcome_to_index, 
                                     max_len = max_len,
                                     metrics = ['accuracy'],
                                     params = p)

# Save the model and training history
MODEL_PATH = TOP + 'illustrative_examples/names/Results/LSTM_names.h5'
HISTORY_PATH = TOP + 'illustrative_examples/names/Results/LSTM_history_dict_names'
md.export_model(model = LSTM_model, path = MODEL_PATH)  # creates a HDF5 file 
md.export_history(history_dict = LSTM_hist, path = HISTORY_PATH)
del LSTM_model, LSTM_hist  # deletes the existing model and history dictionary

# Load the model and training history
LSTM_model = md.import_model(MODEL_PATH)
LSTM_history_dict = md.import_history(path = HISTORY_PATH)

### Evaluate the final model

# Generate plots to assess the performance of the simple LSTM
ev.plot_learning_curve(history_dict = LSTM_history_dict, metric = 'acc', set = 'train_valid')

# Test prediction for a single given cue sequence. Model expect inout as array of shape (1, N_cues) 
cue1_seq = 'y_o_u_s_s_e_f'
outcome1_prob_pred = ev.predict_proba_oneevent_LSTM(model = LSTM_model, 
                                                   cue_seq = cue1_seq, 
                                                   num_cues = N_cues,  
                                                   cue_index = cue_to_index,
                                                   max_len = max_len)
outcome1_prob_pred

# Evaluate the model on the test set
prob_pred = ev.predict_proba_eventfile_LSTM(model = LSTM_model, 
                                           data_test = names_test, 
                                           num_cues = N_cues, 
                                           num_outcomes = N_outcomes, 
                                           cue_index = cue_to_index, 
                                           outcome_index = outcome_to_index, 
                                           max_len = max_len)

# True responses to compare the predictions to
y_test = names_test.replace({'outcomes': outcome_to_index})['outcomes']
y_pred = np.argmax(prob_pred, axis=1)+1

# Overall test accuracy
test_accuracy = accuracy_score(y_test, y_pred)
test_accuracy

# Test accuracy per class
cmat = confusion_matrix(y_test, y_pred) # Confusion matrix
print(cmat.diagonal()/cmat.sum(axis=1)) 

### V. Naive discriminative learning model <a ID="V"></a> 

### Build a simple NDL model

### Build a simple NDL
p = {'epochs': 10, # number of iterations on the full set 
    'lr': 0.001}

# Model fitting
NDL_history_dict, NDL_model = md.train_NDL(data_train = names_train, 
                                           data_valid = names_valid,  
                                           cue_index = cue_to_index, 
                                           outcome_index = outcome_to_index, 
                                           temp_dir = TEMP_DIR,
                                           num_threads = no_threads, 
                                           params = p)

# Generate learning curve
ev.plot_learning_curve(history_dict = NDL_history_dict, metric = 'acc', set = 'train_valid')

### Tune the parameters to find a good model

### Parameter tuning using grid search 
p = {'lr': [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05], # learning rate (x8)
     'epochs': [1, 2, 4, 6, 8], # number of iterations on the full set (x6)
     }
# => Total number of combinations: 8*5 = 40

### Grid search 
TUNING_PATH = TOP + 'illustrative_examples/names/Results/grid_search_NDL_names.csv'
md.grid_search_NDL(data_train = names_train, 
                   data_valid = names_valid, 
                   cue_index = cue_to_index, 
                   outcome_index = outcome_to_index, 
                   temp_dir = TEMP_DIR,
                   params = p, 
                   prop_grid = 0.1, 
                   tuning_output_file = TUNING_PATH, 
                   num_threads = no_threads)

### Assessing the grid search using talos

# Import the results file to analyse the results with talos
r = ta.Reporting(TUNING_PATH)

# get the highest result for any metric
print(f"- Highest validation accuracy: {r.high('val_acc')}")
print(f"- Highest validation f1-score: {r.high('f1score')}")

# get the index of the combination with the best result
r.rounds2high(metric = 'val_acc')

# get the best paramaters
r.data.iloc[16,]

### Retraining with the best parameters

### Hyperparameters to use
p = {'epochs': 7, # number of iterations on the full set 
    'lr': 0.001}

# Model fitting
NDL_history_dict, NDL_model = md.train_NDL(data_train = names_train, 
                                           data_valid = names_valid,  
                                           cue_index = cue_to_index, 
                                           outcome_index = outcome_to_index, 
                                           temp_dir = TEMP_DIR,
                                           num_threads = no_threads, 
                                           verbose = 1,
                                           params = p)

# Save the weights and training history
MODEL_PATH = TOP + 'illustrative_examples/names/Results/NDL_names.h5'
HISTORY_PATH = TOP + 'illustrative_examples/names/Results/NDL_history_dict_names'
md.export_model(model = NDL_model, path = MODEL_PATH)  # create a HDF5 file 
md.export_history(history_dict = NDL_history_dict, path = HISTORY_PATH)
del NDL_model, NDL_history_dict  # delete the existing model and history dictionary

# Load the model and training history
MODEL_PATH = TOP + 'illustrative_examples/names/Results/NDL_names.h5'
HISTORY_PATH = TOP + 'illustrative_examples/names/Results/NDL_history_dict_names'
NDL_model = md.import_model(MODEL_PATH)
NDL_history_dict = md.import_history(path = HISTORY_PATH)

### Evaluate the final model

# Test prediction for a single given cue sequence. Model expect input as array of shape (1, N_cues) 
cue1_seq = 'y_o_u_s_s_e_f'
outcome1_prob_pred = ev.predict_proba_oneevent_NDL(model = NDL_model, 
                                                   cue_seq = cue1_seq)
outcome1_prob_pred

# Evaluate the model on the test set
prob_pred = ev.predict_proba_eventfile_NDL(model = NDL_model, 
                                           data_test = names_test, 
                                           temp_dir = TEMP_DIR,
                                           num_threads = no_threads)

# True responses to compare the predictions to
y_test = names_test.replace({'outcomes': outcome_to_index})['outcomes']
y_pred = np.argmax(prob_pred, axis=1)+1

# Overall test accuracy
test_accuracy = accuracy_score(y_test, y_pred)
test_accuracy

# Test accuracy per class
cmat = confusion_matrix(y_test, y_pred) # Confusion matrix
print(cmat.diagonal()/cmat.sum(axis=1)) 


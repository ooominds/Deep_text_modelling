####################
# Preliminary steps
####################

### Import necessary packages
import os
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dropout, Dense
from keras.optimizers import Adam, Nadam, RMSprop, SGD
from keras.activations import relu, elu
from keras.losses import binary_crossentropy, categorical_crossentropy
from keras import metrics
from keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

### Set working directory
TOP = '/media/sf_PostDoc_ooominds/Programming/Deep_text_modelling_package_repo/'
WD = '/media/sf_PostDoc_ooominds/Programming/Deep_text_modelling_package_repo/package/'
os.chdir(WD)

### Import local packages
import deep_text_modelling.preprocessing as pr
import deep_text_modelling.modelling as md
import deep_text_modelling.evaluation as ev

### File paths
TENSE_FULL = TOP + "illustrative_examples/tense_1k/Data/Tense1k_full.csv"
TENSE_TRAIN = TOP + "illustrative_examples/tense_1k/Data/Tense1k_train.csv"
TENSE_VALID = TOP + "illustrative_examples/tense_1k/Data/Tense1k_valid.csv"
TENSE_TEST = TOP + "illustrative_examples/tense_1k/Data/Tense1k_test.csv"
CUE_INDEX = TOP + "illustrative_examples/tense_1k/Data/Cue_index.csv"
OUTCOME_INDEX = TOP + "illustrative_examples/tense_1k/Data/Outcome_index.csv"

### Parameters to use
N_outcomes = 7  # number of most frequent outcomes to keep 
N_cues = 1000  # number of most frequent cues to keep
max_len = 20 # consider only the first 'max_len' cues in a sequence when applying LSTM
no_threads = 4 # Number of CPU cores to use
prop_valid = 1/14 # proportion of validation data
prop_test = 1/14 # proportion of test data

################
# Load the data
################

# Load the data
tense_full = pd.read_csv(TENSE_FULL, sep = ',', index_col = 0, na_filter = False)

# Number of examples
len(tense_full) # 7000

# First 5 lines 
tense_full.head(5)
#                                                 cues      outcomes
# 0              they_the_language_of_clothes_fluently      PastProg
# 1  yet_the_environmental_consequences_of_unchecke...  FutureSimple
# 2  william_and_harry_with_charles_and_the_rest_of...  FutureSimple
# 3                businessmen_there_and_back_in_a_day   PresentProg
# 4  pleas_from_the_prince_of_wales_for_corporate_h...   PresentPerf

################################################
# Create index systems for the cues and outcomes
################################################

# Create the files containing the index systems
pr.create_index_systems_from_df(data = tense_full, 
                                cue_index_path = CUE_INDEX, 
                                outcome_index_path = OUTCOME_INDEX)

# Import the cue index system and limit it to N_cues most frequent cues
cue_to_index = pr.import_index_system(CUE_INDEX, N_tokens = N_cues)
# Number of elements in the dictionary
len([k for k,v in cue_to_index.items()]) # 1000
pr.display_dictionary(cue_to_index, start = 0, end = 5)
# {the: 1}
# {of: 2}
# {a: 3}
# {and: 4}
# {in: 5}

# Import the outcome index system
outcome_to_index = pr.import_index_system(OUTCOME_INDEX)
# Number of elements in the dictionary
len([k for k,v in outcome_to_index.items()]) # 7
pr.display_dictionary(outcome_to_index, start = 0, end = 7)
# {PastProg: 1}
# {FutureSimple: 2}
# {PresentProg: 3}
# {PresentPerf: 4}
# {PastPerf: 5}
# {PresentSimple: 6}
# {PastSimple: 7}

# Reverse the cue dictionary
index_to_cue = pr.reverse_dictionary(cue_to_index)
pr.display_dictionary(index_to_cue, start = 0, end = 5)
# {1: the}
# {2: of}
# {3: a}
# {4: and}
# {5: in}

# Reverse the outcome dictionary
index_to_outcome = pr.reverse_dictionary(outcome_to_index)

########################
# Train/valid/test split
########################

# Create train, valid and test set files
pr.df_train_valid_test_split(data = tense_full, 
                             train_data_path = TENSE_TRAIN, 
                             valid_data_path = TENSE_VALID, 
                             test_data_path = TENSE_TEST, 
                             p_valid = prop_valid, 
                             p_test = prop_test)
# - Number of rows in the original set is 7000
# - Number of rows in the training set is 6000
# - Number of rows in the validation set is 500
# - Number of rows in the test set is 500

# Load the train, valid and test sets
tense_train = pd.read_csv(TENSE_TRAIN, sep=',', index_col=0, na_filter = False)
tense_valid = pd.read_csv(TENSE_VALID, sep=',', index_col=0, na_filter = False)
tense_test = pd.read_csv(TENSE_VALID, sep=',', index_col=0, na_filter = False)

# Check the files
len(tense_train) # 6000
len(tense_valid) # 500
len(tense_test) # 500

#############
# FNN model
#############

### Tokenisation
event2 = tense_train.iloc[[2]]
event2
#                                     cues  outcomes
# 1687  an_hour_past_he_with_his_ministers  PastPerf

# cues
cues2 = tense_train.loc[tense_train.index[2], 'cues']
cues2_onehot_FNN = md.seq_to_onehot_1darray(cues2, index_system = cue_to_index, N_tokens = N_cues)
cues2_onehot_FNN[:50]
# array([0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0.,
#        1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

# outcomes
outcomes2 = tense_train.loc[tense_train.index[2], 'outcomes']
outcomes2_onehot_FNN = md.seq_to_onehot_1darray(outcomes2, index_system = outcome_to_index, N_tokens = N_outcomes)
outcomes2_onehot_FNN
# array([0., 0., 0., 0., 1., 0., 0.])

### Build a simple FNN with two hidden layers having 64 units 
p = {'epochs': 10, # number of iterations on the full set 
    'batch_size': 16, 
    'hidden_layers': 2, # number of hidden layers 
    'hidden_neuron':64, # number of neurons in the input layer 
    'lr': 0.0001, # learning rate       
    'dropout': 0, 
    'optimizer': Adam, 
    'losses': categorical_crossentropy, 
    'activation': relu, 
    'last_activation': 'softmax'}

FNN_out, FNN_model = md.train_FNN(data_train = tense_train, 
                                  data_valid = tense_valid, 
                                  num_cues = N_cues, 
                                  num_outcomes = N_outcomes, 
                                  cue_index = cue_to_index, 
                                  outcome_index = outcome_to_index, 
                                  generator = md.generator_df_FNN,
                                  shuffle = False, 
                                  use_multiprocessing = True, 
                                  num_threads = no_threads, 
                                  verbose = 2,
                                  metrics = ['accuracy'],
                                  params = p)
# Epoch 1/10
#  - 1s - loss: 1.9463 - acc: 0.1535 - val_loss: 1.9384 - val_acc: 0.1956
# Epoch 2/10
#  - 1s - loss: 1.9323 - acc: 0.2013 - val_loss: 1.9281 - val_acc: 0.2298
# Epoch 3/10
#  - 1s - loss: 1.9126 - acc: 0.2487 - val_loss: 1.9083 - val_acc: 0.2440
# Epoch 4/10
#  - 1s - loss: 1.8790 - acc: 0.2907 - val_loss: 1.8751 - val_acc: 0.2560
# Epoch 5/10
#  - 1s - loss: 1.8305 - acc: 0.3167 - val_loss: 1.8368 - val_acc: 0.2540
# Epoch 6/10
#  - 1s - loss: 1.7788 - acc: 0.3403 - val_loss: 1.8071 - val_acc: 0.2540
# Epoch 7/10
#  - 2s - loss: 1.7329 - acc: 0.3668 - val_loss: 1.7878 - val_acc: 0.2560
# Epoch 8/10
#  - 1s - loss: 1.6921 - acc: 0.3823 - val_loss: 1.7752 - val_acc: 0.2560
# Epoch 9/10
#  - 1s - loss: 1.6531 - acc: 0.4015 - val_loss: 1.7629 - val_acc: 0.2782
# Epoch 10/10
#  - 2s - loss: 1.6152 - acc: 0.4173 - val_loss: 1.7553 - val_acc: 0.2843

FNN_model.summary()
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# dense_7 (Dense)              (None, 64)                64064     
# _________________________________________________________________
# dropout_5 (Dropout)          (None, 64)                0         
# _________________________________________________________________
# dense_8 (Dense)              (None, 64)                4160      
# _________________________________________________________________
# dropout_6 (Dropout)          (None, 64)                0         
# _________________________________________________________________
# dense_9 (Dense)              (None, 7)                 455       
# =================================================================
# Total params: 68,679
# Trainable params: 68,679
# Non-trainable params: 0
# _________________________________________________________________

# Save the model and training history
MODEL_PATH = TOP + 'illustrative_examples/tense_1k/Results/FNN_tense1k.h5'
HISTORY_PATH = TOP + 'illustrative_examples/tense_1k/Results/FNN_history_dict_tense1k'
FNN_model.save(MODEL_PATH)  # creates a HDF5 file 
md.save_history(history = FNN_out, path = HISTORY_PATH)
del FNN_model, FNN_out  # deletes the existing model and history dictionary

# Load the model and training history
FNN_model = load_model(MODEL_PATH)
FNN_history_dict = md.load_history(path = HISTORY_PATH)

### Parameter tuning using grid search 
p = {'epochs': [1, 5, 10, 15, 20], # number of iterations on the full set (x5)
     'batch_size': [8, 16, 32, 64, 128, 256], # (x6)
     'hidden_layers':[0, 1, 2], # number of hidden layers (x3)
     'hidden_neuron':[16, 32, 64, 128], # number of neurons in the input layer (x4)
     'lr': [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01], # learning rate (x7)       
     'dropout': [0, 0.1, 0.2, 0.3, 0.4], # (x5)
     'optimizer': [Adam, Nadam, RMSprop, SGD], # (x4)
     'losses': [categorical_crossentropy], # (x1)
     'activation':[relu, elu], # (x2)
     'last_activation': ['softmax'] # (x1)
     }
# => Total number of combinations: 5*6*3*4*7*5*4*1*2*1 = 100800

TUNING_PATH = TOP + 'illustrative_examples/tense_1k/Results/grid_search_FNN_tense1k.csv'
md.grid_search_FNN(data_train = tense_train, 
                   data_valid = tense_valid, 
                   num_cues = N_cues, 
                   num_outcomes = N_outcomes, 
                   cue_index = cue_to_index, 
                   outcome_index = outcome_to_index,
                   generator = md.generator_df_FNN,
                   params = p,
                   prop_grid = 1e-4, 
                   tuning_output_file = TUNING_PATH, 
                   shuffle = False, 
                   use_multiprocessing = True, 
                   num_threads = no_threads, 
                   verbose = 0)

### Evaluation

# Performance on the last epoch of the training set
FNN_history_dict['loss'][-1]
# 1.6152411381403604
FNN_history_dict['acc'][-1]
# 0.41733333333333333

# Performance on the last epoch of the validation set
FNN_history_dict['val_loss'][-1]
# 1.7552948151865313
FNN_history_dict['val_acc'][-1]
# 0.2842741935483871

# Generate plots to assess the performance of the NN
ev.plot_learning_curve(history_dict = FNN_history_dict, metric = 'acc', set = 'train_valid')

# Test prediction for a single given cue sequence. Model expect inout as array of shape (1, N_cues) 
cue1_seq = 'the_boy_is_currently_playing'
outcome1_prob_pred = ev.predict_proba_oneevent_FNN(model = FNN_model, 
                                                   cue_seq = cue1_seq, 
                                                   num_cues = N_cues,  
                                                   cue_index = cue_to_index)
# Top 5 predicted outcomes along with their corresponding probabilities as a dictionary
ev.top_predicted_outcomes(proba_pred = outcome1_prob_pred, 
                          index_to_outcome_dict = index_to_outcome, 
                          N_top = 3)
# {'PresentProg': 0.24823229, 'PastSimple': 0.14647272, 'PastPerf': 0.14621791}

# Evaluate the model on the test set
prob_pred = ev.predict_proba_eventfile_FNN(model = FNN_model, 
                                           data_test = tense_test, 
                                           num_cues = N_cues, 
                                           num_outcomes = N_outcomes, 
                                           cue_index = cue_to_index, 
                                           outcome_index = outcome_to_index, 
                                           generator = md.generator_df_FNN,
                                           use_multiprocessing = True, 
                                           num_threads = no_threads, 
                                           verbose = 0)
# Prediction for the first 5 sentences
prob_pred[0:5,]
# array([[0.09313215, 0.11285219, 0.13910167, 0.14098921, 0.09828919, 0.28332648, 0.13230906],
#        [0.186723  , 0.10267378, 0.09709628, 0.09116439, 0.1707888 , 0.17686719, 0.17468664],
#        [0.03283693, 0.34060234, 0.20521362, 0.23280212, 0.02222909, 0.12314379, 0.04317214],
#        [0.00878896, 0.3501642 , 0.18847059, 0.16632146, 0.00699056, 0.25786972, 0.02139443],
#        [0.04613779, 0.12868719, 0.36606303, 0.2042106 , 0.11310422, 0.05487594, 0.08692114]], dtype=float32)

# True responses to compare the predictions to
y_test = tense_test.replace({'outcomes': outcome_to_index})['outcomes']
y_pred = np.argmax(prob_pred, axis=1)+1

# Overall test accuracy
accuracy_score(y_test, y_pred)
# 0.284

# Test accuracy per class
cmat = confusion_matrix(y_test, y_pred) # Confusion matrix
print(cmat.diagonal()/cmat.sum(axis=1)) 
# [0.21333333 0.33333333 0.30666667 0.25 0.33333333 0.36144578 0.15254237]

#############
# LSTM model
#############

### Tokenisation
event2 = tense_train.iloc[[2]]
event2
#                                     cues  outcomes
# 1687  an_hour_past_he_with_his_ministers  PastPerf

# cues
cues2 = tense_train.loc[tense_train.index[2], 'cues']
cues2_onehot_LSTM = md.seq_to_onehot_2darray(cues2, index_system = cue_to_index, N_tokens = N_cues, max_len = max_len)
cues2_onehot_LSTM[1:10, :20]
# array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])

# outcomes
outcomes2 = tense_train.loc[tense_train.index[2], 'outcomes']
outcomes2_onehot_LSTM = md.seq_to_onehot_1darray(outcomes2, index_system = outcome_to_index, N_tokens = N_outcomes)
outcomes2_onehot_LSTM
# array([0., 0., 0., 0., 1., 0., 0.])

### Build a simple LSTM that has 64 hidden units 
p = {'epochs': 10, # number of iterations on the full set 
    'batch_size': 32, 
    'hidden_neuron': 64, # number of neurons in the input layer 
    'lr': 0.0001, # learning rate       
    'dropout': 0, 
    'optimizer': RMSprop, 
    'losses': categorical_crossentropy, 
    'last_activation': 'softmax'}

LSTM_out, LSTM_model = md.train_LSTM(data_train = tense_train, 
                                     data_valid = tense_valid, 
                                     num_cues = N_cues, 
                                     num_outcomes = N_outcomes, 
                                     cue_index = cue_to_index, 
                                     outcome_index = outcome_to_index, 
                                     max_len = max_len,
                                     generator = md.generator_df_LSTM,
                                     shuffle = False, 
                                     use_cuda = False,
                                     use_multiprocessing = True, 
                                     num_threads = no_threads, 
                                     verbose = 2,
                                     metrics = ['accuracy'],
                                     params = p)
# Epoch 1/10
#  - 8s - loss: 1.9458 - acc: 0.1400 - val_loss: 1.9457 - val_acc: 0.1729
# Epoch 2/10
#  - 7s - loss: 1.9450 - acc: 0.1529 - val_loss: 1.9445 - val_acc: 0.1708
# Epoch 3/10
#  - 6s - loss: 1.9382 - acc: 0.1768 - val_loss: 1.9261 - val_acc: 0.2021
# Epoch 4/10
#  - 6s - loss: 1.8934 - acc: 0.2032 - val_loss: 1.8795 - val_acc: 0.2250
# Epoch 5/10
#  - 8s - loss: 1.8524 - acc: 0.2109 - val_loss: 1.8581 - val_acc: 0.2292
# Epoch 6/10
#  - 8s - loss: 1.8288 - acc: 0.2176 - val_loss: 1.8463 - val_acc: 0.2146
# Epoch 7/10
#  - 9s - loss: 1.8134 - acc: 0.2254 - val_loss: 1.8387 - val_acc: 0.2021
# Epoch 8/10
#  - 8s - loss: 1.8015 - acc: 0.2289 - val_loss: 1.8376 - val_acc: 0.2042
# Epoch 9/10
#  - 9s - loss: 1.7924 - acc: 0.2311 - val_loss: 1.8323 - val_acc: 0.2146
# Epoch 10/10
#  - 7s - loss: 1.7849 - acc: 0.2320 - val_loss: 1.8363 - val_acc: 0.2021

LSTM_model.summary()
#_________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# lstm_1 (LSTM)                (None, 64)                272640    
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 64)                0         
# _________________________________________________________________
# dense_1 (Dense)              (None, 7)                 455       
# =================================================================
# Total params: 273,095
# Trainable params: 273,095
# Non-trainable params: 0
# _________________________________________________________________

# Save the model and training history
MODEL_PATH = TOP + 'illustrative_examples/tense_1k/Results/LSTM_tense1k.h5'
HISTORY_PATH = TOP + 'illustrative_examples/tense_1k/Results/LSTM_history_dict_tense1k'
LSTM_model.save(MODEL_PATH)  # creates a HDF5 file 
md.save_history(history = LSTM_out, path = HISTORY_PATH)
del LSTM_model, LSTM_out  # deletes the existing model and history dictionary

# Load the model and training history
LSTM_model = load_model(MODEL_PATH)
LSTM_history_dict = md.load_history(path = HISTORY_PATH)

### Parameter tuning using grid search 
p = {'epochs': [1, 5, 10, 15, 20], # number of iterations on the full set (x5)
     'batch_size': [8, 16, 32, 64, 128, 256], # (x6)
     'hidden_neuron':[16, 32, 64, 128], # number of neurons in the input layer (x4)
     'lr': [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01], # learning rate (x7)       
     'dropout': [0, 0.1, 0.2, 0.3, 0.4], # (x5)
     'optimizer': [Adam, Nadam, RMSprop, SGD], # (x4)
     'losses': [binary_crossentropy], # (x1)
     'last_activation': ['sigmoid'] # (x1)
     }
# => Total number of combinations: 5*6*4*7*5*4*1*1 = 16800

TUNING_PATH = TOP + 'illustrative_examples/tense_1k/Results/grid_search_LSTM_tense1k.csv'
md.grid_search_LSTM(data_train = tense_train, 
                    data_valid = tense_valid, 
                    num_cues = N_cues, 
                    num_outcomes = N_outcomes, 
                    cue_index = cue_to_index, 
                    outcome_index = outcome_to_index,
                    max_len = max_len,
                    generator = md.generator_df_LSTM,
                    params = p,
                    prop_grid = 2e-4, 
                    tuning_output_file = TUNING_PATH, 
                    shuffle = False, 
                    use_cuda = False,
                    use_multiprocessing = True, 
                    num_threads = no_threads, 
                    verbose = 0)
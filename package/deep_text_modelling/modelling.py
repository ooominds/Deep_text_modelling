### Import necessary packages
import itertools
import csv 
import gc
import sys
import numpy as np
import pandas as pd
import random 
import json
import os
import time

import keras
from keras.models import Sequential
from keras.layers import Dropout, Dense, LSTM, CuDNNLSTM
from keras import optimizers
from keras import activations
from keras import losses
from keras import metrics
from keras import backend as K

from pyndl.preprocess import filter_event_file
from pyndl.count import cues_outcomes
from pyndl.ndl import ndl
from pyndl.activation import activation

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

### Import local packages
from deep_text_modelling.evaluation import recall, precision, f1score

###############
# Tokenisation
###############

def seq_to_onehot_1darray(seq, index_system, N_tokens):

    """Convert a text sequence to a one-hot 1d array (used by FNN)

    Parameters
    ----------
    seq: str
        string to convert
    index_system: dict
        index system giving the mapping from tokens to indices
    N_tokens: int
        number of allowed tokens (cues or outcomes). This determines the length of the one-hot array

    Returns
    -------
    numpy array
        one-hot unidimensional array with 0s everywhere except for the indices corresponding to the tokens that appear in the sequence 
    """

    # Initialise the array
    onehot_list = np.zeros(N_tokens + 1)

    # List of words in the sentence 
    targets = seq.split('_')
    target_indices = [index_system[w] for w in targets if w in index_system]
    onehot_list[target_indices] = 1
    onehot_list = onehot_list[1:]

    return onehot_list

def seq_to_onehot_2darray(seq, index_system, N_tokens, max_len):

    """Convert a text sequence to a one-hot 2d array (used by LSTM)

    Parameters
    ----------
    seq: str
        string to convert (e.g. sequence of cues)
    index_system: dict
        index system giving the mapping from tokens to indices
    N_tokens: int
        number of allowed tokens. This determines the number of columns of the one-hot array
    max_len: int
        Consider only 'max_len' first tokens to in a sequence

    Returns
    -------
    numpy array
        one-hot two-dimensional array with the i-th row giving the one-hot encoding of the i-th token (0s everywhere except for the indices corresponding to the token) 
    """

    # Initialise the array
    onehot_array = np.zeros((max_len, (N_tokens+1)))

    # List of words in the sentence 
    targets = seq.split('_')
    target_indices = [index_system[w] for w in targets if w in index_system]

    # First define the maximum index of word from the current sentence i to be considered
    max_len_cur = min(max_len, len(target_indices)) # Current max_len
    for j in range(max_len_cur):
        onehot_array[j, target_indices[j]] = 1 
    
    # Trim the first column (empty)
    onehot_array = onehot_array[:, 1:]

    return onehot_array
                                         
###############################
# Feedforward neural networks
###############################

class generator_textfile_FNN(keras.utils.Sequence):

    """ Class that generates batches of data ready for training a FNN model. The data is expected to  
    come from an indexed text file and to follow event style (cue and ouctomes seperated by underscores)

    Attributes
    ----------
    data: class
        indexed text file (no heading)
    batch_size: int
        number of examples in each batch 
    num_cues: int
        number of allowed cues
    num_outcomes: int
        number of allowed outcomes
    cue_index: dict
        mapping from cues to indices
    outcome_index: dict
        mapping from outcomes to indices
    shuffle: Boolean
        whether to shuffle the data after every epoch

    Returns
    -------
    class object
        generator for keras. It inherites from keras.utils.Sequence  
    """

    def __init__(self, data, batch_size, num_cues, num_outcomes, 
                 cue_index, outcome_index, shuffle = False):
        'Initialization'
        self.data =  data
        self.batch_size = batch_size    
        self.num_cues = num_cues
        self.num_outcomes = num_outcomes
        self.cue_index = cue_index
        self.outcome_index = outcome_index
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes_batch = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        X, Y = self.__data_generation(indexes_batch)
        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes_batch):
        'Generates data containing batch_size samples'
        X_arrays = []
        Y_arrays = []
        for raw_event in self.data[indexes_batch]:
            # extract the cues and outcomes sequences
            cue_seq, outcome_seq = raw_event.strip().split('\t')
            cues_onehot = seq_to_onehot_1darray(cue_seq, self.cue_index, self.num_cues)
            outcomes_onehot = seq_to_onehot_1darray(outcome_seq, self.outcome_index, self.num_outcomes)
            X_arrays.append(cues_onehot)
            Y_arrays.append(outcomes_onehot)
        X =  np.stack(X_arrays, axis = 0)
        Y =  np.stack(Y_arrays, axis = 0)

        # Generate data
        return X, Y

class generator_df_FNN(keras.utils.Sequence):

    """ Class that generates batches of data ready for training a FNN model. The data is expected to  
    come from a dataframe and to follow event style (cue and ouctomes seperated by underscores)

    Attributes
    ----------
    data: dataframe
        dataframe with the first column containing the cues and second colum containing the outcomes
    batch_size: int
        number of examples in each batch 
    num_cues: int
        number of allowed cues
    num_outcomes: int
        number of allowed outcomes
    cue_index: dict
        mapping from cues to indices
    outcome_index: dict
        mapping from outcomes to indices
    shuffle: Boolean
        whether to shuffle the data after every epoch

    Returns
    -------
    class object
        generator for keras. It inherites from keras.utils.Sequence  
    """

    def __init__(self, data, batch_size, num_cues, num_outcomes, 
                 cue_index, outcome_index, shuffle = False):
        'Initialization'
        self.data =  data
        self.batch_size = batch_size    
        self.num_cues = num_cues
        self.num_outcomes = num_outcomes
        self.cue_index = cue_index
        self.outcome_index = outcome_index
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes_batch = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        X, Y = self.__data_generation(indexes_batch)
        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes_batch):
        'Generates data containing batch_size samples' 
        X_arrays = [seq_to_onehot_1darray(cue_seq, self.cue_index, self.num_cues) for cue_seq in self.data.loc[self.data.index[indexes_batch], 'cues']]
        X =  np.stack(X_arrays, axis=0)

        Y_arrays = [seq_to_onehot_1darray(outcome_seq, self.outcome_index, self.num_outcomes) for outcome_seq in self.data.loc[self.data.index[indexes_batch], 'outcomes']]
        Y =  np.stack(Y_arrays, axis=0)

        # Generate data
        return X, Y

def train_FNN(data_train, data_valid, num_cues, 
              num_outcomes, cue_index, outcome_index, 
              generator = generator_textfile_FNN, shuffle = False, 
              use_multiprocessing = False, num_threads = 0, verbose = 0,
              metrics = ['accuracy', precision, recall, f1score],
              params = {'epochs': 1, # number of iterations on the full set 
                        'batch_size': 128, 
                        'hidden_layers': 0, # number of hidden layers 
                        'hidden_neuron':64, # number of neurons in the input layer 
                        'lr': 0.0001, # learning rate       
                        'dropout': 0, 
                        'optimizer': optimizers.Adam, 
                        'losses': losses.binary_crossentropy, 
                        'activation': activations.relu, 
                        'last_activation': 'sigmoid'}):

    """ Train a feedforward neural network

    Parameters
    ----------
    data_train: dataframe or class
        dataframe or indexed text file containing training data
    data_valid: class or dataframe
        dataframe or indexed text file containing validation data
    num_cues: int
        number of allowed cues
    num_outcomes: int
        number of allowed outcomes
    cue_index: dict
        mapping from cues to indices
    outcome_index: dict
        mapping from outcomes to indices
    generator: class
        use 'generator = generator_df_FNN' if the data is given as a dataframe or 
        'generator = generator_textfile_FNN' if the data is given as an indexed file 
    shuffle: Boolean
        whether to shuffle the data after every epoch
    use_multiprocessing: Boolean
        whether to generate batches in parallel. Default: False
    num_threads: int
        maximum number of processes to spin up when using generating the batches. Default: 0
    verbose: int (0, 1, or 2)
        verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
    metrics: list
    params: dict
        model parameters:
        'epochs'
        'batch_size'
        'hidden_layers'
        'hidden_neuron'
        'lr'
        'dropout'
        'optimizer'
        'losses'
        'activation'
        'last_activation'

    Returns
    -------
    tuple
        keras fit history and model objects  
    """

    ### Initiate the generators for the train, valid and test data
    train_gen = generator(data = data_train, 
                          batch_size = params['batch_size'],
                          num_cues = num_cues,
                          num_outcomes = num_outcomes,
                          cue_index = cue_index,
                          outcome_index = outcome_index,
                          shuffle = shuffle)
    valid_gen = generator(data = data_valid, 
                          batch_size = params['batch_size'],
                          num_cues = num_cues,
                          num_outcomes = num_outcomes,
                          cue_index = cue_index,
                          outcome_index = outcome_index,
                          shuffle = shuffle)

    ### Initialise the model
    model = Sequential()  

    ### Add the layers depending on the parameter 'hidden_layers'

    # If there is no hidden layer add the output layer directly
    if params['hidden_layers'] == 0:
        # Output layer
        model.add(Dense(num_outcomes, 
                        input_dim = num_cues,
                        activation = params['last_activation']))
    # If there are hidden layers
    else:
        # First hidden layer   
        model.add(Dense(params['hidden_neuron'],
                        input_dim = num_cues,
                        activation = params['activation']))
        # Add drop out
        model.add(Dropout(params['dropout']))

        # Add more hidden layers if they were requested 
        if params['hidden_layers'] > 1:
            
            for j in range(2, (params['hidden_layers']+1)):
                # Add jth hidden layer
                model.add(Dense(params['hidden_neuron'],
                                activation = params['activation']))  
                # Add drop out
                model.add(Dropout(params['dropout']))
            
        # Add output layer
        model.add(Dense(num_outcomes, 
                        activation = params['last_activation'])) 

    
    ### Compile the model 
    model.compile(loss = params['losses'],
                  optimizer = params['optimizer'](lr = params['lr']),
                  metrics = metrics)
    
    # Fit the model 
    out = model.fit_generator(generator = train_gen,
                              validation_data = valid_gen,
                              epochs = params['epochs'],
                              use_multiprocessing = use_multiprocessing,
                              verbose = verbose,
                              workers = num_threads)
    
    return out, model

def grid_search_FNN(data_train, data_valid, num_cues, 
                    num_outcomes, cue_index, outcome_index, 
                    generator, params, prop_grid, tuning_output_file,         
                    shuffle = False, use_multiprocessing = False, 
                    num_threads = 0, verbose = 0):

    """ Grid search for feedforward neural networks

    Parameters
    ----------
    data_train: dataframe or class
        dataframe or indexed text file containing training data
    data_valid: class or dataframe
        dataframe or indexed text file containing validation data
    num_cues: int
        number of allowed cues
    num_outcomes: int
        number of allowed outcomes
    cue_index: dict
        mapping from cues to indices
    outcome_index: dict
        mapping from outcomes to indices
    generator: class
        use 'generator = generator_df_FNN' if the data is given as a dataframe or 
        'generator = generator_textfile_FNN' if the data is given as an indexed file 
    params: dict of lists
        model parameters:
        'epochs'
        'batch_size'
        'hidden_layers'
        'hidden_neuron'
        'lr'
        'dropout'
        'optimizer'
        'losses'
        'activation'
        'last_activation'
    prop_grid: float
        proportion of the grid combinations to sample 
    tuning_output_file: str
        path of the csv file where the grid search results will be stored
    shuffle: Boolean
        whether to shuffle the data after every epoch
    use_multiprocessing: Boolean
        whether to generate batches in parallel. Default: False
    num_threads: int
        maximum number of processes to spin up when using generating the batches. Default: 0
    verbose: int (0, 1, or 2)
        verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.

    Returns
    -------
    None
        save csv files
    """

    ### Create a list of dictionaries giving all possible parameter combinations
    keys, values = zip(*params.items())
    grid_full = [dict(zip(keys, v)) for v in itertools.product(*values)]
    # shuffle the list of params
    random.shuffle(grid_full)

    ### Select the combinations to use 
    N_comb = round(prop_grid * len(grid_full)) 
    grid_select = grid_full[:N_comb]

    ### Write to the csv file that encodes the results
    with open(tuning_output_file, mode = 'w') as o:
        csv_writer = csv.writer(o, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        heading = list(params.keys())
        heading.extend(['loss', 'acc', 'precision', 'recall', 'f1score', 
                        'val_loss', 'val_acc', 'val_precision', 'val_recall', 'val_f1score'])
        csv_writer.writerow(heading)

        ### Run the experiments
        for i, param_comb in enumerate(grid_select):

            # Message at the start of each iteration
            print(f'Iteration {i+1} out of {len(grid_select)}: {param_comb}\n')

            # Fit the model given the current param combination
            out, model = train_FNN(data_train = data_train, 
                                   data_valid = data_valid, 
                                   num_cues = num_cues, 
                                   num_outcomes = num_outcomes, 
                                   cue_index = cue_index, 
                                   outcome_index = outcome_index, 
                                   generator = generator, 
                                   shuffle = False, 
                                   use_multiprocessing = False, 
                                   num_threads = 0, 
                                   verbose = verbose,
                                   metrics = ['accuracy', precision, recall, f1score],
                                   params = param_comb)

            # Export the results to a csv file
            row_values = list(param_comb.values())
            # Add the performance scores
            # training
            loss_i = out.history['loss'][-1]
            acc_i = out.history['acc'][-1]
            precision_i = out.history['precision'][-1]
            recall_i = out.history['recall'][-1]            
            f1score_i = out.history['f1score'][-1]
            # validation
            val_loss_i = out.history['val_loss'][-1]
            val_acc_i = out.history['val_acc'][-1]
            val_precision_i = out.history['val_precision'][-1]
            val_recall_i = out.history['val_recall'][-1]            
            val_f1score_i = out.history['val_f1score'][-1]
            row_values.extend([loss_i, acc_i, precision_i, recall_i, f1score_i, 
                               val_loss_i, val_acc_i, val_precision_i, val_recall_i, val_f1score_i])
            # Write the row
            csv_writer.writerow(row_values)
            o.flush()

            # Clear memory           
            del model, out
            gc.collect()
            K.clear_session()

########
# LSTM
########

class generator_textfile_LSTM(keras.utils.Sequence):

    """ Class that generates batches of data ready for training an LSTM model. The data is expected to  
    be event style (cue and ouctomes seperated by underscores)

    Attributes
    ----------
    data: class
        indexed text file
    batch_size: int
        number of examples in each batch 
    num_cues: int
        number of allowed cues
    num_outcomes: int
        number of allowed outcomes
    cue_index: dict
        mapping from cues to indices
    outcome_index: dict
        mapping from outcomes to indices
    max_len: int
        Consider only 'max_len' first tokens in a sequence
    shuffle: Boolean
        whether to shuffle the data after every epoch

    Returns
    -------
    class object
        generator for keras. It inherites from keras.utils.Sequence  
    """

    def __init__(self, data, batch_size, num_cues, num_outcomes, 
                 cue_index, outcome_index, max_len, shuffle = False):
        'Initialization'
        self.data =  data
        self.batch_size = batch_size    
        self.num_cues = num_cues
        self.num_outcomes = num_outcomes
        self.cue_index = cue_index
        self.outcome_index = outcome_index
        self.max_len = max_len
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes_batch = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        X, Y = self.__data_generation(indexes_batch)
        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes_batch):
        'Generates data containing batch_size samples'
        X_arrays = []
        Y_arrays = []
        for raw_event in self.data[indexes_batch]:
            # extract the cues and outcomes sequences
            cue_seq, outcome_seq = raw_event.strip().split('\t')
            cues_onehot = seq_to_onehot_2darray(cue_seq, self.cue_index, self.num_cues, self.max_len)
            outcomes_onehot = seq_to_onehot_1darray(outcome_seq, self.outcome_index, self.num_outcomes)
            X_arrays.append(cues_onehot)
            Y_arrays.append(outcomes_onehot)
        X =  np.stack(X_arrays, axis = 0)
        Y =  np.stack(Y_arrays, axis = 0)

        # Generate data
        return X, Y

class generator_df_LSTM(keras.utils.Sequence):

    """ Class that generates batches of data ready for training an LSTM model. The data is expected to  
    come from a dataframe and to follow event style (cue and ouctomes seperated by underscores)

    Attributes
    ----------
    data: dataframe
        dataframe with the first column containing the cues and second colum containing the outcomes
    batch_size: int
        number of examples in each batch 
    num_cues: int
        number of allowed cues
    num_outcomes: int
        number of allowed outcomes
    cue_index: dict
        mapping from cues to indices
    outcome_index: dict
        mapping from outcomes to indices
    max_len: int
        Consider only 'max_len' first tokens in a sequence
    shuffle: Boolean
        whether to shuffle the data after every epoch

    Returns
    -------
    class object
        generator for keras. It inherites from keras.utils.Sequence  
    """

    def __init__(self, data, batch_size, num_cues, num_outcomes, 
                 cue_index, outcome_index, max_len, shuffle = False):
        'Initialization'
        self.data =  data
        self.batch_size = batch_size    
        self.num_cues = num_cues
        self.num_outcomes = num_outcomes
        self.cue_index = cue_index
        self.outcome_index = outcome_index
        self.max_len = max_len
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes_batch = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        X, Y = self.__data_generation(indexes_batch)
        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes_batch):

        'Generates data containing batch_size samples' # X : (batch_size, *dim, n_channels)
        X_arrays = [seq_to_onehot_2darray(cue_seq, self.cue_index, self.num_cues, self.max_len) for cue_seq in self.data.loc[self.data.index[indexes_batch], 'cues']]
        X =  np.stack(X_arrays, axis=0)

        Y_arrays = [seq_to_onehot_1darray(outcome_seq, self.outcome_index, self.num_outcomes) for outcome_seq in self.data.loc[self.data.index[indexes_batch], 'outcomes']]
        Y =  np.stack(Y_arrays, axis=0)

        # Generate data
        return X, Y

def train_LSTM(data_train, data_valid, num_cues, 
               num_outcomes, cue_index, outcome_index, max_len, 
               generator = generator_textfile_LSTM, shuffle = False, 
               use_cuda = False, use_multiprocessing = False, 
               num_threads = 0, verbose = 0,
               metrics = ['accuracy', precision, recall, f1score],
               params = {'epochs': 1, # number of iterations on the full set 
                         'batch_size': 128, 
                         'hidden_neuron':64, # number of neurons in the input layer 
                         'lr': 0.0001, # learning rate       
                         'dropout': 0, 
                         'optimizer': optimizers.RMSprop, 
                         'losses': losses.binary_crossentropy, 
                         'last_activation': 'sigmoid'}):

    """ Train an LSTM

    Parameters
    ----------
    data_train: dataframe or class
        dataframe or indexed text file containing training data
    data_valid: class or dataframe
        dataframe or indexed text file containing validation data
    num_cues: int
        number of allowed cues
    num_outcomes: int
        number of allowed outcomes
    cue_index: dict
        mapping from cues to indices
    outcome_index: dict
        mapping from outcomes to indices
    max_len: int
        Consider only 'max_len' first tokens in a sequence
    generator: class
        use 'generator = generator_df_LSTM' if the data is given as a dataframe or 
        'generator = generator_textfile_LSTM' if the data is given as an indexed file    
    shuffle: Boolean
        whether to shuffle the data after every epoch
    use_cuda: Boolean
        whether to use the cuda optimised LSTM layer for faster training. Use only if 
        an Nvidia GPU is available with CUDA installed
    use_multiprocessing: Boolean
        whether to generate batches in parallel. Default: False
    num_threads: int
        maximum number of processes to spin up when using generating the batches. Default: 0
    verbose: int (0, 1, or 2)
        verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
    metrics: list
    params: dict
        model parameters:
        'epochs'
        'batch_size'
        'hidden_neuron'
        'lr'
        'dropout'
        'optimizer'
        'losses'
        'last_activation'

    Returns
    -------
    tuple
        keras fit history and model objects  
    """

    ### Initiate the generators for the train, valid and test data
    train_gen = generator(data = data_train, 
                          batch_size = params['batch_size'],
                          num_cues = num_cues,
                          num_outcomes = num_outcomes,
                          cue_index = cue_index,
                          outcome_index = outcome_index,
                          max_len = max_len,
                          shuffle = shuffle)
    valid_gen = generator(data = data_valid, 
                          batch_size = params['batch_size'],
                          num_cues = num_cues,
                          num_outcomes = num_outcomes,
                          cue_index = cue_index,
                          outcome_index = outcome_index,
                          max_len = max_len,
                          shuffle = shuffle)

    ### Initialise the model
    model = Sequential()  

    # LSTM layer 
    if use_cuda == False:
        model.add(LSTM(params['hidden_neuron'], return_sequences = False, input_shape = (max_len, num_cues))) 
    else:
        model.add(CuDNNLSTM(params['hidden_neuron'], return_sequences = False, input_shape = (max_len, num_cues)))

    # Add drop out
    model.add(Dropout(params['dropout']))

    # Add output layer
    model.add(Dense(num_outcomes, 
                    activation = params['last_activation']))

    # Compile the model
    model.compile(loss = params['losses'],
                  optimizer = params['optimizer'](lr = params['lr']),
                  metrics = metrics)

    
    # Fit the model 
    out = model.fit_generator(generator = train_gen,
                              validation_data = valid_gen,
                              epochs = params['epochs'],
                              use_multiprocessing = use_multiprocessing,
                              verbose = verbose,
                              workers = num_threads)
    
    return out, model
 
def grid_search_LSTM(data_train, data_valid, num_cues, 
                     num_outcomes, cue_index, outcome_index, max_len,
                     generator, params, prop_grid, tuning_output_file, 
                     shuffle = False, use_cuda = False, 
                     use_multiprocessing = False, num_threads = 0, verbose = 0):

    """ Grid search for LSTM

    Parameters
    ----------
    data_train: dataframe or class
        dataframe or indexed text file containing training data
    data_valid: class or dataframe
        dataframe or indexed text file containing validation data
    num_cues: int
        number of allowed cues
    num_outcomes: int
        number of allowed outcomes
    cue_index: dict
        mapping from cues to indices
    outcome_index: dict
        mapping from outcomes to indices
    max_len: int
        Consider only 'max_len' first tokens in a sequence
    generator: class
        use 'generator = generator_df_LSTM' if the data is given as a dataframe or 
        'generator = generator_textfile_LSTM' if the data is given as an indexed file    
    params: dict
        model parameters:
        'epochs'
        'batch_size'
        'hidden_neuron'
        'lr'
        'dropout'
        'optimizer'
        'losses'
        'last_activation'
    prop_grid: float
        proportion of the grid combinations to sample 
    tuning_output_file: str
        path of the csv file where the grid search results will be stored
    shuffle: Boolean
        whether to shuffle the data after every epoch
    use_cuda: Boolean
        whether to use the cuda optimised LSTM layer for faster training. Use only if 
        an Nvidia GPU is available with CUDA installed
    use_multiprocessing: Boolean
        whether to generate batches in parallel. Default: False
    num_threads: int
        maximum number of processes to spin up when using generating the batches. Default: 0
    verbose: int (0, 1, or 2)
        verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.

    Returns
    -------
    None
        save csv files
    """

    ### Create a list of dictionaries giving all possible parameter combinations
    keys, values = zip(*params.items())
    grid_full = [dict(zip(keys, v)) for v in itertools.product(*values)]
    # shuffle the list of params
    random.shuffle(grid_full)

    ### Select the combinations to use 
    N_comb = round(prop_grid * len(grid_full)) 
    grid_select = grid_full[:N_comb]

    ### Write to the csv file that encodes the results
    with open(tuning_output_file, mode = 'w') as o:
        csv_writer = csv.writer(o, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        heading = list(params.keys())
        heading.extend(['loss', 'acc', 'precision', 'recall', 'f1score', 
                        'val_loss', 'val_acc', 'val_precision', 'val_recall', 'val_f1score'])
        csv_writer.writerow(heading)

        ### Run the experiments
        for i, param_comb in enumerate(grid_select):

            # Message at the start of each iteration
            print(f'Iteration {i+1} out of {len(grid_select)}: {param_comb}\n')

            # Fit the model given the current param combination
            out, model = train_LSTM(data_train = data_train, 
                                    data_valid = data_valid, 
                                    num_cues = num_cues, 
                                    num_outcomes = num_outcomes, 
                                    cue_index = cue_index, 
                                    outcome_index = outcome_index, 
                                    max_len = max_len,
                                    generator = generator,
                                    shuffle = shuffle, 
                                    use_cuda = use_cuda, 
                                    use_multiprocessing = use_multiprocessing, 
                                    num_threads = num_threads, 
                                    verbose = verbose,
                                    metrics = ['accuracy', precision, recall, f1score],
                                    params = param_comb)

            # Export the results to a csv file
            row_values = list(param_comb.values())
            # Add the performance scores
            # training
            loss_i = out.history['loss'][-1]
            acc_i = out.history['acc'][-1]
            precision_i = out.history['precision'][-1]
            recall_i = out.history['recall'][-1]            
            f1score_i = out.history['f1score'][-1]
            # validation
            val_loss_i = out.history['val_loss'][-1]
            val_acc_i = out.history['val_acc'][-1]
            val_precision_i = out.history['val_precision'][-1]
            val_recall_i = out.history['val_recall'][-1]            
            val_f1score_i = out.history['val_f1score'][-1]
            row_values.extend([loss_i, acc_i, precision_i, recall_i, f1score_i, 
                               val_loss_i, val_acc_i, val_precision_i, val_recall_i, val_f1score_i])
            # Write the row
            csv_writer.writerow(row_values)
            o.flush()

            # Clear memory           
            del model, out
            gc.collect()
            K.clear_session()

#####################################
# Naive discriminative learning model
#####################################

class NDL_model():

    """ Class that contains the training results after ndl training (weights, activations and performance measures)

    Attributes
    ----------
    weights: xarray.DataArray
        matrix of association weights
    activations_train: xarray.DataArray
        matrix of activations for the training data
    activations_valid: xarray.DataArray
        matrix of activations for the validation data

    Returns
    -------
    class object

    """

    # performance_hist: dict
    #     dictionary containing the performance scores in all epochs depending on the metrics that were used during training

    def __init__(self, weights, activations_train, activations_valid):
        'Initialization'
        self.weights = weights
        self.activations_train = activations_train
        self.activations_valid = activations_valid
        #self.performance_hist = performance_hist

def train_ndl(data_train, data_valid, cue_index, outcome_index, temp_dir,
              shuffle = False, num_threads = 1, verbose = 0, metrics = ['accuracy'],
              #metrics = ['accuracy', 'precision', 'recall', 'f1score'],
              params = {'epochs': 1, # number of iterations on the full set 
                        'lr': 0.0001}):

    """ Train a native discriminative learning model

    Parameters
    ----------
    data_train: dataframe or str
        dataframe or path to the file containing training data
    data_valid: class or dataframe
        dataframe or path to the file containing validation data
    cue_index: dict
        mapping from cues to indices
    outcome_index: dict
        mapping from outcomes to indices
    temp_dir: str
        directory where to store temporary files while training NDL
    shuffle: Boolean
        whether to shuffle the data after every epoch
    use_multiprocessing: Boolean
        whether to generate batches in parallel. Default: False
    num_threads: int
        maximum number of processes to use when training NDL. Default: 1
    verbose: int (0, 1)
        verbosity mode. 0 = silent, 1 = one line per epoch.
    metrics: list
        for now only ['accuracy'] is accepted
    params: dict
        model parameters:
        'epochs'
        'lr'

    Returns
    -------
    tuple
        fit history and NDL_model class object (stores the weight and activation matrices) 
    """

    from deep_text_modelling.evaluation import activations_to_predictions

    ### Paths of the files
    events_train_path = data_train
    events_valid_path = data_valid
    filtered_events_train_path = os.path.join(temp_dir, 'filtered_events_train.gz')  
    filtered_events_valid_path = os.path.join(temp_dir, 'filtered_events_valid.gz')  

    ### Filter the event files by retaining only the cues and outcomes that are in the index system (most frequent tokens)
    cues_to_keep = [cue for cue in cue_index.keys()]
    outcomes_to_keep = [outcome for outcome in outcome_index.keys()]
    # Train set 
    filter_event_file(events_train_path,
                      filtered_events_train_path,
                      number_of_processes = num_threads,
                      keep_cues = cues_to_keep,
                      keep_outcomes = outcomes_to_keep)
    # Validation set
    filter_event_file(events_valid_path,
                      filtered_events_valid_path,
                      number_of_processes = num_threads,
                      keep_cues = cues_to_keep,
                      keep_outcomes = outcomes_to_keep) 

    # Initialise the weight matrix
    weights = None

    # Initialise the lists where we will store the performance scores in each epoch for the different metrics
    # train
    acc_hist = []
    # precision_hist = []
    # recall_hist = []
    # f1score_hist = []

    # valid
    val_acc_hist = []
    # val_precision_hist = []
    # val_recall_hist = []
    # val_f1score_hist = []
    
    # Train NDL for the chosen number of epochs. Each time save and print the metric scores
    for j in range(1, (1+params['epochs'])):

        # Record start time
        start = time.time()

        # Train ndl to get the weight matrix 
        weights = ndl(events = filtered_events_train_path,
                      alpha = params['lr'], 
                      betas = (1, 1),
                      method = "openmp",
                      weights = weights,
                      number_of_threads = num_threads,
                      remove_duplicates = True,
                      temporary_directory = temp_dir,
                      verbose = False)

        # Compute the activation matrix on the training set
        activations_train = activation(events = filtered_events_train_path, 
                                       weights = weights,
                                       number_of_threads = num_threads,
                                       remove_duplicates = True,
                                       ignore_missing_cues = True)

        # Compute the activation matrix on the validation set
        activations_valid = activation(events = filtered_events_valid_path, 
                                       weights = weights,
                                       number_of_threads = num_threads,
                                       remove_duplicates = True,
                                       ignore_missing_cues = True)

        # Predicted outcomes from the activations
        y_train_pred = activations_to_predictions(activations_train) 
        y_valid_pred = activations_to_predictions(activations_valid)

        ### True outcomes 
        # tain set 
        events_train_df = pd.read_csv(filtered_events_train_path, header = 0, sep='\t', quotechar='"')
        y_train_true = events_train_df['outcomes'].tolist()    
        # validation set 
        events_valid_df = pd.read_csv(filtered_events_valid_path, header = 0, sep='\t', quotechar='"')
        y_valid_true = events_valid_df['outcomes'].tolist()
        
        # Compute performance scores for the different metrics
        # accuracy
        acc_j = accuracy_score(y_train_true, y_train_pred)
        acc_hist.append(acc_j)
        val_acc_j = accuracy_score(y_valid_true, y_valid_pred)
        val_acc_hist.append(val_acc_j)

        # # precision
        # precision_j = precision_score(y_train_true, y_train_pred)
        # precision_hist.append(precision_j)
        # val_precision_j = precision_score(y_valid_true, y_valid_pred)
        # val_precision_hist.append(val_precision_j)

        # # recall
        # recall_j = recall_score(y_train_true, y_train_pred)
        # recall_hist.append(recall_j)
        # val_recall_j = recall_score(y_valid_true, y_valid_pred)
        # val_recall_hist.append(val_recall_j)

        # # F1-score
        # f1score_j = f1_score(y_train_true, y_train_pred)
        # f1score_hist.append(f1score_j)
        # val_f1score_j = f1_score(y_valid_true, y_valid_pred)
        # val_f1score_hist.append(val_f1score_j)  

        # Display progress message  
        if verbose == 1:
            now = time.time()
            sys.stdout.write('Epoch %d/%d\n' % (j, params['epochs']))
            sys.stdout.write(' - %.0fs - acc: %.4f - val_acc: %.4f\n' % ((now - start), acc_j, val_acc_j))
            sys.stdout.flush()

    ### Model object
    model = NDL_model(weights, activations_train, activations_valid)

    ### Fit history object
    hist = {'acc': acc_hist,
            # 'precision': precision_hist,
            # 'recall': recall_hist,
            # 'f1score': f1score_hist,
            'val_acc': val_acc_hist
            # 'val_precision': val_precision_hist,
            # 'val_recall': val_recall_hist,
            # 'val_f1score': val_f1score_hist
            }

    
    return hist, model

##################################
# Saving and loading keras objects
##################################

def save_history(history, path):

    """ Save a keras training history as a dictionary 

    Parameters
    ----------
    history: class
        keras history object
    path: str
        path where to save the file

    Returns
    -------
    None
        export a dictionary as a json file
    """

    # Get the history dictionary containing the performance scores (loss and other metrics) for each epoch
    history_dict = history.history

    # Save it as a json file
    json.dump(history_dict, open(path, 'w'))

def load_history(path):

    """ Save a keras training history as a dictionary 

    Parameters
    ----------
    history: class
        keras history object
    path: str
        path where to save the file

    Returns
    -------
    dict
        history dictionary containing the performance scores (loss and other metrics) for each epoch
    """

    history_dict = json.load(open(path, 'r'))
    return history_dict

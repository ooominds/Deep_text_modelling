### Import necessary packages
import itertools
import csv 
import gc
import sys
import numpy as np
import random 
import json

import keras
from keras.models import Sequential
from keras.layers import Dropout, Dense, LSTM, CuDNNLSTM
from keras import optimizers
from keras import activations
from keras import losses
from keras import metrics
from keras import backend as K

### Import local packages
from Deep_text_classifiers.evaluation import recall, precision, f1score

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

import gzip
import csv
import numpy as np
import random
from collections.abc import Iterable
from collections import Counter
from random import shuffle
from keras.preprocessing.text import Tokenizer

#######################################
# Conversion between csv and gz formats
#######################################

def df_to_gz(data, gz_outfile):

    """Export a dataframe containing events to a .gz file

    Parameters
    ----------
    data: dataframe
        dataframe to export to a gz file
    gz_outfile: str
        path of the gz file  

    Returns
    -------
    None 
        save a .gz file
    """

    with gzip.open(gz_outfile, 'wt', encoding='utf-8') as out:
        data.to_csv(out, sep = '\t', index = False)

def csv_to_gz(csv_infile, gz_outfile):

    """Convert a csv containing events to a .gz file

    Parameters
    ----------
    csv_infile: str
        path of the csv file to convert
    gz_outfile: str
        path of the gz file  

    Returns
    -------
    None 
        save a .gz file
    """

    with gzip.open(gz_outfile, 'wt') as out:
        for line in csv.reader(open(csv_infile, 'r')):
            line = '\t'.join(line)+'\n'
            out.write(line)

def gz_to_csv(gz_infile, csv_outfile):

    """Convert a gz file containing events to csv format

    Parameters
    ----------
    gz_infile: str
        path of the gz file to convert
    csv_outfile: str
        path of the csv file  

    Returns
    -------
    None 
        save a csv file
    """

    with open(csv_outfile, mode = 'w') as outfile:
        with gzip.open(gz_infile, 'rt') as infile:
            csv_writer = csv.writer(outfile, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
            for line in infile:
                csv_writer.writerow(line.strip().split('\t'))

######################################
# Use text file as an indexable object
######################################

class IndexedFile():

    """ Class that adds indexing/slicing capabilities to a text file (.txt, .gz or .csv). This allows to quickly access specific lines in a text file

    Attributes
    ----------
    file_path: str
        text file path
    file_type: str
        could be 'txt', 'csv' or 'gz'

    Returns
    -------
    class object

    Usage
    ----------
    corpus_file = IndexedFile(CORPUS_path, 'gz')
    len(corpus_file) # number of lines
    line = corpus_file[0]  # indexing
    lines = corpus_file[:5]  # slicing (first 5 lines)
    lines = corpus_file[np.array([0, 4, 7])] # line indices given as a numpy array
    lines = corpus_file[::2] # even lines
    """

    def __init__(self, file_path, file_type):
        'Initialization'
        self.file_path = file_path
        self.file_type = file_type
        if self.file_type == 'gz':
            self.file = gzip.open(self.file_path, mode = 'rt')
        else:
            self.file = open(self.file_path)
        self.offsets = self.index_file(self)

    @staticmethod
    def index_file(self):
        offsets = [0]
        if self.file_type == 'gz':
            with gzip.open(self.file_path, mode = 'rt') as f:
                while f.readline():
                    offsets.append(f.tell())
        else:
            with open(self.file_path) as f:
                while f.readline():
                    offsets.append(f.tell())
        return offsets

    def __getitem__(self, index):
        'Generate lines for a specific range of indices'
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [self[idx] for idx in range(start, stop, step)]

        elif isinstance(index, Iterable):
            return [self[idx] for idx in index]

        elif isinstance(index, (int, np.integer)):
            pass

        else:
            raise NotImplementedError(f"Cannot index with {type(index)}!")

        offset = self.offsets[index]
        self.file.seek(offset)
        return self.file.readline()

    def __len__(self):
        'return the number of lines'
        return len(self.offsets) - 1

################################################
# Create index systems for the cues and outcomes
################################################

def create_index_systems_from_counters(cue_counter, outcome_counter, cue_index_path, outcome_index_path):

    """Save index sytems for cues and outcomes from frequency counters for cues and outcomes

    Parameters
    ----------
    cue_counter: counter 
        counter encoding cues and their frequencies. It is expected that this will be generated using cues_outcomes() from pyndl
    outcome_counter: counter 
        counter encoding outcomes and their frequencies. It is expected that this will be generated using cues_outcomes() from pyndl
    cue_index_path: str
        path of the csv file where to save the cue index system 
    outcome_index_path: str
        path of the csv file where to save the outcome index system 

    Returns
    -------
    None 
        save csv files
    """

    # Index systems as dictionaries
    cue_index = {cue:(i+1) for i, (cue, freq) in enumerate(cue_counter.most_common())} 
    outcome_index = {outcome:(i+1) for i, (outcome, freq) in enumerate(outcome_counter.most_common())} 

    ### Export the cue and outcome index systems
    # For the contexts
    with open(cue_index_path, 'w') as fc:
        for key in cue_index.keys():
            fc.write("%s,%s\n"%(key, cue_index[key]))

    # For the tenses
    with open(outcome_index_path, 'w') as fo:
        for key in outcome_index.keys():
            fo.write("%s,%s\n"%(key, outcome_index[key]))

def create_index_systems_from_df(data, cue_index_path, outcome_index_path):

    """Save index sytems for cues and outcomes from a dataframe containing events

    Parameters
    ----------
    data: dataframe
        should have two columns 'cues' and 'outcomes'
    outcome_counter: counter 
        counter encoding outcomes and their frequencies. It is expected that this will be generated using cues_outcomes() from pyndl
    cue_index_path: str
        path of the csv file where to save the cue index system 
    outcome_index_path: str
        path of the csv file where to save the outcome index system 

    Returns
    -------
    None 
        save csv files
    """

    ### Create the cue index system
    # First initialise a tokenizer
    tokenizer_cue = Tokenizer(filters = '', split = '_', lower = False)

    # Second build the tokens index set
    tokenizer_cue.fit_on_texts(data["cues"])

    # Finally extract the tokens index set
    cue_index = tokenizer_cue.word_index 

    ### Create the outcome index system
    # First initialise a tokenizer
    tokenizer_outcome = Tokenizer(filters = '', split = '_', lower = False)

    # Second build the tokens index set
    tokenizer_outcome.fit_on_texts(data["outcomes"])

    # Finally extract the tokens index set
    outcome_index = tokenizer_outcome.word_index 

    ### Export the cue and outcome index systems
    # For the contexts
    with open(cue_index_path, 'w') as fc:
        for key in cue_index.keys():
            fc.write("%s,%s\n"%(key, cue_index[key]))

    # For the tenses
    with open(outcome_index_path, 'w') as fo:
        for key in outcome_index.keys():
            fo.write("%s,%s\n"%(key, outcome_index[key]))
            

def import_index_system(index_system_path, N_tokens = None):

    """Import an index as dictionary

    Parameters
    ----------
    index_system_path: str
        path of the csv file that encodes the index system 
    N_tokens: int
        import only the most frequent tokens if N_tokens is given

    Returns
    -------
    dict
        mapping from tokens (cues or outcomes) to indices
    """

    # Load the index system 
    with open(index_system_path, 'r') as file:
        index_system_df = csv.reader(file)
        index_system_dict = {}
        # Import all indices if N_tokens is not given
        if N_tokens == None:
            for line in index_system_df:
                k, v = line
                index_system_dict[k] = int(v)
        # Limit the index system to the 'N_tokens' first enteries  
        else:
            for i in range(N_tokens):
                k, v = next(index_system_df)
                index_system_dict[k] = int(v)

    return index_system_dict

def display_dictionary(dict_var, start = 0, end = 5):

    """Display some elements of a dictionary

    Parameters
    ----------
    dict_var: dict
    start: int
        index of the first element to display
    end: int
        index of the last element (not included)

    Returns
    -------
    None
        print values on screen
    """

    for key in list(dict_var)[start:end]:
        print ("{{{}: {}}}".format(key, dict_var[key]))

def reverse_dictionary(dict_var):

    """Reverse the role of keys and values in a dictionary

    Parameters
    ----------
    dict_var: dict

    Returns
    -------
    dict
        reversed of the enetered dictionary
    """

    return {v:k for k,v in dict_var.items()}

#############################
# Train/validation/test split
#############################

def text_train_valid_test_split(original_file_path, train_file_path, valid_file_path, 
                                test_file_path, train_idxs_path = None, valid_idxs_path = None, 
                                test_idxs_path = None, p_valid = 0.1, p_test = 0.1, file_type = 'gz', 
                                input_header = True, output_header = False):

    """ Split a text file into training, valid and test set.

    Parameters
    ----------
    original_file_path: str
        path of the text file to split
    train_file_path: str
        path of the text file where the training data will be stored
    valid_file_path: str
        path of the text file where the validation data will be stored
    test_file_path: str
        path of the text file where the test data will be stored
    train_idxs_path: str
        path of the csv file where the indices of the training data will be stored
    valid_idxs_path: str
        path of the csv file where the indices of the validation data will be stored
    test_idxs_path: str
        path of the csv file where the indices of the test data will be stored
    p_valid: float
        proportion of the data to use for the validation set 
    p_test: float
        proportion of the data to use for the test set 
    file_type: str
        accepts 'txt', 'csv' or 'gz'
    input_header: bool
        whether the input file has a header
    output_header: bool
        whether the output file should have a header (copied from the header of the input file)

    Returns
    -------
    Null (save text files)
    """

    # Calculate number of lines in each set
    orig_ind_file = IndexedFile(original_file_path, file_type)
    N_total = len(orig_ind_file) - int(input_header) # the second term is 1 if there is a heading and 0 otherwise
    N_valid = round(N_total * p_valid) # 100
    N_test = round(N_total * p_test) # 100
    N_train = N_total - N_valid - N_test # 800

    ### Generate Training/Valid/Test indices
    # All indices 
    ind_all = np.array(range(1, N_total+int(input_header))) 
    # Train indices
    np.random.seed(1)
    ind_train = np.random.choice(ind_all, size = N_train, replace = False) 
    # Remaing indices (either test or valid + test)
    ind_hold = np.setdiff1d(ind_all, ind_train)
    np.random.seed(1)
    ind_valid = np.random.choice(ind_hold, size = N_valid, replace = False)
    ind_test = np.setdiff1d(ind_hold, ind_valid)

    ### Export the train, valid and test indices
    if train_idxs_path and test_idxs_path and valid_idxs_path:
        np.savetxt(train_idxs_path, ind_train, delimiter = ",")
        np.savetxt(valid_idxs_path, ind_valid, delimiter = ",")
        np.savetxt(test_idxs_path, ind_test, delimiter = ",")

    ### Prepare the train, valid and test files
    if file_type == 'gz':
        with gzip.open(original_file_path, mode = 'rb') as f_all:
            with gzip.open(train_file_path, mode = 'wb') as f_train:
                with gzip.open(valid_file_path, mode = 'wb') as f_valid:
                    with gzip.open(test_file_path, mode = 'wb') as f_test:
                        for i, line in enumerate(f_all):
                            if i in ind_train:
                                f_train.write(line)
                            elif i in ind_valid:
                                f_valid.write(line)
                            elif i in ind_test:
                                f_test.write(line)
                            elif i == 0: 
                                if (input_header == True) and (output_header == False): # case when to not write heading
                                    pass
                                else:
                                    f_train.write(line)
                                    f_valid.write(line)
                                    f_test.write(line)
                            else:
                                print(f"Warning! The {i}th was not written: {line}")
                        
    else: # txt or csv
        with open(original_file_path, mode = 'r') as f_all:
            with open(train_file_path, mode = 'w') as f_train:
                with open(valid_file_path, mode = 'w') as f_valid:
                    with open(test_file_path, mode = 'w') as f_test:                       
                        for i, line in enumerate(f_all):
                            if i in ind_train:
                                f_train.write(line)
                            elif i in ind_valid:
                                f_valid.write(line)
                            elif i in ind_test:
                                f_test.write(line)
                            elif i == 0: # heading
                                if (input_header == True) and (output_header == False): # case when to not write heading
                                    pass
                                else:
                                    f_train.write(line)
                                    f_valid.write(line)
                                    f_test.write(line)
                            else:
                                print(f"Warning! This line was not written: {line}")

    print(f"- Number of lines in the original set is {N_total}")
    print(f"- Number of lines in the training set is {N_train}")
    print(f"- Number of lines in the validation set is {N_valid}")
    print(f"- Number of lines in the test set is {N_test}")


def df_train_valid_test_split(data, train_data_path, valid_data_path, 
                              test_data_path, train_idxs_path = None, valid_idxs_path = None, 
                              test_idxs_path = None, p_valid = 0.1, p_test = 0.1):

    """ Split data stored in a dataframe into training, valid and test set.

    Parameters
    ----------
    data: dataframe
        data to split
    train_data_path: str
        path of the text file where the training data will be stored
    valid_data_path: str
        path of the text file where the validation data will be stored
    test_data_path: str
        path of the text file where the test data will be stored
    train_idxs_path: str
        path of the csv file where the indices of the training data will be stored
    valid_idxs_path: str
        path of the csv file where the indices of the validation data will be stored
    test_idxs_path: str
        path of the csv file where the indices of the test data will be stored
    p_valid: float
        proportion of the data to use for the validation set 
    p_test: float
        proportion of the data to use for the test set 

    Returns
    -------
    Null (save dataframes as csv files)
    """

    # Calculate number of lines in each set
    N_total = len(data)
    N_valid = round(N_total * p_valid) # 100
    N_test = round(N_total * p_test) # 100
    N_train = N_total - N_valid - N_test # 800

    ### Generate Training/Valid/Test indices
    # All indices 
    ind_all = np.array(range(1, N_total)) 
    # Train indices
    np.random.seed(1)
    ind_train = np.random.choice(ind_all, size = N_train, replace = False) 
    # Remaing indices (either test or valid + test)
    ind_hold = np.setdiff1d(ind_all, ind_train)
    np.random.seed(1)
    ind_valid = np.random.choice(ind_hold, size = N_valid, replace = False)
    ind_test = np.setdiff1d(ind_hold, ind_valid)

    ### Export the train, valid and test indices
    if train_idxs_path and test_idxs_path and valid_idxs_path:
        np.savetxt(train_idxs_path, ind_train, delimiter = ",")
        np.savetxt(valid_idxs_path, ind_valid, delimiter = ",")
        np.savetxt(test_idxs_path, ind_test, delimiter = ",")

    ### Prepare the train, valid and test sets and export them
    # train
    data_train = data.iloc[ind_train, ]
    data_train.to_csv(train_data_path, sep = ',', index = False)
    del data_train 
    # valid
    data_valid = data.iloc[ind_valid, ]
    data_valid.to_csv(valid_data_path, sep = ',', index = False)
    del data_valid
    # test
    data_test = data.iloc[ind_test, ]
    data_test.to_csv(test_data_path, sep = ',', index = False)
    del data_test

    ### Wrap-up
    print(f"- Number of rows in the original set is {N_total}")
    print(f"- Number of rows in the training set is {N_train}")
    print(f"- Number of rows in the validation set is {N_valid}")
    print(f"- Number of rows in the test set is {N_test}")

################
# Create ngrams
################

def orthoCoding(sent, gram_size, remove_duplicates = False, randomize_order = False):

    """Create ngrams from a sentence.
    Parameters
    ----------
    sent: str
    gram_size: int or (int, int)
    remove_duplicates: Bool
    randomize_order  : Bool
    """
    
    # gram size could be integer or range
    try:
        min_n, max_n = gram_size
    except TypeError:
        min_n = max_n = gram_size

    tokens = sent.lower().split()

    if randomize_order:
        shuffle(tokens)

    string = u"#" + u"#".join(tokens) + u"#"
    cues = list()
    for n in range(min_n, max_n + 1):
        cues += [string[i:i + n] for i in range(len(string) - n + 1)]

    if gram_size == 1:
        cues = [x for x in cues if x != u"#"]
    if remove_duplicates:
        cues = set(cues)

    cues = u"_".join(cues)
    return cues

################
# Create epochs 
################

def shuffle_textfile(infile_path, outfile_path):

    """shuffle an event dataset that is stored as a gz file    

    Parameters
    ----------
    infile_path: str
        path to the event file to shuffle
    outfile_path: str
        path to the shuffled file

    Returns
    -------
    None 
        save a .gz file
    """

    with gzip.open(infile_path, 'rt', encoding = 'utf-8') as fr: 
        with gzip.open(outfile_path , 'wt', encoding = 'utf-8') as fw:
            lines = fr.readlines()
            lines = [lines[0]] + random.shuffle(lines[1:]) # lines[0] is the heading     
            fw.writelines(lines)

def create_epochs_textfile(infile_path, outfile_path, epoch, shuffle_epoch = False):

    """Generate epochs from a dataset of events that is stored as a gz file    

    Parameters
    ----------
    infile_path: str
        path to the event file to duplicate
    outfile_path: str
        path to the duplicatd file
    epoch: int
        number of epochs to generate
    shuffle_epoch: Boolean
        whether to shuffle the data after every epoch

    Returns
    -------
    None 
        save a .gz file
    """

    with gzip.open(infile_path, 'rt', encoding = 'utf-8') as fr: 
        with gzip.open(outfile_path , 'wt', encoding = 'utf-8') as fw:
            lines = fr.readlines()
            lines_epoch = lines
            if shuffle_epoch:
                for j in range(1, epoch):
                    lines_epoch = lines_epoch + random.shuffle(lines[1:])
            else:
                lines_epoch = [lines[0]] + (lines[1:] * epoch) # lines[0] is the heading          
            fw.writelines(lines_epoch)


import gzip
import csv
import numpy as np
import pandas as pd
import random
import re
import sys
from itertools import islice
from functools import partial
from multiprocessing import Pool
from nltk.util import ngrams
from collections.abc import Iterable
from collections import Counter
from random import shuffle
from keras.preprocessing.text import Tokenizer

### Set of allowed English characters
ENGLISH = "abcdefghijklmnopqrstuvwxyz"
ENGLISH = ENGLISH + ENGLISH.upper()

# Compiled regular expression that detects disallowed English characters
not_symbol_pattern_en = re.compile(f"[^{ENGLISH}]")

#######################################
# Conversion between csv and gz formats
#######################################

def df_to_gz(data, gz_outfile, sep_gz = '\t', encoding = 'utf-8'):

    """Export a dataframe containing events to a .gz file

    Parameters
    ----------
    data: dataframe
        dataframe to export to a gz file
    gz_outfile: str
        path of the gz file 
    sep_gz: str
        field delimiter for the gz file. Default: '\t'
    encoding: str
        file encoding to use. Default: 'utf-8'

    Returns
    -------
    None 
        save a .gz file
    """

    #with gzip.open(gz_outfile, 'wt', encoding = encoding) as out:
    data.to_csv(gz_outfile, compression = 'gzip', sep = sep_gz, index = False, encoding = encoding)

def csv_to_gz(csv_infile, gz_outfile, sep_gz = '\t', encoding = 'utf-8'):

    """Convert a csv containing events to a .gz file

    Parameters
    ----------
    csv_infile: str
        path of the csv file to convert
    gz_outfile: str
        path of the gz file 
    sep_gz: str
        field delimiter for the gz file. Default: '\t' 
    encoding: str
        file encoding to use. Default: 'utf-8'

    Returns
    -------
    None 
        save a .gz file
    """

    with gzip.open(gz_outfile, 'wt', encoding = encoding) as out:
        for line in csv.reader(open(csv_infile, 'r', encoding = encoding)):
            line = sep_gz.join(line)+'\n'
            out.write(line)

def gz_to_csv(gz_infile, csv_outfile, sep_gz = '\t', encoding = 'utf-8'):

    """Convert a gz file containing events to csv format

    Parameters
    ----------
    gz_infile: str
        path of the gz file to convert
    csv_outfile: str
        path of the csv file  
    sep_gz: str
        field delimiter for the gz file. Default: '\t' 
    encoding: str
        file encoding to use. Default: 'utf-8'

    Returns
    -------
    None 
        save a csv file
    """

    with open(csv_outfile, mode = 'w', newline = '\n', encoding = encoding) as outfile:
        with gzip.open(gz_infile, 'rt', encoding = encoding) as infile:
            csv_writer = csv.writer(outfile, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
            for line in infile:
                csv_writer.writerow(line.strip().split(sep_gz))

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
    # For the cues
    with open(cue_index_path, 'w', encoding = 'utf-8') as fc:
        for key in cue_index.keys():
            fc.write("%s,%s\n"%(key, cue_index[key]))

    # For the toutcomes
    with open(outcome_index_path, 'w', encoding = 'utf-8') as fo:
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
    with open(cue_index_path, 'w', encoding = 'utf-8') as fc:
        for key in cue_index.keys():
            fc.write("%s,%s\n"%(key, cue_index[key]))

    # For the tenses
    with open(outcome_index_path, 'w', encoding = 'utf-8') as fo:
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
    with open(index_system_path, 'r', encoding = 'utf-8') as file:
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
                                test_idxs_path = None, p_valid = 0.1, p_test = 0.1, 
                                file_type = 'gz', input_header = True, output_header = False, 
                                encoding = 'utf-8', seed = None):

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
    encoding: str
        file encoding to use. Default: 'utf-8'
    seed : int or None
        random seed to initialise the pseudorandom number generator. Use it if you want to have replicable results. 
        Default: None



    Returns
    -------
    Null (save text files)
    """

    # Calculate number of lines in each set
    orig_ind_file = IndexedFile(original_file_path, file_type)
    N_total = len(orig_ind_file) - int(input_header) # the second term is 1 if there is a heading and 0 otherwise
    N_valid = round(N_total * p_valid) 
    N_test = round(N_total * p_test) 
    N_train = N_total - N_valid - N_test 

    # Set the seed
    np.random.seed(seed)

    ### Generate Training/Valid/Test indices
    # All indices 
    ind_all = np.array(range(N_total+int(input_header))) 
    # Train indices
    ind_train = np.random.choice(ind_all, size = N_train, replace = False) 
    # Remaing indices (either test or valid + test)
    ind_hold = np.setdiff1d(ind_all, ind_train)
    ind_valid = np.random.choice(ind_hold, size = N_valid, replace = False)
    ind_test = np.setdiff1d(ind_hold, ind_valid)
    np.random.shuffle(ind_test)

    ### Export the train, valid and test indices
    if train_idxs_path and test_idxs_path and valid_idxs_path:
        np.savetxt(train_idxs_path, ind_train, delimiter = ",")
        np.savetxt(valid_idxs_path, ind_valid, delimiter = ",")
        np.savetxt(test_idxs_path, ind_test, delimiter = ",")

    ### Prepare the train, valid and test files
    if file_type == 'gz':
        with gzip.open(original_file_path, mode = 'rb', encoding = encoding) as f_all:
            with gzip.open(train_file_path, mode = 'wb', encoding = encoding) as f_train:
                with gzip.open(valid_file_path, mode = 'wb', encoding = encoding) as f_valid:
                    with gzip.open(test_file_path, mode = 'wb', encoding = encoding) as f_test:
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
        with open(original_file_path, mode = 'r', encoding = encoding) as f_all:
            with open(train_file_path, mode = 'w', encoding = encoding) as f_train:
                with open(valid_file_path, mode = 'w', encoding = encoding) as f_valid:
                    with open(test_file_path, mode = 'w', encoding = encoding) as f_test:                       
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
                              test_idxs_path = None, p_valid = 0.1, p_test = 0.1, seed = None):

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
    seed : int or None
        random seed to initialise the pseudorandom number generator. Use it if you want to have replicable results. 
        Default: None

    Returns
    -------
    Null (save dataframes as csv files)
    """

    # Calculate number of lines in each set
    N_total = len(data)
    N_valid = round(N_total * p_valid) 
    N_test = round(N_total * p_test) 
    N_train = N_total - N_valid - N_test 

    # Set the seed
    np.random.seed(seed)

    ### Generate Training/Valid/Test indices
    # All indices 
    ind_all = np.array(range(N_total)) 
    # Train indices
    ind_train = np.random.choice(ind_all, size = N_train, replace = False) 
    # Remaing indices (either test or valid + test)
    ind_hold = np.setdiff1d(ind_all, ind_train)
    ind_valid = np.random.choice(ind_hold, size = N_valid, replace = False)
    ind_test = np.setdiff1d(ind_hold, ind_valid)
    np.random.shuffle(ind_test)

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

#####################
# Create event files
#####################

def uniquify_list(seq): 

    """Uniquifies a list while preserving the original order of the list

    Parameters
    ----------
    seq : list
        List to uniquify 
    
    Returns:
    --------
    list
        list after removing duplicates while preserving the order of the items
    """

    seen = {}
    result = []
    for item in seq:
        if item in seen: 
            continue
        seen[item] = 1
        result.append(item)
    return result

def process_line(line, lowercase = True, not_symbol_pattern = not_symbol_pattern_en, remove_weird_words = False):

    """Splits line into lowercase words based on spaces and non-allowed characters (Carful when you have hyphens or contractions that you want to keep). 
    Ex: We've had my mother-in-law in the house -> ['we','ve','had','my','mother','in','law','in','the','house'])

    Parameters
    ----------
    line: str
    lowercase: Bool
    not_symbol_pattern: compiled regular expression
    remove_weird_words: Bool
        Whether to remove words containing non-allowed characters (True) or split on each non-allowed character (False). 
        Default: False

    Returns:
    --------
    list
        list of words/tokens in line
    """

    if remove_weird_words:
        words0 = line.strip().lower().split(" ")
        words = [w for w in words0 if (not not_symbol_pattern.search(w) and w != "")]
    else:
        line = not_symbol_pattern.sub(" ", line.strip().lower())
        words = [w for w in line.split(" ") if w != ""]
    return words

def extract_word_ngrams(line, ngram_size = 1, not_symbol_pattern = not_symbol_pattern_en, 
                        remove_weird_words = True, sep_words = "#", sep_ngrams = '_', 
                        lowercase = True, remove_duplicates = False, randomise_order = False):

    """Generate word ngram cues from a text line. 
    
    Parameters
    ----------
    line : str
        String from which to extract the ngrams 
    ngram_size : int or (int, int)
        If ngram_size an integer, then only ngrams of size 'ngram_size' are created. 
        If ngram_size a tuple (a, b), then ngrams of size between a and b are created. 
        Default: 1
    not_symbol_pattern : compiled regular expression
        Regular expression that matches disallowed characters (e.g. punctuation)
    remove_weird_words: Bool
        Whether to remove words containing non-allowed characters (True) or split on each non-allowed character (False). 
        Default: False
    sep_words: str 
        Symbole used to seperate the words in a single ngram. Default: "#"
    sep_ngrams: str 
        Symbole used to seperate the ngrams. Default: '_'
    lowercase: Bool
    remove_duplicates: Bool
    randomize_order  : Bool    
    
    Returns:
    --------
    str
        word n-grams seperated using sep_ngrams
    """

    # check if ngram size is an integer or range
    if isinstance(ngram_size, int):
        n_min = n_max = ngram_size
    elif len(ngram_size) == 2 and all(isinstance(j, int) for j in ngram_size):
        n_min, n_max = ngram_size
    else:
        raise ValueError("ngram_size value should be an integer or a range of integers")

    # Clean and tokenise the string
    words = process_line(line, 
                         lowercase = lowercase, 
                         not_symbol_pattern = not_symbol_pattern, 
                         remove_weird_words = remove_weird_words)

    line_ngrams = []
    for n in range(n_min, n_max + 1):
        line_ngrams.extend(list(ngrams(words, n)))

    # Remove duplicated ngrams if asked for
    if remove_duplicates:
        if randomise_order: # randomise as well if asked for (set changes the order automatically)  
            line_ngrams = list(set(line_ngrams)) # Remove duplicated ngrams 
        else: # use list_unique() to preserve cue order 
            line_ngrams = uniquify_list(line_ngrams)
    else:        
        if randomise_order: # Shuffle the ngrams
            shuffle(line_ngrams)

    return sep_ngrams.join([sep_words.join(ngram) for ngram in line_ngrams])

# # Test
# line0 = "We've had my mother-in-law in the ho1use "
# line0 = "We've had my mother-in-law "
# process_line(line0, remove_weird_words= False)
# extract_word_ngrams(line0, ngram_size = 2, remove_weird_words= False)

def extract_letter_ngrams(line, ngram_size = 1, not_symbol_pattern = not_symbol_pattern_en, 
                          remove_weird_words = False, sep_ngrams = '_', mark_word_boundary = True, 
                          lowercase = True, remove_duplicates = False, randomise_order = False):

    """Generate letter ngrams from a text line.

    Parameters
    ----------
    line: str
    gram_size: int or (int, int)
    remove_weird_words: Bool
        Whether to remove words containing non-allowed characters (True) or split on each non-allowed character (False). 
        Default: False
    mark_word_boundary: Bool
    lowercase: Bool
    remove_duplicates: Bool
    randomize_order  : Bool 

    Returns:
    --------
    str
        letter n-grams (n-graphs) seperated using sep_ngrams
    """
    
    # check if ngram size is an integer or range
    if isinstance(ngram_size, int):
        n_min = n_max = ngram_size
    elif len(ngram_size) == 2 and all(isinstance(j, int) for j in ngram_size):
        n_min, n_max = ngram_size
    else:
        raise ValueError("ngram_size value should be an integer or a range of integers")

    # Clean and tokenise the string
    words = process_line(line, 
                         lowercase = lowercase, 
                         not_symbol_pattern = not_symbol_pattern, 
                         remove_weird_words = remove_weird_words)

    # Initialise the list the will contain the letter n-grams
    line_ngrams = list()

    if mark_word_boundary:
        line_processed = "#" + "#".join(words) + "#"
        for n in range(n_min, n_max + 1):
            line_ngrams.extend([line_processed[i:i + n] for i in range(len(line_processed) - n + 1)])
        # Remove word boundaries (#) if uni-grams requested 
        if n_min == 1:
            line_ngrams = [ngram for ngram in line_ngrams if ngram != "#"]
    else:
        for word in words:
            for n in range(n_min, n_max + 1):
                line_ngrams.extend([word[i:i + n] for i in range(len(word) - n + 1)])

    # Remove duplicated ngrams if asked for
    if remove_duplicates:
        if randomise_order: # randomise as well if asked for (set changes the order automatically)  
            line_ngrams = list(set(line_ngrams)) # Remove duplicated ngrams 
        else: # use list_unique() to preserve ngram order 
            line_ngrams = uniquify_list(line_ngrams)
    else:        
        if randomise_order: # Shuffle the ngrams
            shuffle(line_ngrams)

    #ngrams = "_".join(ngrams)

    return sep_ngrams.join(line_ngrams)

# # Test
# line0 = "We've had my mother-in-law "
# process_line(line0)
# extract_letter_ngrams(line0, ngram_size = 2)

def extract_cues(line, ngram_base = 'words', ngram_size = 1, not_symbol_pattern = not_symbol_pattern_en, 
                 remove_weird_words = False, sep_words = "#", sep_ngrams = '_', mark_word_boundary = True, 
                 lowercase = True, remove_duplicates = False, randomise_order = False):

    """Prepare cues (word or letter n-grams) based on a text line.

    Parameters
    ----------
    line : str
        String from which to extract the ngrams 
    ngram_base: str
        If ngram_base = 'words', then word n-gram cues are used
        If ngram_base = 'letters', then letter n-gram cues are used
    ngram_size : int or (int, int)
        If ngram_size an integer, then only ngrams of size 'ngram_size' are created. 
        If ngram_size a tuple (a, b), then ngrams of size between a and b are created. 
        Default: 1
    not_symbol_pattern : compiled regular expression
        Regular expression that matches disallowed characters (e.g. punctuation)
    remove_weird_words: Bool
        Whether to remove words containing non-allowed characters (True) or split on each non-allowed character (False). 
        Default: False
    sep_words: str 
        Symbole used to seperate the words in a single ngram (relevant only if ngram_base == 'words'). Default: "#"
    sep_ngrams: str 
        Symbole used to seperate the ngrams. Default: '_'
    mark_word_boundary: Bool 
        (relevant only if ngram_base == 'words')
    lowercase: Bool
    remove_duplicates: Bool
    randomize_order  : Bool    
    
    Returns:
    --------
    str
        n-gram cues seperated with sep_ngrams
    """

    # Check cue type
    if ngram_base == 'words':
        cues = extract_word_ngrams(line = line,
                                   ngram_size = ngram_size, 
                                   not_symbol_pattern = not_symbol_pattern, 
                                   remove_weird_words = remove_weird_words, 
                                   sep_words = sep_words,
                                   sep_ngrams = sep_ngrams,
                                   lowercase = lowercase,
                                   remove_duplicates = remove_duplicates, 
                                   randomise_order = randomise_order)
    elif ngram_base == 'letters':
        cues = extract_letter_ngrams(line = line, 
                                     ngram_size = ngram_size, 
                                     not_symbol_pattern = not_symbol_pattern, 
                                     remove_weird_words = remove_weird_words, 
                                     sep_ngrams = sep_ngrams, 
                                     mark_word_boundary = mark_word_boundary, 
                                     lowercase = lowercase, 
                                     remove_duplicates = remove_duplicates, 
                                     randomise_order = randomise_order)
    else:
        raise ValueError("ngram_base should be either 'words' or 'letters'")

    # outcomes_list = process_line(line = line, 
    #                         lowercase = lowercase, 
    #                         not_symbol_pattern = not_symbol_pattern, 
    #                         remove_weird_words = remove_weird_words) 
    # outcomes = "_".join(outcomes_list)

    return cues

# # Test
# line0 = "We've had my mother-in-law "
# process_line(line0)
# extract_cues(line0, ngram_base = 'letters', ngram_size = (1,2))

def extract_events(line, ngram_base = 'words', ngram_size = 1, outcomes_provided = True, 
                   not_symbol_pattern = not_symbol_pattern_en, remove_weird_words = False, 
                   sep_words = "#", sep_ngrams = '_', mark_word_boundary = True, 
                   lowercase = True, remove_duplicates = False, randomise_order = False):

    """Prepare events (cues + outcomes) based on a text line. Outcomes are either provided or 
       generated as words in the line

    Parameters
    ----------
    line : str
        String from which to extract the ngram cues and outcomes. If the outcomes are provided,
        then they should be seperated with sep_ngrams (usually '_'). input text from which the 
        cues are extracted and the outcomes should be seperated with a tab (e.g. 
        'she went back home\tpast.simple' ) 
    ngram_base: str
        If ngram_base = 'words', then word n-gram cues are used
        If ngram_base = 'letters', then letter n-gram cues are used
    ngram_size : int or (int, int)
        If ngram_size an integer, then only ngrams of size 'ngram_size' are created. 
        If ngram_size a tuple (a, b), then ngrams of size between a and b are created. 
        Default: 1
    outcomes_provided: Bool
        If True, outcomes are copied to make the event 
        If False, outcomes are the words that appear in the line 
    not_symbol_pattern : compiled regular expression
        Regular expression that matches disallowed characters (e.g. punctuation)
    remove_weird_words: Bool
        Whether to remove words containing non-allowed characters (True) or split on each non-allowed character (False). 
        Default: False
    sep_words: str 
        Symbole used to seperate the words in a single ngram (relevant only if ngram_base == 'words'). Default: "#"
    sep_ngrams: str 
        Symbole used to seperate the ngrams. Default: '_'
    mark_word_boundary: Bool 
        (relevant only if ngram_base == 'words')
    lowercase: Bool
    remove_duplicates: Bool
    randomize_order  : Bool    
    
    Returns:
    --------
    tuple
        pair of seq of cues and outcomes, each seperated with sep_ngrams
    """

    # Check whether outcomes are provided or not
    if outcomes_provided:
        try:
            cue_input, outcomes = line.split('\t')
        except:
            raise ValueError("Text line doesn't contain 2 enteries for cue input and outcomes")
    else:
        cue_input = line
        outcomes_list = process_line(line = line, 
                                     lowercase = lowercase, 
                                     not_symbol_pattern = not_symbol_pattern, 
                                     remove_weird_words = remove_weird_words) 
        outcomes = "_".join(outcomes_list)

    cues = extract_cues(line = cue_input,
                        ngram_base = ngram_base, 
                        ngram_size = ngram_size,
                        not_symbol_pattern = not_symbol_pattern, 
                        remove_weird_words = remove_weird_words, 
                        sep_words = sep_words,
                        sep_ngrams = sep_ngrams,
                        mark_word_boundary = mark_word_boundary,
                        lowercase = lowercase,
                        remove_duplicates = remove_duplicates, 
                        randomise_order = randomise_order)

    return (cues, outcomes)

# # Test 1
# line0 = "she went back home\tpast.simple"
# extract_events(line0, ngram_base = 'letters', ngram_size = (1,2), outcomes_provided = True)

# # Test 2
# line1 = "she went back home"
# extract_events(line1, ngram_base = 'letters', ngram_size = (1,2), outcomes_provided = False)

def chunk(iterable, chunksize):
    
    """Returns lazy iterator that yields chunks from iterable.
    """

    iterator = iter(iterable)
    return iter(lambda: list(islice(iterator, chunksize)), [])

def prepare_events_generator(lines, ngram_base, ngram_size, outcomes_provided, 
                             not_symbol_pattern, remove_weird_words, sep_words, 
                             sep_ngrams, mark_word_boundary, lowercase, 
                             remove_duplicates, randomise_order, num_threads, 
                             chunksize):

    """Creates a generator that prepare events in parallel from a sequence of lines

    Parameters
    ----------
    lines : str
        Lines from which to extract the ngram cues and outcomes. If the outcomes are provided,
        then they should be seperated with sep_ngrams (usually '_'). input text from which the 
        cues are extracted and the outcomes should be seperated with a tab (e.g. 
        'she went back home\tpast.simple' ) 
    ngram_base: str
        If ngram_base = 'words', then word n-gram cues are used
        If ngram_base = 'letters', then letter n-gram cues are used
    ngram_size : int or (int, int)
        If ngram_size an integer, then only ngrams of size 'ngram_size' are created. 
        If ngram_size a tuple (a, b), then ngrams of size between a and b are created. 
        Default: 1
    outcomes_provided: Bool
        If True, outcomes are copied to make the event 
        If False, outcomes are the words that appear in the line 
    not_symbol_pattern : compiled regular expression
        Regular expression that matches disallowed characters (e.g. punctuation)
    remove_weird_words: Bool
        Whether to remove words containing non-allowed characters (True) or split on each non-allowed character (False). 
        Default: False
    sep_words: str 
        Symbole used to seperate the words in a single ngram (relevant only if ngram_base == 'words'). Default: "#"
    sep_ngrams: str 
        Symbole used to seperate the ngrams. Default: '_'
    mark_word_boundary: Bool 
        (relevant only if ngram_base == 'words')
    lowercase: Bool
    remove_duplicates: Bool
    randomize_order  : Bool 
    num_threads : int
        Number of parallel processes to use (should be <= number of threads)
    chunksize : int
        Number of lines each process will work on in batches
        (Higher values increase memory consumption, but decrease processing
        time, with diminishing returns)   
    """

    # Fills arguments for later use with .imap
    _extract_events = partial(extract_events,
                              ngram_base = ngram_base, 
                              ngram_size = ngram_size, 
                              outcomes_provided = outcomes_provided, 
                              not_symbol_pattern = not_symbol_pattern, 
                              remove_weird_words = remove_weird_words, 
                              sep_words = sep_words, 
                              sep_ngrams = sep_ngrams, 
                              mark_word_boundary = mark_word_boundary, 
                              lowercase = lowercase, 
                              remove_duplicates = remove_duplicates, 
                              randomise_order = randomise_order)

    with Pool(num_threads) as pool:
        for _chunk in chunk(lines, chunksize * num_threads):
            yield from pool.imap(_extract_events, _chunk, # imap gives the results one at a time (# from map)
                                 chunksize = chunksize)

def prepare_event_file(data, header = False, event_file_path = None, 
                       ngram_base = 'words', ngram_size = 1, 
                       outcomes_provided = True, 
                       not_symbol_pattern = not_symbol_pattern_en,
                       remove_weird_words = False, sep_words = "#", 
                       sep_ngrams = '_', mark_word_boundary = True, 
                       lowercase = True, remove_duplicates = False, 
                       randomise_order = False, num_threads = 1, 
                       chunksize = 100000, verbose = 0):

    """Prepare event file from a text file or dataframe.

    Parameters
    ----------
    data : dataframe or str
        dataframe or path to the txt file containing textual data to extract events from.
        If it is a dataframe, it should contains two columns: the first is the input text to
        be transformed into cues and the second column should contain the sequence of outcomes
    header: Bool
        Relevant only if data is a path. 
        - if header = True, then the the file has a header to be skipped.
        - if header = False, then the the file doesn't have a header.
    event_file_path : str
        Relevant only if data is a path. In such as case, it is the path of the event file 
        that will be prepare if data is a path
    ngram_base: str
        If ngram_base = 'words', then word n-gram cues are used
        If ngram_base = 'letters', then letter n-gram cues are used
    ngram_size : int or (int, int)
        If ngram_size an integer, then only ngrams of size 'ngram_size' are created. 
        If ngram_size a tuple (a, b), then ngrams of size between a and b are created. 
        Default: 1
    not_symbol_pattern : compiled regular expression
        Regular expression that matches disallowed characters (e.g. punctuation)
    remove_weird_words: Bool
        Whether to remove words containing non-allowed characters (True) or split on each non-allowed character (False). 
        Default: False
    sep_words: str 
        Symbole used to seperate the words in a single ngram (relevant only if ngram_base == 'words'). Default: "#"
    sep_ngrams: str 
        Symbole used to seperate the ngrams. Default: '_'
    mark_word_boundary: Bool 
        (relevant only if ngram_base == 'words')
    lowercase: Bool
    remove_duplicates: Bool
    randomize_order  : Bool    
    num_threads: int
        maximum number of processes to use - it should be >= 1. Default: 1
    chunksize : int
        Number of lines each process will work on in batches
        (Higher values increase memory consumption, but decrease processing
        time, with diminishing returns) 
    
    Returns:
    --------
    None
        save a .gz event file
    """

    # Check data type
    if isinstance(data, str): # If the input data is a text file

        with gzip.open(event_file_path, "wt", encoding='utf-8') as outfile:  
            outfile.write("cues\toutcomes\n")
            with open(data, encoding='utf-8') as infile:
                if header:
                    next(infile) # skip header
                results = prepare_events_generator(lines = infile, 
                                                ngram_base = ngram_base, 
                                                ngram_size = ngram_size, 
                                                outcomes_provided = outcomes_provided, 
                                                not_symbol_pattern = not_symbol_pattern, 
                                                remove_weird_words = remove_weird_words, 
                                                sep_words = sep_words,
                                                sep_ngrams = sep_ngrams,
                                                mark_word_boundary = mark_word_boundary, 
                                                lowercase = lowercase, 
                                                remove_duplicates = remove_duplicates, 
                                                randomise_order = randomise_order, 
                                                num_threads = num_threads, 
                                                chunksize = chunksize)
                #list(results)
                for i, event in enumerate(results):
                    cues, outcomes = event
                    outfile.write(f"{cues}\t{outcomes}\n")   
                    if verbose == 1 and i % 100000 == 0:
                        sys.stdout.write('.')
                        sys.stdout.flush()

    elif isinstance(data, pd.DataFrame): # If the input data is a dataframe
        raise NotImplementedError('data as a dataframe is not implemented yet.') 
    else:
        raise ValueError("data should be either a path to a txt file or a dataframe")


################
# Create epochs 
################

def shuffle_textfile(infile_path, outfile_path, seed = None):

    """shuffle an event dataset that is stored as a gz file    

    Parameters
    ----------
    infile_path: str
        path to the event file to shuffle
    outfile_path: str
        path to the shuffled file
    seed : int or None
        random seed to initialise the pseudorandom number generator. Use it if you want to have replicable results. 
        Default: None

    Returns
    -------
    None 
        save a .gz file
    """

    with gzip.open(infile_path, 'rt', encoding = 'utf-8') as fr: 
        with gzip.open(outfile_path , 'wt', encoding = 'utf-8') as fw:
            lines = fr.readlines()
            header = [lines[0]] # lines[0] is the heading   
            body = lines[1:]
            random.Random(seed).shuffle(body)
            lines = header + body   
            fw.writelines(lines)

def create_epochs_textfile(infile_path, outfile_path, epoch, shuffle_epoch = False, seed = None):

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
    seed : int or None
        random seed to initialise the pseudorandom number generator. Use it if you want to have replicable results. 
        Default: None

    Returns
    -------
    None 
        save a .gz file
    """

    with gzip.open(infile_path, 'rt', encoding = 'utf-8') as fr: 
        with gzip.open(outfile_path , 'wt', encoding = 'utf-8') as fw:
            lines = fr.readlines()
            body = lines[1:]
            lines_epoch = lines.copy()
            if shuffle_epoch:
                for j in range(1, epoch):           
                    random.Random(seed).shuffle(body)
                    lines_epoch = lines_epoch + body 
            else:
                lines_epoch = [lines[0]] + (lines[1:] * epoch) # lines[0] is the heading          
            fw.writelines(lines_epoch)

#############
# Embeddings
#############

def extract_embedding_dim(embedding_input):

    """ Extract the dimension of the embedding vectors from and embedding file or numpy matrix

    Parameters
    ----------
    embedding_input: str or numpy array
        numpy matrix or txt file that contains the embedding vectors
    """

    ### Extract embedding dimension
    if isinstance(embedding_input, np.ndarray):
        embedding_dim = np.size(embedding_input, 1)
    elif isinstance(embedding_input, str): # if pre-trained embedding provided
        with open(embedding_input) as f:
            for ii, line in enumerate(f):
                if ii == 0: # skip the first line just in case the file has a heading
                    continue
                if ii == 1: 
                    word, coefs = line.split(maxsplit=1)
                    coefs = np.fromstring(coefs, 'f', sep=' ')
                    embedding_dim = coefs.size
                else:
                    break

    return embedding_dim

def prepare_embedding_matrix(embedding_input, cue_index):

    """ Prepare the embedding matrix to use with Keras from a text file or from available gensim embeddings

    Parameters
    ----------
    embedding_input: str
        for now only accepts path to a txt file that contains the embedding vectors
    cue_index: dict
        mapping from cues to indices. The dictionary should include only the cues to be used for modelling
    """

    # Number of words in the index system
    N_cues = len(cue_index)

    # Extract dimension of the embedding vectors
    embedding_dim = extract_embedding_dim(embedding_input = embedding_input)

    # Create the embedding index system, that is, a dictionary that maps words to their embedding vectors.
    # Restrict to words that exist in the index system 
    embeddings_index = {}
    with open(embedding_input) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            if word in cue_index and len(embeddings_index) < N_cues:
                coefs = np.fromstring(coefs, 'f', sep=' ')
                embeddings_index[word] = coefs
            elif word not in cue_index and len(embeddings_index) < N_cues:
                continue
            else:
                break
            
    # Preparing the word embedding matrix for training
    embedding_mat = np.zeros((N_cues+1, embedding_dim)) #] vectors for Words that do not appear in the embedding file will be set to 0
    for word, i in cue_index.items():
        embedding_vec = embeddings_index.get(word)
        if embedding_vec is not None:
            embedding_mat[i-1] = embedding_vec

    return embedding_mat
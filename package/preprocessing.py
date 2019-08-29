import gzip
import csv
import numpy as np
from collections.abc import Iterable
from collections import Counter
from random import shuffle

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
    corpus_file = IndexedFile(CORPUS_path, 'gz)
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


###################
# Train/test split
###################

def text_train_valid_test_split(original_file_path, train_file_path, valid_file_path, 
                                test_file_path, train_idxs_path = None, valid_idxs_path = None, 
                                test_idxs_path = None, p_valid = 0.1, p_test = 0.1, file_type = 'gz', 
                                input_heading = True, output_heading = False):

    """ Split an indexed text file into training, valid and test set.

    Parameters
    ----------
    original_file_path: str
    train_file_path: str
    valid_file_path: str
    test_file_path: str
    train_idxs_path: str
    valid_idxs_path: str
    test_idxs_path: str
    p_valid: float
    p_test: float
    file_type: str
    input_heading: bool
    output_heading: bool

    Returns
    -------
    Null (save text files)
    """

    # Calculate number of lines in each set
    orig_ind_file = IndexedFile(original_file_path, file_type)
    N_total = len(orig_ind_file) - int(input_heading) # the second term is 1 if there is a heading and 0 otherwise
    N_valid = round(N_total * p_valid) # 100
    N_test = round(N_total * p_test) # 100
    N_train = N_total - N_valid - N_test # 800

    ### Generate Training/Valid/Test indices
    # All indices 
    ind_all = np.array(range(1, N_total+int(input_heading))) 
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
                                if (input_heading == True) and (output_heading == False): # case when to not write heading
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
                                if (input_heading == True) and (output_heading == False): # case when to not write heading
                                    pass
                                else:
                                    f_train.write(line)
                                    f_valid.write(line)
                                    f_test.write(line)
                            else:
                                print(f"Warning! This line was not written: {line}")

    print(f"- The number of lines in the original set is {N_total}")
    print(f"- The number of lines in the training set is {N_train}")
    print(f"- The number of lines in the validation set is {N_valid}")
    print(f"- The number of lines in the test set is {N_test}")

################################################
# Create index systems for the cues and outcomes
################################################

def create_index_systems(cue_counter, outcome_counter, cue_index_path, outcome_index_path):

    """Save index sytems for cues and outcomes

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

def create_epochs():
    pass


# Deep text modelling
A Python package providing tools for processing and modelling text data

## Overview

The main purpose of this package is to make the process of training language learning models using large corpora easier. It offers tools for preparing the data for modelling (e.g. tokenisation, training/validation split), running biologically inspired and deep learning models (Widroff-Hoff, feedforward neural networks and LSTM), parameter tuning, as well as evaluating those models (e.g. learning curves, performance scores, evaluation on a test set).  

## Key features

The usual workflow for fitting a model to the data is:

1) **Preprocessing**: Prepare the data for modelling (e.g. training/validation/test split).
2) **Modelling**: Build a model for the data after selecting the model parameters to use.  
3) **Evaluation**: Assess the selected model (e.g. test accuracy, comparing with behavioural data).

The Deep text modelling (DTM) package provides tools to facilitate each of these steps.

### Preprocessing

All the models contained in the package work with the same unified data format. Most of the transformations needed for the different types of models are done in the background, so the user can experiment with the models quickly and with a minimal amount of coding. More specifically, the data to model should have two columns and would look like this: 

cues | outcomes
---- | -------
this_major_prestigious_event_currently_by_northern_arts | PresentProgressive
they_later_today | FutureSimple
the_surprise_total | PastSimple

- The **cues** column provides the tokens or units that will be used to predict the labels (e.g. words excluding the main verb in a sentence, as in the example above). The cues could be letters, trigraphs, words or ngrams; all separeted with underscores. 

- The **outcomes** column contains the labels that are supposed to be triggered by the cues that appear in the same row (e.g. tense of the main verb). If there are multiple outcomes, they should be seperated with underscores as is the case for the cues. 

- Each row is called an **event** (i.e. a combination of a set of cues and a set of outcomes).

Note that this 'cues-outcomes' or 'event' format, adopted from the associative learning literature, allows to model a wide range of language problems ranging from generating prerdictions for machine learning problems to explaning behavioural phenomenons (e.g., response time or participants' choices) as will be illustrated in the 'examples' section below. For standard calssification problems (e.g. deciding whether an email is a spam or non-spam), it is easy to get to the event format by replacing the spaces between the words with underscores. For more complex cases like extracting events from a corpus (e.g. generating cues and outcomes from each sentence), you can use the excellent [pyndl](https://pyndl.readthedocs.io/en/latest/index.html) package.  

The preprocessing step will generally consists of:

1) Creating index systems for the cues and outcomes (i.e. mapping between the tokens and integer indices), which are necessary for running the deep learning models. 

2) Spliting the data into training, validation and test sets from dataframes or from text files.

In addition, the package makes it possible to work with dataframes loaded in the workspace or directly with data files through their paths. It also has a tool that allows you to handle text files as you would handle indexed objects, that is, access any element through indices (e.g. `corpus[1]` would output the 2nd line of the corpus) 

### Modelling

The package provide functions to train feedforward neural networks and LSTMs based on Keras. Although, [Keras](https://keras.io/) is a simple framework to learn in comparison with the other major deep learning framworks, it is also a general-purpose package that was designed to work with different types of problems. By focusing on a specific type of problems (language learning and processing), we were able to create wrapper functions for training keras models that reduce the amount of coding and necessary learning. The package also offers tools for tunning the parameters and evaluating the naive discriminative model (NDL) provided in [pyndl](https://pyndl.readthedocs.io/en/latest/index.html). 

The modelling tools allows you to either (1) run a quick model with, for example, the default parameters; or (2) search for good (or the 'best') parameters for the model in order to fit the data well. 

### Evaluation

To be completed...

## Examples

### Binary classification (predicting gender from name)

[Report](https://nbviewer.jupyter.org/github/Adnane017/Deep_text_modelling/blob/master/illustrative_examples/names/names.ipynb)

### Multiclass classification (predicting tense from context)

[Report]

### Multiclass multilabel classification (predicting words from orthographic cues)

[Report]

## Installation

All you need to start using the package is to copy the folder inside 'package' in your computer and make it as your working directory in Python. You will also need to install the following packages:

- numpy
- pandas
- pyndl
- keras
- talos (optional)

## Authors

Adnane Ez-zizi

# Deep text modelling (DTM)
A Python package providing tools for processing and modelling text data

## Overview

The main purpose of this package is to make the process of training language learning models using large corpora easier. It offers tools for preparing the data for modelling (e.g. tokenisation, training/validation split), running biologically inspired and deep learning models (Widroff-Hoff, feedforward neural networks and LSTM), parameter tuning, as well as evaluating those models (e.g. learning curves, performance scores, evaluation on a test set).  

## Key features

The usual workflow for fitting a model to the data is:

1) *Preprocessing*: prepare the data for modelling (e.g. training/validation/test split).
2) *Modelling*: build a model for the data after selecting the model parameters to use.  
3) *Evaluation*: assess the selected model (e.g. test accuracy, comparison with behavioural data).

The Deep text modelling (DTM) package provides tools to facilitate each of these steps.

### Preprocessing

All the models contained in the package work with the same unified data format. Most of the transformations needed for the different types of models are done in the background, so the user can experiment with the models quickly and with a minimal amount of coding. More specifically, the data to model should have two columns and would look like this: 

cues | outcomes
---- | -------
this_major_prestigious_event_currently_by_northern_arts | PresentProgressive
they_later_today | FutureSimple
the_surprise_total | PastSimple

- The **cues** column provides the tokens or units that will be used to predict the labels (e.g. words excluding the main verb in a sentence, as in the example above). The cues could be letters, trigraphs, words or ngrams; all separated with underscores. 

- The **outcomes** column contains the labels that are supposed to be triggered by the cues that appear in the same row (e.g. tense of the main verb). If there are multiple outcomes, they should be separated with underscores as is the case for the cues. 

- Each row is called an **event** (i.e. a combination of a set of cues and a set of outcomes).

Note that this 'cues-outcomes' or 'event' format, adopted from the associative learning literature, allows to model a wide range of language problems ranging from generating predictions for machine learning problems to explaining behavioural phenomenons (e.g., response time or participants' choices) as will be illustrated in the 'examples' section below. For standard classification problems (e.g. deciding whether an email is a spam or non-spam), it is easy to get to the event format by replacing the spaces between the words with underscores. For more complex cases like extracting events from a corpus (e.g. generating cues and outcomes from each sentence), you can use the excellent [pyndl](https://pyndl.readthedocs.io/en/latest/index.html) package.  

The preprocessing step will generally consist of:

1) Creating index systems for the cues and outcomes (i.e. mapping between the tokens and integer indices), which are necessary for running the deep learning models. 

2) Splitting the data into training, validation and test sets from dataframes or from text files.

In addition, the package makes it possible to work with dataframes loaded in the workspace or directly with data files through their paths. It also has a tool that allows you to handle text files as you would handle indexed objects, that is, to access any element through indices (e.g. `corpus[1]` would output the 2nd line of the corpus) 

### Modelling

The package provides functions to train feedforward neural networks (FNNs) and LSTMs based on Keras. Although [Keras](https://keras.io/) is a simple framework to learn in comparison with the other major deep learning frameworks, it is also a general-purpose package that was designed to work with different types of problems. By focusing on a specific type of problems (language learning and processing), we were able to create wrapper functions for training keras models that reduce the amount of coding and necessary learning. The package also offers tools for tunning the parameters and evaluating the naive discriminative model (NDL) provided in [pyndl](https://pyndl.readthedocs.io/en/latest/index.html). 

The modelling tools allow you to either (1) run a quick model with, for example, the default parameters; or (2) search for good (or the 'best') parameters for the model in order to fit the data well. 

All models can predict the probability of each possible outcome given a certain set of cues. For FNN and LSTM, these are provided directly as the main output of the model from the activation function of the output layer. For NDL, a function is provided in the package to generate these from the main output of the model. These probabilities can then be used either to generate predictions about the presence or not of each outcome (for those with probability > 0.5) or to explain behavioural metrics like reaction time or the probability distribution of choices among the subjects.  

### Evaluation

DTM contains some useful functions for assessing your models. These include:

- Plotting learning curves. These, basically, displays the performance of your model as a function of training time. Monitoring the training and validation learning curves is useful to detect when the model starts to overfit, and hence training needs to be stopped.  
- Generating predictions from the model based on the test data, which can be then used to compute performance scores (e.g. accuracy or precision/recall) on the test data. 
- Extracting the top predicted outcomes, or in other words, the outcomes corresponding to the top *N* predicted probabilities.

## Examples

### Binary classification (predicting gender from name)

This is a very simple example that shows how to build FNN, LSTM and NDL models, and compare them on the problem of predicting the gender of a person (outcome) based on the letters in his or her name (cues). In this case, only one of two outcomes is possible for each event. [[notebook]](https://nbviewer.jupyter.org/gist/Adnane017/b4f66f33b248653808868345a0612434)

### Multiclass classification (predicting English tense from context)

Here, we show how to apply the models to the problem of predicting English tense (outcome) from the words surrounding the main verb in a sentence (cues). In this case, only one outcome out of seven is possible for each event. [[notebook]](https://nbviewer.jupyter.org/gist/Adnane017/dbdc2659b4b53756aab209237e2f407e)

### Multiclass multilabel classification (predicting words from orthographic cues)

This example illustrates how to estimate for each sentence, the probabilities of its constituent words (outcomes) based on trigraphs (i.e. three-letter sequences extracted from the words; cues). Here, multiple outcomes are possible for each event. [[notebook]](https://nbviewer.jupyter.org/github/Adnane017/Deep_text_modelling/blob/master/illustrative_examples/names/names.ipynb)

## Installation

All you need to start using the package is to copy the folder inside 'package' in your computer and make it as your working directory in Python. You will also need to install the following packages:

- numpy
- pandas
- keras
- pyndl
- scikit-learn 
- talos (optional)

## Authors

Dr Adnane Ez-zizi (a.ez-zizi@bham.ac.uk) \
Dr Petar Milin (p.milin@bham.ac.uk)

## Funder acknowledgement 

This project was supported by a Leverhulme Trust Research Leadership Award (RL-2016-001) to Prof Dagmar Divjak.

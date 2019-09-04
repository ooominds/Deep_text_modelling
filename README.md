# Deep text modelling
A Python package providing tools for processing and modelling text data

## Overview

The purpose of this package is to make the process of training language learning models using large coprora easier. It offers tools for preparing the data for modelling (e.g. tokenisation, training/validation split), running biologically inspired and deep learning models (Widroff-Hoff, feedforward neural networks and LSTM), parameter tuning, as well as evaluating those models (e.g. learning curves, performance scores, evaluation on a test set).  

## Key features

### Preprocessing

Before starting to build models, we need to first get the data in the right format. We specifically expect data to have two columns: (1) a column providing cues seperated with underscores. These could be letters, trigraphs, words or ngrams; (2) a column containg outcomes, which are supposed to be triggered by the cues that appear in the same row. To illustrate, here is an example of such data format:

cues | outcomes
---- | -------
this_major_prestigious_event_currently_by_northern_arts | PresentProgressive
they_later_today | FutureSimple
the_surprise_total | PastSimple

Note that this format allows to model a wide range of problems ranging from generating prerdictions for machine learning problems to explaning behavioural phenomenons (e.g., response time or participants' choices) as will be illustrated in the 'examples' section below. 

### Modelling

To be completed...

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
- talos

## Authors

Adnane Ez-zizi

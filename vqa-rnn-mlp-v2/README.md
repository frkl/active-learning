# Active Learning for VQA

Written in lua torch

## Dataset

Preprocessed VQA v2.0 train and val datasets used for this experiment are available [here](https://www.dropbox.com/s/sj08pqkjbzpza3j/balanced_vqa.zip?dl=0)

## Usage

```
th train.lua -data <path to train.t7> -data_test <path to dataset_val.t7>
```

The program will report accuracy on train (trainacc) and val (testacc) after each round of active learning, as well as some other statistics behind the algorithm. (If it ever pops an error on couldn't find sessions, create a sessions folder.)

Options such as active learning scoring function (-method) and restricting test set on only yes/no questions (-test_on_binary) are available, check train.lua for details.

# Further (minor) details

This program takes a bit more than a week to run for 50 iterations on Tesla K80 with default parameters.

The data preprocessing script is missing. But the structure of this dataset is straightforward, preprocessing is basically VGG19 features and NLTK tokenize, so could use improvements.

The base model is a LSTM+CNN model using dot-product to combine the scores. We learn a model parameter generator, and average over model parameters when making predicitons using pm.lua. This framework called "Bayesian NN" is broadly applicable to all NNs. Accuracy could be slightly lower than without Bayesian NN, but Bayesian NN provides new opportunities. 

An implementation of pm.lua for pytorch is given in pm.py.

This implementation uses the same active learning algorithm as vqa-rnn-mlp, but with new default parameters, clarifications, cleanups and functionality improvements.

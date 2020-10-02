# Using Long Short Term Memory Networks (LSTM) to Evaluate the Pollution of Beijing, China.

[![Keras](https://img.shields.io/pypi/format/Keras)](https://travis-ci.org/keras-team/keras) [![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://www.tensorflow.org/api_docs/)

## About LSTM

LSTM are a special kind of Recurrent Neural Network, capable of learning long-term dependencies. They were introduced by [Hochreiter & Schmidhuber (1997)](http://www.bioinf.jku.at/publications/older/2604.pdf), and were refined and popularized by many people in following works [[Sundermeyer et al. (2012)](https://www.isca-speech.org/archive/interspeech_2012/i12_0194.html), [Zhou, Chunting, et al. (2015)](https://arxiv.org/abs/1511.08630), [Gensler, André, et al. (2016)](https://ieeexplore.ieee.org/abstract/document/7844673)].

All recurrent neural networks have the form of a chain of repeating modules of neural network. LSTM also have this chain like structure, bellow follow the module in an LSTM.

<p aling="center"><img src="lstm_cell.png" width="85%" height="85%"></p> 
<h6 align="center">The repeating module in an LSTM containg four interacting layers. To know more about lstm neural networks, please <a href="http://elgibborsms.com/blog/introduction-to-long-short-term-memory/">click here</a>.</h6>


## Getting Started

### Prerequisites

Before you install the packages, it's necessaryly some knowledgmentes:

```
  Have a good knowledgment of Deep Learning, specialy of LSTM layer, and very nice knowledgment of python too.
```

### Install

Bellow, follow some of packages you have to be install for run this experiment:

- The first step is create run virtual environment (this step is optional, but I recommend):
```
  python3 -m pip install --user virtualenv
  source .virtualenv/bin/activate
```

- The second step is install Sklearn 
```
  pip install -U scikit-learn
```

- After install Sklearno, you need to install TensorFlow:

```
  pip install tensorflow
```

- When TensorFlow installation has finished, please install the Keras library:

```
  pip install keras
```

- And the last library was MatPlotLib for you plot some graphics to analisy the results:

```
  python -m pip install -U matplotlib
```

### Run the Experiment

After the instalations, you can run now the experiment type the following command in your terminal:

```
  sh run.sh
```

## Result  

Bellow, follow the train process history used for build the model and also for verify the overfitting.

<p align="center"><img src="plot.png" width="65%" height="65%"></p> 
<h6 align="center">History evaluation for see if are overfitting during the training process or no. In this figure was used mean squared error (MSE) as metric of the loss (y-axis). The x-axis are the number of epochs that used for train the model.</h6>

#

<p align="center"><b>Sincerely:</b> <a href="https://github.com/neemiasbsilva">Neemias B. da Silva</a></p>

#

''' Shared utility functions for downsampled hash sequence experiments. '''

import lasagne
import numpy as np
import os
import collections
import deepdish
import traceback
import functools
import glob
import sys
import pse
import theano

import feature_extraction

N_BITS = 32
OUTPUT_DIM = 128
N_HIDDEN = 2048
PSE_BEST_MODEL = "models/pse_best_model.h5"
CVR_BEST_MODEL = "models/cvr_best_model.h5"
QBSH_BEST_MODEL = "models/qbsh_best_model.h5"


#Best model hyperparams:
BEST_HYPERPARAMETERS = {'learning_rate': 9.7289563945014042e-05,
 'momentum': 0.0,
 'n_attention': 1,
 'n_conv': 0,
 'negative_importance': 5.613532235406355,
 'negative_threshold': 0.33213199035922386,
 'network': 'pse_big_filter',
 'downsample_frequency' : True,
}

def load_network(filetype, hyperparams, params_file):
    # Building the network
    build_network = build_pse_net_big_filter
    layers = build_network(
     (None, 1, None, feature_extraction.N_NOTES),
     np.zeros((1, feature_extraction.N_NOTES), theano.config.floatX),
     np.ones((1, feature_extraction.N_NOTES), theano.config.floatX),
     hyperparams['downsample_frequency'],
     hyperparams['n_attention'], hyperparams['n_conv'])
    # Loading params from disk
    network_params = deepdish.io.load(params_file)
    if "PSE" in params_file:
        if filetype == 'mp3':
            lasagne.layers.set_all_param_values(layers[-1], network_params['X'])
        elif filetype == 'mid':
            lasagne.layers.set_all_param_values(layers[-1], network_params['Y'])
    #else:
    #    lasagne.layers.set_all_param_values(layers[-1], network_params)        
    # Compile function for computing the output of the network
    compute_output = theano.function([layers[0].input_var], 
        lasagne.layers.get_output(layers[-1], deterministic=True))
    return layers, compute_output

def _build_input(input_shape, input_mean, input_std):
    layers = [lasagne.layers.InputLayer(shape=input_shape)]
    # Utilize training set statistics to standardize all inputs
    layers.append(lasagne.layers.standardize(
        layers[-1], input_mean, input_std, shared_axes=(0, 2)))
    return layers

def _build_big_filter_frontend(layers, downsample_frequency, n_conv):
    # Construct the pooling size based on whether we pool over frequency
    if downsample_frequency:
        pool_size = (2, 2)
    else:
        pool_size = (2, 1)
    # The first convolutional layer has filter size (5, 12), and Lasagne
    # doesn't allow same-mode convolutions with even filter sizes.  So, we need
    # to explicitly use a pad layer.
    filter_size = (5, 12)
    num_filters = 16
    if n_conv > 0:
        layers.append(lasagne.layers.PadLayer(
            layers[-1], width=((int(np.ceil((filter_size[0] - 1) / 2.)),
                            int(np.floor((filter_size[0] - 1) / 2.))),
                            (int(np.ceil((filter_size[1] - 1) / 2.)),
                            int(np.floor((filter_size[1] - 1) / 2.))))))
        # We will initialize weights to \sqrt{2/n_l}
        n_l = num_filters*np.prod(filter_size)
        layers.append(lasagne.layers.Conv2DLayer(
            layers[-1], stride=(1, 1), num_filters=num_filters,
            filter_size=filter_size,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Normal(np.sqrt(2./n_l))))
        layers.append(lasagne.layers.MaxPool2DLayer(
            layers[-1], pool_size, ignore_border=False))
    # Add n_conv 3x3 convlayers with 32 and 64 filters and pool layers
    filter_size = (3, 3)
    filters_per_layer = [32, 64]
    for num_filters in filters_per_layer[:n_conv - 1]:
        n_l = num_filters*np.prod(filter_size)
        layers.append(lasagne.layers.Conv2DLayer(
            layers[-1], stride=(1, 1), num_filters=num_filters,
            filter_size=filter_size,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Normal(np.sqrt(2./n_l)), pad='same'))
        layers.append(lasagne.layers.MaxPool2DLayer(
            layers[-1], pool_size, ignore_border=False))
    return layers


def _build_small_filters_frontend(layers, downsample_frequency, n_conv):
    # Construct the pooling size based on whether we pool over frequency
    if downsample_frequency:
        pool_size = (2, 2)
    else:
        pool_size = (2, 1)
    # Add three groups of 2x 3x3 convolutional layers followed by a pool layer
    filter_size = (3, 3)
    # Up to three conv layer groups will be made, with the following # filters
    filters_per_layer = [16, 32, 64]
    # Add in n_conv groups of 2x 3x3 filter layers and a max pool layer
    for num_filters in filters_per_layer[:n_conv]:
        n_l = num_filters*np.prod(filter_size)
        layers.append(lasagne.layers.Conv2DLayer(
            layers[-1], stride=(1, 1), num_filters=num_filters,
            filter_size=filter_size,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Normal(np.sqrt(2./n_l)), pad='same'))
        layers.append(lasagne.layers.Conv2DLayer(
            layers[-1], stride=(1, 1), num_filters=num_filters,
            filter_size=filter_size,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Normal(np.sqrt(2./n_l)), pad='same'))
        layers.append(lasagne.layers.MaxPool2DLayer(
            layers[-1], pool_size, ignore_border=False))
    return layers

def _build_ff_attention_dense(layers, n_attention, output_dim):
    # Combine the "n_channels" dimension with the "n_features"
    # dimension
    layers.append(lasagne.layers.DimshuffleLayer(layers[-1], (0, 2, 1, 3)))
    layers.append(lasagne.layers.ReshapeLayer(layers[-1], ([0], [1], -1)))
    # Function which constructs attention layers
    attention_layer_factory = lambda: pse.AttentionLayer(
        layers[-1], N_HIDDEN,
        # We must force He initialization because Lasagne doesn't like 1-dim
        # shapes in He and Glorot initializers
        v=lasagne.init.Normal(1./np.sqrt(layers[-1].output_shape[-1])),
        # We must also construct the bias scalar shared variable ourseves
        # because deepdish won't save numpy scalars
        ) 
        #c=theano.shared(np.array([0.], theano.config.floatX),
        #                broadcastable=(True,)))
    # Construct list of attention layers for later concatenation
    attention_layers = [attention_layer_factory() for _ in range(n_attention)]
    # Add all attention layers into the list of layers
    layers += attention_layers
    # Concatenate all attention layers
    layers.append(lasagne.layers.ConcatLayer(attention_layers))
    # Add dense hidden layers
    for hidden_layer_size in [N_HIDDEN, N_HIDDEN]:
        layers.append(lasagne.layers.DenseLayer(
            layers[-1], num_units=hidden_layer_size,
            nonlinearity=lasagne.nonlinearities.rectify))
    # Add output layer
    layers.append(lasagne.layers.DenseLayer(
        layers[-1], num_units=output_dim,
        nonlinearity=lasagne.nonlinearities.tanh))
    return layers



def build_pse_net_big_filter(input_shape, input_mean, input_std,
                             downsample_frequency, n_attention,
                             n_conv=3, output_dim=OUTPUT_DIM):
    '''
    Construct a list of layers of a network which embeds sequences in a
    fixed-dimensional output space using feedforward attention, which has a
    ``big'' 5x12 input filter and two 3x3 convolutional layers, all followed by
    max-pooling layers.

    Parameters
    ----------
    input_shape : tuple
        In what shape will data be supplied to the network?
    input_mean : np.ndarray
        Training set mean, to standardize inputs with.
    input_std : np.ndarray
        Training set standard deviation, to standardize inputs with.
    downsample_frequency : bool
        Whether to max-pool over frequency
    n_attention : int
        Number of attention layers
    n_conv : int
        Number of convolutional/pooling layer groups
    output_dim : int
        Output dimensionality

    Returns
    -------
    layers : list
        List of layer instances for this network.
    '''
    # Use utility functions to construct input, frontend, and dense output
    layers = _build_input(input_shape, input_mean, input_std)
    layers = _build_big_filter_frontend(
        layers, downsample_frequency, n_conv)
    layers = _build_ff_attention_dense(
        layers, n_attention, output_dim)
    return layers


def build_pse_net_small_filters(input_shape, input_mean, input_std,
                                downsample_frequency, n_attention,
                                n_conv=3, output_dim=OUTPUT_DIM):
    '''
    Construct a list of layers of a network which embeds sequences in a
    fixed-dimensional output space using feedforward attention, which has
    groups of two 3x3 convolutional layers and max-pooling layers.

    Parameters
    ----------
    input_shape : tuple
        In what shape will data be supplied to the network?
    input_mean : np.ndarray
        Training set mean, to standardize inputs with.
    input_std : np.ndarray
        Training set standard deviation, to standardize inputs with.
    downsample_frequency : bool
        Whether to max-pool over frequency
    n_attention : int
        Number of attention layers
    n_conv : int
        Number of convolutional/pooling layer groups
    output_dim : int
        Output dimensionality

    Returns
    -------
    layers : list
        List of layer instances for this network.
    '''
    # Use utility functions to construct input, frontend, and dense output
    layers = _build_input(input_shape, input_mean, input_std)
    layers = _build_small_filters_frontend(
        layers, downsample_frequency, n_conv)
    layers = _build_ff_attention_dense(
        layers, n_attention, output_dim)
    return layers

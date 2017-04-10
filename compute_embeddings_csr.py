import glob
import time

import theano
import theano.tensor as T
import lasagne
import deepdish
import numpy as np

import my_experiment_utils
import feature_extraction

MODEL_FILENAME = "models/cvr_best_model.h5"
BATCH_SIZE = 600
DATA_DIRECTORY = "data/cqts/"
TEST_FILENAMES = "data/myshs_test.txt"

#TODO: Create network, set params from best CSR model, get output
#		function
hyperparams = my_experiment_utils.BEST_HYPERPARAMETERS
def get_embedding_function(params, 
	hyperparams=hyperparams, filetype='mp3'):
    # Building the network
    build_network = my_experiment_utils.build_pse_net_big_filter
    layers = build_network(
     (None, 1, None, feature_extraction.N_NOTES),
     np.zeros((1, feature_extraction.N_NOTES), theano.config.floatX),
     np.ones((1, feature_extraction.N_NOTES), theano.config.floatX),
     hyperparams['downsample_frequency'],
     hyperparams['n_attention'], hyperparams['n_conv'])
    # Loading params from disk
    network_params = deepdish.io.load(params_file)
    if filetype == 'mp3':
      lasagne.layers.set_all_param_values(layers[-1], network_params['X'])
    else:
      lasagne.layers.set_all_param_values(layers[-1], network_params['Y'])
    # Compile function for computing the output of the network
    compute_output = theano.function([layers[0].input_var], 
        lasagne.layers.get_output(layers[-1], deterministic=True))
    return compute_output 


if __name__ == "__main__":
	with open('TEST_FILENAMES', 'r') as f:
		test_filenames = f.read()
	test_filenames = test_filenames.split('\n')
	if '' in test_filenames:
		test_filenames.remove('')
	test_embeddings = []
	mp3_embedding_fn=get_embedding_function("models/cvr_best_model.h5")
        i=0
	for test_filename in test_filenames:
                print("Computing embedding for file {}/{}: {}".format(
                             i, len(test_filenames), test_filename))
                i=i+1
		mp3_gram = deepdish.io.load("data/cqts/"+test_filename)['gram']
		test_embeddings.append(
			mp3_embedding_fn(mp3_gram.reshape(1, 1, *mp3_gram.shape))
			)
	np.savetxt('output/test_embeddings.csv', test_embeddings, delimiter=',')

	#TODO: Load spectrogram h5 files every batch and pass them through
	#		the network.

	#TODO: Write the 

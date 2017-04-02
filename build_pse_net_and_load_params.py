'''
This script creates a pse net and loads parameters from disk
'''
import my_experiment_utils
import feature_extraction
import numpy as np
import theano
import deepdish
import lasagne
import os
import librosa
import traceback
import pretty_midi

hyperparams = my_experiment_utils.BEST_HYPERPARAMETERS
hyperparams['downsample_frequency'] = True

def create_network(filetype, hyperparams):
    # Building the network
    build_network = my_experiment_utils.build_pse_net_big_filter
    layers = build_network(
     (None, 1, None, feature_extraction.N_NOTES),
     np.zeros((1, feature_extraction.N_NOTES), theano.config.floatX),
     np.ones((1, feature_extraction.N_NOTES), theano.config.floatX),
     hyperparams['downsample_frequency'],
     hyperparams['n_attention'], hyperparams['n_conv'])
    # Loading params from disk
    network_params = deepdish.io.load('best_model.h5')
    if filetype == 'mp3':
      lasagne.layers.set_all_param_values(layers[-1], network_params['X'])
    else:
      lasagne.layers.set_all_param_values(layers[-1], network_params['Y'])
    # Compile function for computing the output of the network
    compute_output = theano.function([layers[0].input_var], 
        lasagne.layers.get_output(layers[-1], deterministic=True))
    return compute_output 

def create_midi_embedding(midi_embedding_fn, mid_filename):
    m = pretty_midi.PrettyMIDI(mid_filename)
    midi_gram = feature_extraction.midi_cqt(m)
    return midi_embedding_fn(midi_gram.reshape(1, 1, *midi_gram.shape))

def create_mp3_embedding(mp3_embedding_fn, mp3_filename):
    audio_data, _ = librosa.load(mp3_filename, 
        sr=feature_extraction.AUDIO_FS)
    mp3_gram = feature_extraction.audio_cqt(audio_data)
    return mp3_embedding_fn(mp3_gram.reshape(1, 1, *mp3_gram.shape))
"""
def process_one_mp3(mp3_filename, skip=True):
    '''
    Load in an mp3, get the features, and write the features out

    :parameters:
        - mp3_filename : str
            Path to an mp3 file
        - skip : bool
            Whether to skip files when the h5 already exists
    '''
    # h5 files go in the 'h5' dir instead of 'mp3'
    output_filename = mp3_filename.replace('mp3', 'h5')
    # Skip files already created
    if skip and os.path.exists(output_filename):
        return
    try:
        # Load audio and compute CQT
        audio_data, _ = librosa.load(
            mp3_filename, sr=feature_extraction.AUDIO_FS)
        cqt = feature_extraction.audio_cqt(audio_data)
        # Create subdirectories if they don't exist
        if not os.path.exists(os.path.split(output_filename)[0]):
            os.makedirs(os.path.split(output_filename)[0])
        # Save CQT
        deepdish.io.save(output_filename, {'gram': cqt})
    except Exception as e:
        print("Error processing {}: {}".format(
            mp3_filename, traceback.format_exc(e)))
"""
"""
'''
def process_one_midi(midi_filename, skip=True):
    '''
'''
    Load in an mp3, get the features, and write the features out

    :parameters:
        - midi_filename : str
            Path to an midi file
        - skip : bool
            Whether to skip files when the h5 already exists
'''
    '''
    # h5 files go in the 'h5' dir instead of 'mp3'
    output_filename = midi_filename.replace('mid', 'h5')
    # Skip files already created
    if skip and os.path.exists(output_filename):
        return
    try:
        # Load audio and compute CQT
        midi_data, _ = librosa.load(
            midi_filename, sr=feature_extraction.AUDIO_FS)
        cqt = feature_extraction.audio_cqt(audio_data)
        # Create subdirectories if they don't exist
        if not os.path.exists(os.path.split(output_filename)[0]):
            os.makedirs(os.path.split(output_filename)[0])
        # Save CQT
        deepdish.io.save(output_filename, {'gram': cqt})
    except Exception as e:
        print("Error processing {}: {}".format(
            mp3_filename, traceback.format_exc(e)))
'''
'''

def create_midi_network():
    build_network = my_experiment_utils.build_pse_net_big_filter
    layers = build_network2(
     (None, 1, None, feature_extraction.N_NOTES),
     np.zeros((1, feature_extraction.N_NOTES), theano.config.floatX),
     np.ones((1, feature_extraction.N_NOTES), theano.config.floatX),
     hyperparams['downsample_frequency'],
     hyperparams['n_attention'], hyperparams['n_conv'])
    lasagne.set_all_param_values(layers[-1], network_params['Y'])
    compute_output = theano.function([layers[0].input_var],
        lasagne.layers.get_output(layers[-1], deterministic=True))
    return compute_output

def get_embedding(filename, embedding_function):
    #TODO: save CQT spectrogram for file as h5 file
    #TODO: load h5 spectrogram
    #TODO: if filename has mp3 extension, get mp3 net embedding
    #TODO: if filename has midi extension, get midi net embedding
    #TODO: print filename along

def get_pairwise_distance(embeding1, embedding2):
    #TODO: return euclidean distance between two embeddings
    
'''


'''
# Load in CQTs and write out downsampled hash sequences
for entry in file_list:
            try:
                # Construct CQT h5 file path from file index entry
                h5_file = os.path.join(
                    DATA_PATH, dataset, 'h5', entry['path'] + '.h5')
                # Load in CQT
                gram = deepdish.io.load(h5_file)['gram']
                # Compute embedding for this sequence
                embedding = compute_output(
                    gram.reshape(1, 1, *gram.shape))
                # Construct output path to the same location in
                # RESULTS_PATH/dhs_(dataset)_hash_sequences
                output_file = os.path.join(
                    RESULTS_PATH, 'pse_{}_embeddings'.format(dataset),
                    entry['path'] + '.mpk')
'''
"""
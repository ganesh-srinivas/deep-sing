import glob
import time

import theano
import theano.tensor as T
import lasagne
import deepdish
import numpy as np

import my_experiment_utils
import feature_extraction

BATCH_SIZE = 60
N_EPOCHS = 1000
DATA_DIRECTORY = "data/"

#Mentioned in Thesis. I think it refers to triplet loss constants.
#ALPHA = 3.342 
#M = 1.055 


#TODO: load network with PSE model weights
hyperparams = my_experiment_utils.BEST_HYPERPARAMETERS
layers, _ = my_experiment_utils.load_network('mp3', 
	hyperparams,
	my_experiment_utils.PSE_BEST_MODEL)
embedding = lasagne.layers.get_output(layers[-1])

#TODO: Define the cost function and update rule
dis1 = ((embedding[::3] - embedding[1::3])**2).sum(axis=1)
dis2 = ((embedding[::3] - embedding[2::3])**2).sum(axis=1)
s = dis1 - dis2 + 1
loss = T.sum(s*T.gt(s, 0.0))
loss = loss.mean()
params = lasagne.layers.get_all_params(layers[-1], 
	trainable=True)
# Define the cost function and update rule
updates = lasagne.updates.rmsprop(loss, 
	params, 
	learning_rate=hyperparams['learning_rate']/2,
	rho=hyperparams['momentum']
	)

#TODO: Compile the training function
training_function = theano.function([layers[0].input_var],
	loss, updates=updates)

#TODO: Train the model (epochs and batches)
with open(DATA_DIRECTORY+"myshs_train.txt", "r") as f:
  xtrain_filenames = f.read()
xtrain_filenames = xtrain_filenames.split('\n')
xtrain_filenames.remove('')

n_batches = len(xtrain_filenames)/BATCH_SIZE

with open(DATA_DIRECTORY+"myshs_validation.txt", "r") as g:
  xvalidation_filenames = g.read()
xvalidation_filenames = xvalidation_filenames.split('\n')
xvalidation_filenames.remove('')
n_validation_batches = len(xvalidation_filenames)/BATCH_SIZE

best_objective = np.inf
start_time = time.time()
cost_history = []
for epoch in range(N_EPOCHS):
	#try:
        if True:
		st = time.time()
		batch_cost_history = []
		for batch in range(n_batches + 1):
			#TODO: load BATCH_SIZE spectrograms from train into xbatch
			input_data = [deepdish.io.load(DATA_DIRECTORY+"cqts/"+f)['gram'] for f in 
			xtrain_filenames[batch*BATCH_SIZE: (batch+1)*BATCH_SIZE]]
			size=min(i.shape[0] for i in input_data)
			offsets = [np.random.random_integers(0, i.shape[0]-size)
				for i in input_data]
			batch_sampled = [i[o:o+size,:] for i,o in zip(input_data, offsets)]
			shape = [batch_sampled[0].shape[-2], batch_sampled[0].shape[-1]]
			shape[:0] = [len(batch_sampled)]
                        #print(type(batch_sampled), type(shape), shape)
			xbatch = np.concatenate(batch_sampled).reshape(shape)
                        xbatch = xbatch.reshape(xbatch.shape[0], 1, xbatch.shape[-2], xbatch.shape[-1])
			batch_cost = training_function(xbatch)
			batch_cost_history.append(batch_cost)
                        if batch%10==0:
                          print("Epoch {}/{}: Batch {}/{}: Batch loss {}.".format(epoch, N_EPOCHS, batch, n_batches, batch_cost))
			del input_data, batch_sampled, xbatch
		epoch_cost = np.mean(batch_cost_history)
		cost_history.append(epoch_cost)
		#Compute validation loss
		#if loss on validation set (0.05% of dataset) is less than 
		#best_objective, consider that epoch as best epoch, and 
		#store the weights of that epoch
		validation_batch_cost_history=[]
		for vbatch in range(n_validation_batches):
			#TODO: load BATCH_SIZE spectrograms from validate
			# into xbatch
			validation_data = [deepdish.io.load(DATA_DIRECTORY+"cqts/"+f)['gram'] for f in 
			xvalidation_filenames[vbatch*BATCH_SIZE: (vbatch+1)*BATCH_SIZE]]

			size=min(i.shape[0] for i in validation_data)
			offsets = [np.random.random_integers(0, i.shape[0]-size)
				for i in validation_data]
			vbatch_sampled = [i[o:o+size,:] for i,o in zip(validation_data, offsets)]
			shape = [vbatch_sampled[0].shape[-2], vbatch_sampled[0].shape[-1]]
			shape[:0] = [len(vbatch_sampled)]
			xvbatch = np.concatenate(vbatch_sampled).reshape(shape)
                        xvbatch = xvbatch.reshape(xvbatch.shape[0], 1, xvbatch.shape[-2], xvbatch.shape[-1])

			validation_batch_cost = training_function(xvbatch)
			validation_batch_cost_history.append(validation_batch_cost)
			del validation_data, vbatch_sampled, xvbatch

		validation_loss = np.mean(validation_batch_cost_history)
		if validation_loss < best_objective:
			best_objective = validation_loss
			best_epoch = epoch
			best_model = {'X': lasagne.layers.get_all_param_values(
				layers)}
			deepdish.io.save(my_experiment_utils.CVR_BEST_MODEL, best_model)

		en = time.time()
		print("Epoch {}/{}, Average train loss: {}. Average \
			validation loss: {}. Elapsed time: {} \
			seconds".format(epoch+1, N_EPOCHS, epoch_cost, 
				validation_loss, en-st))
	#except Exception as e:
        #		print(e)
	#	continue

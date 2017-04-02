# Deep Sing (incomplete!)
## A deep learning solution to the Query by Singing/Humming (QBSH) 
## problem in Music Information Retrieval

This code (in-progress) implements the following solution:
1. A neural network that produces embeddings for songs in order to do cover  
 song recognition. Specifically, an attention-based feedforward convolutional 
 neural network is trained using a triplet loss function on a subset of the 
 Second Hand Songs (SHS) dataset. The loss function encourages the network 
 to produce similar (in terms of pairwise distance) embeddings for a song 
 and its cover, and dissimilar embeddings for unrelated songs.

2. Transfer learning is performed after this: weights are learned for doing 
 cover song recognition, and then the same network is trained a little 
 further on the comparatively smaller dataset for the related and **desired** 
 task of query by singing/humming (QBSH).

Note: This idea and much of the code is adapted heavily from a part of Colin Raffel's PhD thesis work: [musical sequences embedded in a Euclidean space by training on matching and non-matching pairs of sequences](http://colinraffel.com/publications/icassp2016pruning.pdf). The code this project has taken after can be examined at [MIDI Dataset](https://github.com/craffel/midi-dataset).


## Datasets
1. MP3 preview clips for a subset of the [Second Hand Songs](https://labrosa.ee.columbia.edu/millionsong/secondhand) dataset: for training the convnet to produce embeddings for cover song recognition.

2. [Jang dataset](http://www.music-ir.org/mirex/wiki/2016:MIREX2016_Results): for fine-tuning the network weights to produce similar embeddings for ground truth audio and hummed queries against that.

## Requirements
- Theano
- Lasagne
- [pse](https://github.com/craffel/pse)
- deepdish
- librosa
- traceback
- pretty_midi


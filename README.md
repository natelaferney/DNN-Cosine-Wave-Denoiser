Cosine Wave Denoiser
====================

The purpose of this code is to train a neural network
to label a cosine waves that are at fixed frequencies but 
different phases and are corrupted with noise. 

The classifier is built using a deep neural network with five
layers with cross entropy as the loss function.

The noise consists of two components, gaussian white noise
and harmonic noise. The harmonic noise 
selects a random number (up to 12) of cosine waves 
at varying frequencies and are added to the original
cosine wave of interest.

To run the neural network, run the `dnn_denoiser.py` file.

The model reports back about 96-97% accuracy when sampled
at 112 samples per unit of time with four different phases.

Dependencies
=============

TensorFlow v1.0

Numpy v1.11

Credits
==========

This application uses Open Source components. You can find
the source code of their open source project along with licence
information below. We acknowledge and are grateful to these
developers for their contribution to open source software.

Project: tensorflow-mnist-tutorial (https://github.com/martin-gorner/tensorflow-mnist-tutorial)
Specifically (https://github.com/martin-gorner/tensorflow-mnist-tutorial/blob/master/mnist_2.2_five_layers_relu_lrdecay_dropout.py)
Licence: Apache Licence 2.0


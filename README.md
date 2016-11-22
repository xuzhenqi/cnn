cnn
===

This is a matlab-code implementation of convolutional neural network.

***Notes***: This repo was deprecated. I suggest you use other deep learning tools, such as caffe, mxnet, tensorflow. They are far more easy to use.

Functionality
---

* supported layertypes : 'conv', 'sigmoid', 'maxpool', 'meanpool', 'relu', 'tanh', 'softmax', 'stack2line', 'softsign'
* supported loss function : 'crossEntropy' 
* supported training method : 'SGD'
* debug tools : deconvnet, display\_training, gradent\_check
* supported demo dataset : 'MNIST', 'GENKI-R2009a'

Usage
---

The structure of convolutional neural network is
	conv pool [conv pool] stack2line ['nonlinear']
[] means optional, and can be replicated for many times.

Layer
---

### conv ###
implement convolution computing. To make codes flexible, I do not implemente non-linear functions after convlution. You can add a layer to complete the non-linear instead.
To use 'conv' layer, you should specify the following parameters:
	**filterDim**
	**numFilters**
	**nonlineartype**
If the inputs has multimaps, then you may specify the connection table between the input maps and the output maps:
	**conn\_matrix**
If you don't specify the connection table, then each output map is connected to all input maps.

### pool/pool ###
'maxpool' and 'meanpool' are both pooling layer. To use pooling layer, the following parameters should be specified:
	**poolDim**
	**pooltypes**
### relu/tanh/sigmoid/softmax/softsign ###
These four types of layers mainly do the non-linear function to the input.
	y = max(0,x)
	y = tanh(x)
	y = 1/exp(-x)
	y = softmax(x)
	y = x/(1+abs(x))
To use them, the following parameters should be specified:
	**size**
Besides, the softmax layer is usually used as output layer.
### stack2line ###
After convlution and pooling, the multi-dimention "outputs" usually are converted to a vector to be used as the inputs of the densely connected non-linear layers. And stack2line layer is to indicate this converting.

Training Method
---

### SGD ###

Computing Device
---

### CPUonly ###

Debug Tools
---

### deconvnet ###
### display\_training ###
### gradient\_check ###

Dataset
---

### MNIST ###
### GENKI-R2009a ###

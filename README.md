cnn
===

This is a matlab-code implementation of convolutional neural network.

Functionality
---

* supported layertypes : 'conv', 'sigmoid', 'maxpool', 'meanpool', 'relu', 'tanh', 'softmax', 'stack2line'
* supported training method : 'SGD'
* supported computing device : 'GPU', 'CPUonly'
* debug tools : deconvnet, display\_training, gradent\_check
* supported demo dataset : 'MNIST', 'GENKI-R2009a'

Usage
---

Layer
---

### conv ###
	implement convolution computing. To make codes flexible, I do not implemente non-linear functions after convlution. You can add a layer to complete the non-linear instead.
	To use 'conv' layer, you should specify the following parameters:
	**filterDim**
	**numFilters**
	If the inputs has multimaps, then you may specify the connection table between the input maps and the output maps:
	**connect\_table**
	If you don't specify the connection table, then each output map is connected to all input maps.

### sigmoid ###
	y = 1/exp(-x)
### pool/pool ###
	'maxpool' and 'meanpool' are both pooling layer. To use pooling layer, the following parameters should be specified:
	**poolDim**
	**pooltypes**
### relu ###
	y = max(0,x)
### tanh ###
	y = tanh(x)
### softmax ###
	y = softmax(x)
	The softmax layer usually use in the output layer.
### stack2line ###
	After convlution and pooling, the multi-dimention "outputs" usually are converted to a vector to be used as the inputs of the densely connected non-linear layers. And stack2line layer is to indicate this converting.

Training Method
---

### SGD ###

Computing Device
---

### GPU ###
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

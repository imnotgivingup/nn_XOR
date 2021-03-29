# nn_XOR
A basic neural net created from scratch, to implement the XOR function

This would not be possible without the outstanding book written by Michael Nielsen, the link to which I am posting down below.
http://neuralnetworksanddeeplearning.com

The model is very basic, a network of 3 [2 --> 10 --> 1] linear layers, with a hidden layer of 10 neurons.
There is only 1 batch, with 4 training data elements viz the standard XOR truth table entries.
I wouldn't call this learning, it's more of memorizing, as the system iterates over the same input data (of length 4) over and over again until it reaches a fine state of accuracy and precision.
The program, once done training, evaluates the network based on the truth table itself, again not very abiding to the laws of evaluation of neural networks.

It finally prints the predicted value for each set of inputs, and the average accuracy percentage.

Here is an example output...
Predicted result [array([[0.00721381]]), array([[0.99074585]]), array([[0.98971749]]), array([[0.01135228]])]
Expected result [[0],[1],[1],[0]]
99.04743118916099 % accurate

/*Header File for FFNNet.cpp
* This will define everything for the Feed Forward Neural Network (FFNN) class!
* What I'll be doing first is a feed forward neural network based on 
* backpropagation learning with momentum. 
* The learning rate, momentum and number of hidden nodes need to be variable 
* so the network performance under different conditions can be recorded.
* The activation function will need to be toggle-able between the logistic 
* and tanh functions.
* The "holdout" and "cross validation" techniques will be used on testing data.
*
* Author @ Alex vanKooten
* Version: 1.1 (02.23.2017)                                                   */

#ifndef NNet
#define NNet
#include "Scanner.h"

class FFNNet {
	//friend class NNetTrainer; //NNTrainer can access private memebers of FFNNet

public:
	FFNNet(int num_in, int num_hid, int num_out); //basic constructor
	~FFNNet();	//basic destructor
	bool loadWeights(char* input_file_name); //load neuron weights
	bool saveWeights(char* output_file_name); //save neuron weights
	int* feedForwardPattern(double* input_pattern); //returns the result of a FF through the NN 
	double getDatasetMSE(std::vector<dataSet*>& data_set); //return the MSE of the FFNN on the data set
	double getDatasetAcc(std::vector<dataSet*>& data_set); //returns the accuracy of the FFNN on the data set

	/* I wanted the following to be private but was having trouble with the friend class not granting me access. 
	 * I'll try to remedy this later but for functionality it's not really important.*/
	int num_inputs;		//number of input neurons
	int num_hidden;		//number of hidden neurons
	int num_outputs;	//number of output neurons
	double* input_neurons;	//vector of input neurons
	double* hidden_neurons;	//vector of hidden neurons
	double* output_neurons;	//vector of output neurons
	double** input_hidden_weights;	//weight matrix from input layer to hidden layer
	double** hidden_output_weights;	//weight matric from hidden layer to output layer

	void initWeights();	//randomly initialize the weights
	inline int squashOutput(double o); //squash outputs to either 0, 1 or -1
	inline double activationFunc(double af); //apply the activation function
	void feedForward(double* input_pattern); //does weight calc & sets neuron values
};

#endif
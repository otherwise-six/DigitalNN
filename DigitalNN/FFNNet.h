/*Header File for FFNNet.cpp
* This will define everything for our glorious 
* Feed Forward Neural Network (FFNN) class!
*
* Author @ Alex vanKooten
* Version: 1.0 (02.23.2017)                                                   */

#pragma once
#ifndef NNet
#define NNet
#include "Scanner.h"

class NNetTrainer;

class FFNNet {

private:
	friend NNetTrainer;
	
	//number of each kind of neuron
	int num_inputs;
	int num_hidden;
	int num_outputs;
	
	//vectors of neuron types
	double* input_neurons;
	double* hidden_neurons;
	double* output_neurons;
	
	//weight matrices
	double** input_hidden_weights;
	double** hidden_output_weights;

	void init_weights();	//randomly initialize the weights
	inline int squash_output(double o); //squash outputs to either 0, 1 or -1
	inline double activation_func(double af); //apply the activation function
	void feed_forward(double* input_pattern); //does weight calc & sets neuron values
	
public:
	FFNNet(int num_in, int num_hid, int num_out); //basic constructor
	~FFNNet();	//basic destructor
	bool load_weights(char* input_file_name); //load neuron weights
	bool save_weights(char* output_file_name); //save neuron weights
	int* feed_forward_pattern(double* input_pattern); //returns the result of a FF through the NN 
	double get_dataset_MSE(std::vector<dataSet*>& data_set); //return the MSE of the FFNN on the data set
	double get_dataset_acc(std::vector<dataSet*>& data_set); //returns the accuracy of the FFNN on the data set

};

#endif
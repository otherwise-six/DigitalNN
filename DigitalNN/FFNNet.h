/*Header File for FFNNet.cpp
* This will define everything for our glorious 
* Feed Forward Neural Network (FFNN) class!
*
* Author @ Alex vanKooten
* Version: 0.9 (02.21.2017)                                                   */

#pragma once
#ifndef NNet
#define NNet
#include "Scanner.h"

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


}

#endif
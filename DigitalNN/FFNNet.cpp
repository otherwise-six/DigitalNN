/*This is the Feed Forward Neural Network (FFNN) class!
* Every function that the FFNN does is housed and explained here.
* What I'll be doing first is a feed forward neural network based on 
* backpropagation learning with momentum. 
* The learning rate, momentum and number of hidden nodes need to be variable 
* so the network performance under different conditions can be recorded.
* The activation function will need to be toggle-able between the logistic 
* and tanh functions.
* The "holdout" and "cross validation" techniques will be used on testing data.
* Author @ Alex vanKooten
* Version: 1.1 (02.23.2017)                                                   */

#include "FFNNet.h"
#include <math.h>
#include <vector>
#include <fstream>
#include <iostream>

/*FFNN constuctor*/
FFNNet::FFNNet(int num_in, int num_hid, int num_out) : 
	num_inputs(num_in), num_hidden(num_hid), num_outputs(num_out) {
	
	input_neurons = new(double[num_inputs + 1]);
	for (int i = 0; i < num_inputs; i++) { //make input neuron list
		input_neurons[i] = 0;
	}
	input_neurons[num_inputs] = -1; //make an input bias neuron

	hidden_neurons = new(double[num_hidden + 1]);
	for (int i = 0; i < num_hidden; i++) { //make hidden neuron list
		hidden_neurons[i] = 0;
	}
	hidden_neurons[num_hidden] = -1; //make hidden bias neuron

	output_neurons = new(double[num_outputs]); 
	for (int i = 0; i < num_outputs; i++) { //make output neuron list
		output_neurons[i] = 0;
	}

	input_hidden_weights = new(double*[num_inputs + 1]);
	for (int i = 0; i <= num_inputs; i++) {	//make input-to-hidden weight list
		input_hidden_weights[i] = new (double[num_hidden]);
		for (int j = 0; j < num_hidden; j++) {
			input_hidden_weights[i][j] = 0;
		}
	}

	hidden_output_weights = new(double*[num_hidden + 1]);
	for (int i = 0; i <= num_hidden; i++) { //make hidden-to-output weight list
		hidden_output_weights[i] = new (double[num_outputs]);
		for (int j = 0; j < num_outputs; j++) {
			hidden_output_weights[i][j] = 0;
		}
	}

	initWeights(); //initialize the weights
}

/*FFNN destructor (basically just burn it all to the ground)*/
FFNNet::~FFNNet(){
	/*delete all neurons*/
	delete[] input_neurons;
	delete[] hidden_neurons;
	delete[] output_neurons;

	/*delete all the stored weights*/
	for (int i = 0; i <= num_inputs; i++) {
		delete[] input_hidden_weights[i];
	}
	for (int j = 0; j <= num_hidden; j++) {
		delete[] hidden_output_weights[j];
	}
	delete[] input_hidden_weights;
	delete[] hidden_output_weights;
}

//TODO: switch to that newer one that Anna lectured!
/*activation function (for now sigmoid)*/
inline double FFNNet::activationFunc(double af) {
	return 1 / (1 + exp(-af)); //sigmoid function
}

/*squash the output to either 0, 1 or -1*/
inline int FFNNet::squashOutput(double o) {
	if (o < 0.1) {
		return 0;
	}
	else if (o > 0.9) {
		return 1;
	}
	else {
		return -1;
	}
}

/*randomly initialize the neuron weights*/
void FFNNet::initWeights() {
	double hidden_range = (1 / sqrt((double)num_inputs)); //set hidden range
	double output_range = (1 / sqrt((double)num_hidden)); //set output range

	for (int i = 0; i <= num_inputs; i++) { //init input-to-hidden weights 	
		for (int j = 0; j < num_hidden; j++) { //randomize weights
			input_hidden_weights[i][j] = ((((double)(rand() % 100) + 1) / 100)
				* hidden_range * 2) - hidden_range;
		}
	}

	for (int m = 0; m <= num_hidden; m++) { //init input-to-hidden weights 	
		for (int n = 0; n < num_outputs; n++) { //randomize weights
			hidden_output_weights[m][n] = ((((double)(rand() % 100) + 1) / 100)
				* output_range * 2) - output_range;
		}
	}
}

/*load neuron weights*/
bool FFNNet::loadWeights(char* file_name) {
	std::fstream input_file;
	input_file.open(file_name, std::ios::in); //open weight file

	if (input_file.is_open()) { 
		std::vector<double> weights;
		std::string line = "";

		while (!input_file.eof()) { //while there's still data to read
			getline(input_file, line);

			if (line.length() > 2) { //read the line
				//store inputs		
				char* char_string = new char[line.size() + 1];
				char* token;
				strcpy_s(char_string, line.size() + 1, line.c_str());

				//tokenize
				int i = 0;
				char* next_token = NULL;
				token = strtok_s(char_string, ",", &next_token);

				while (token != NULL) { //as long as there's more stuff in token
					weights.push_back(atof(token));
					token = strtok_s(NULL, ",", &next_token); //move token forward
					i++;
				}

				//free memory
				delete[] char_string;
			}
		}

		//check if sufficient weights were loaded
		if (weights.size() != (((num_inputs + 1) * num_hidden) 
			+ ((num_hidden + 1) * num_outputs))) { //check if the right number of weights were loaded
			std::cout << "\nInsuffiecient number of weights in the file: " << file_name << "!\n";
			input_file.close();
			return false;
		} else {
			int ptr = 0; //pointer to help set weights
			for (int i = 0; i <= num_inputs; i++) { 
				for (int j = 0; j < num_hidden; j++) {
					input_hidden_weights[i][j] = weights[ptr++]; //set input-to-hidden weights
				}
			}
			for (int i = 0; i <= num_hidden; i++) {
				for (int j = 0; j < num_outputs; j++) {
					hidden_output_weights[i][j] = weights[ptr++]; //set hidden-to-output weights
				}
			}

			std::cout << "\nNeuron weights from " << file_name << " loaded successfully!\n";
			input_file.close(); //always close what you open!
			return true;
		}
	} else {
		std::cout << "\n" << file_name << " was unable to be opened!\n";
		return false;
	}
}

/*save neuron weights*/
bool FFNNet::saveWeights(char* file_name) {
	std::fstream output_file;
	output_file.open(file_name, std::ios::out); //open weight file to save into

	if (output_file.is_open()) {
		output_file.precision(40); //TODO: see how precision change affects results

		for (int i = 0; i <= num_inputs; i++) { //save input-to-hidden weights
			for (int j = 0; j < num_hidden; j++) {
				output_file << input_hidden_weights[i][j] << ",";
			}
		}

		for (int m = 0; m <= num_hidden; m++) { //save hidden-to-output weights
			for (int n = 0; n < num_outputs; n++) {
				output_file << hidden_output_weights[m][n];
				if ((m * num_outputs + n + 1) != ((num_hidden + 1) * num_outputs)) {
					output_file << ",";
				}
			}
		}

		std::cout << "\nNeuron weights successfully saved as " << file_name << "\n";
		output_file.close(); //always close what you open!
		return true;
	} else {
		std::cout << "\n" << file_name << " was unable to be created!\n";
		return false;
	}
}

/*returns the result from feeding an input pattern forward through the network*/
int* FFNNet::feedForwardPattern(double *input_pattern) {
	feedForward(input_pattern);

	int* results = new int[num_outputs]; //create a copy of the output results
	for (int i = 0; i < num_outputs; i++) {
		results[i] = squashOutput(output_neurons[i]);
	}
	return results;
}

/*feed forward method; does weight calc & sets neuron values*/
void FFNNet::feedForward(double* input_pattern) {
	for (int i = 0; i < num_inputs; i++) {
		input_neurons[i] = input_pattern[i]; //set input neurons values
	}

	for (int j = 0; j < num_hidden; j++) { //calc hidden layer values (including the bias neuron)
		hidden_neurons[j] = 0; //clear the hidden value
		for (int i = 0; i <= num_inputs; i++) { //calc weighted sum of input pattern (and the bias neuron)
			hidden_neurons[j] += input_neurons[i] * input_hidden_weights[i][j];
		}
		hidden_neurons[j] = activationFunc(hidden_neurons[j]); //set to squashed result
	}

	for (int k = 0; k < num_outputs; k++) { //calc output layer values (including the bias neuron)
		output_neurons[k] = 0; //clear the output value
		for (int j = 0; j <= num_hidden; j++) { //calc weighted sum of input pattern (and the bias neuron)
			output_neurons[k] += hidden_neurons[j] * hidden_output_weights[j][k];
		}
		output_neurons[k] = activationFunc(output_neurons[k]); //set to squashed result
	}
}

/*return the mean squared error (MSE) of the FFNN on the data set*/
double FFNNet::getDatasetMSE(std::vector<dataSet*>& dataset) {
	double mse = 0;
	for (int i = 0; i < (int)dataset.size(); i++) { //cycle through array of training inputs
		feedForward(dataset[i]->input_data);		//feed forward inputs, backprop errors
		for (int k = 0; k < num_outputs; k++) { //compare outputs with target output values
			mse += pow((output_neurons[k] - dataset[i]->target[k]), 2); //sum MSEs together
		}
	}
	return (mse / (num_outputs * dataset.size())); //calc error percentage
}

/*returns the accuracy of the FFNN on the data set*/
double FFNNet::getDatasetAcc(std::vector<dataSet*>& dataset) {
	double incorrect_results = 0; //keep track of incorrect results

	for (int i = 0; i < (int)dataset.size(); i++) { //cycle through array of training inputs
		feedForward(dataset[i]->input_data); //feed forward inputs, backprop errors
		bool correct_result = true; //correct input pattern "flag"

		for (int k = 0; k < num_outputs; k++) { //compare outputs with target output values
			if ((squashOutput(output_neurons[k])) != (dataset[i]->target[k])) {
				correct_result = false; //if target != output, then flag = false
			}
		}

		if (!correct_result) { //incorrect results increase the training error 
			incorrect_results++; 
		}
	}

	 //calculate error and return as percentage
	return (100 - ((incorrect_results / dataset.size()) * 100)); //calc error percentage
}
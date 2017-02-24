/*Header File for NNTrainer.cpp
* This will define everything we need to train a neural network.
* This will be a basic Gradient Descent trainer that uses 
* batch learning and incorporates momentum.
*
* Author @ Alex vanKooten
* Version: 1.0 (02.23.2017)                                                   */

#pragma once
#ifndef NeuralNetTrainer
#define NeuralNetTrainer

#include "FFNNet.h"
#include <fstream>
#include <vector>

#define MOMENTUM 0.9		//default momentum
#define MAX_EPOCHS 1500		//default maximum number of epochs before the NN stops training
#define GOAL_ACC 85			//default accuracy the NN will have to hit to stop training
#define GOAL_MSE 0.001		//default mean squared error
#define LEARNING_RATE 0.001	//default learning rate

class NNTrainer {
public:
	NNTrainer(FFNNet* nn); //basic constructor
	void useBatchLearn(bool flag) { //true = use batch training
		use_batch = flag;
	}
	void trainNNet(trainDataSet* train_set); //train a NN using gradient descent
	void setStopConditions(int m_epochs, double g_acc); //set up training stopping conditions 
	void enableLog(const char* file_name, int resolution);	//enable log to record training
	void setTrainingVariables(double learn_rate, double mo, bool batch); //set basic training variables

private:
	FFNNet* neural_net;		//the network to be trained
	long epoch;				//current epoch of training for the NN
	bool use_batch;			//flag to use batch learning
	long max_epochs;		//maximum number of allowable epochs for NN to be trained
	double momentum;		//improves stochastic learning performance
	double goal_acc;		//desired MSE/accuracy the NN wants
	double learning_rate;	//adjusts weight update	step size
		
	std::fstream log_file;	//log file
	bool log_enabled;		//flag to log training
	int log_resolution;		//sets the level of detail of the log
	int last_logged_epoch;	//number of the last epoch successfully logged

	double training_set_acc;		//training set accuracy per epoch
	double training_set_MSE;		//training set MSE per epoch
	double validation_set_acc;		//validation set accuracy per epoch
	double validation_set_MSE;		//validation set MSE per epoch
	double generalization_set_acc;	//generalization set accuracy per epoch
	double generalization_set_MSE;	//generalization set MSE per epoch

	double** input_hidden_delta;	//weight change from input to hidden layer
	double** hidden_output_delta;	//weight change from hidden to output layer
	double* hidden_err_gradients;	//array of hidden layer error gradients
	double* output_err_gradients;	//array of output layer error gradients

	void printHeader();		//prints a header to console with training info
	void updateWeights();	//method to update all NN weights using delta values
	void backprop(double* goal_outputs); //method to peform backprop of errors through the NN and get delta values 
	double getHiddenErrGradient(int target); //returns the input error gradient 
	void runSingleEpoch(std::vector<dataSet*> training_set); //runs a singel training epoch
	inline double getOutputErrGradient(double target_value, double output_value); //returns the output error gradient
};

#endif

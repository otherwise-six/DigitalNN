/*This class defines all methods needed to train a neural network.
* This will be a basic Gradient Descent trainer that uses
* batch learning and incorporates momentum.
*
* Author @ Alex vanKooten
* Version: 1.2 (02.23.2017)                                                   */

#include "NNTrainer.h"
#include <iostream>
#include <fstream>
#include <math.h>

/*default constructor*/
NNTrainer::NNTrainer(FFNNet *nn) : 
	neural_net(nn),					//the network to be trained		
	epoch(0),						//current epoch of training for the NN
	use_batch(false),				//flag to use batch learning
	max_epochs(MAX_EPOCHS),			//maximum number of allowable epochs for NN to be trained
	momentum(MOMENTUM),				//improves stochastic learning performance
	goal_acc(GOAL_ACC),				//desired MSE/accuracy the NN wants
	learning_rate(LEARNING_RATE),	//adjusts weight update	step size
	training_set_acc(0.0),			//training set accuracy per epoch
	training_set_MSE(0.0),			//training set MSE per epoch
	validation_set_acc(0.0),		//validation set accuracy per epoch
	validation_set_MSE(0.0),		//validation set MSE per epoch
	generalization_set_acc(0.0),	//generalization set accuracy per epoch
	generalization_set_MSE(0.0)		//generalization set MSE per epoch
	{	
		//init storage lists for the input-to-hidden delta values
		input_hidden_delta = new(double*[(neural_net->num_inputs) + 1]);
		for (int i = 0; i <= (neural_net->num_inputs); i++) {
			input_hidden_delta[i] = new (double[(neural_net->num_hidden)]);
			for (int j = 0; j < (neural_net->num_hidden); j++) {
				input_hidden_delta[i][j] = 0;
			}
		}

		//init storage lists for the hidden-to-output delta values
		hidden_output_delta = new(double*[(neural_net->num_hidden) + 1]);
		for (int m = 0; m <= (neural_net->num_hidden); m++) {
			hidden_output_delta[m] = new (double[(neural_net->num_outputs)]);
			for (int n = 0; n < (neural_net->num_outputs); n++) {
				hidden_output_delta[m][n] = 0;
			}
		}

		hidden_err_gradients = new(double[(neural_net->num_hidden) + 1]);
		for (int p = 0; p <= (neural_net->num_hidden); p++) {	//init storage for hidden error gradients 
			hidden_err_gradients[p] = 0;
		}

		output_err_gradients = new(double[(neural_net->num_outputs) + 1]);
		for (int q = 0; q <= (neural_net->num_outputs); q++) { //init storage for output error gradients 
			output_err_gradients[q] = 0;
		}
};

/*set training variables that aren't defaulted above*/
void NNTrainer::setTrainingVariables(double learn_rate, double mo, bool batch) {
	learning_rate = learn_rate;
	momentum = mo;
	use_batch = batch;
};

/*set up training stopping conditions*/
void NNTrainer::setStopConditions(int m_epochs, double g_acc) {
	max_epochs = m_epochs;
	goal_acc = g_acc;
};

/*enable the training log*/
void NNTrainer::enableLog(const char* file_name, int resolution = 1) {
	//create log file
	if (!log_file.is_open()) {
		log_file.open(file_name, std::ios::out);

		if (log_file.is_open()) {
			//write log file header
			log_file << "Epoch  #, Training Set Acc, Generalization Set Acc, Training Set MSE, Generalization Set MSE" << std::endl;
			log_enabled = true; //enable the training log

			//resolution setting;
			log_resolution = resolution;
			last_logged_epoch = -resolution;
		}
	}
};

/*returns the output erorr gradient*/
inline double NNTrainer::getOutputErrGradient(double target_value, double output_value) {
	return (output_value * ((1 - output_value) * (target_value - output_value)));
};

/*returns the input error gradient*/
double NNTrainer::getHiddenErrGradient(int target) {
	double weighted_sum = 0; //sum (hidden-to-output weights * output error gradients)
	for (int i = 0; i < neural_net->num_outputs; i++) {
		weighted_sum += neural_net->hidden_output_weights[target][i] * output_err_gradients[i];
	}
	return (weighted_sum * (neural_net->hidden_neurons[target] * (1 - neural_net->hidden_neurons[target])));
};

/*train the NN using gradient descent and incorporating momentum*/
void NNTrainer::trainNNet(trainDataSet* t_set) {
	epoch = 0; //reset the # of epochs
	last_logged_epoch = -log_resolution; //reset the last logged epoch
	printHeader(); //print a header for this run of training

	/*train the FFNN on the training dataset and test it with the generalization dataset*/
	while ((training_set_acc < goal_acc || generalization_set_acc < goal_acc) && epoch < max_epochs) {
		double last_train_acc = training_set_acc; //track last training accuracy 
		double last_general_acc = generalization_set_acc; //track last generalization accuracy
		runSingleEpoch(t_set->training_set); //train the network on a training set

		generalization_set_acc = neural_net->getDatasetAcc(t_set->generalization_set); //get gen set acc 
		generalization_set_MSE = neural_net->getDatasetMSE(t_set->generalization_set); //get gen set MSE 

		//if the training log is enabled, record the results~!
		if (log_enabled && log_file.is_open() && ((epoch - last_logged_epoch) == log_resolution)) {
			log_file << epoch << "," << training_set_acc << "," << generalization_set_acc << "," << training_set_MSE << "," << generalization_set_MSE << std::endl;
			last_logged_epoch = epoch;
		}

		//if training or generalization accuracy improve more than 1%; print
		if ((ceil(last_general_acc) != ceil(generalization_set_acc)) || (ceil(last_train_acc) != ceil(training_set_acc))) {
				std::cout << "Epoch: " << epoch << std::endl;
				std::cout << " Training Set Acc:" << training_set_acc << "%, MSE: " << training_set_MSE << std::endl;
				std::cout << " Generalization Set Acc:" << generalization_set_acc << "%, MSE: " << generalization_set_MSE << std::endl;
		}
		epoch++; //increment the epoch counter
	}

	validation_set_acc = neural_net->getDatasetAcc(t_set->validation_set); //get validation set acc
	validation_set_MSE = neural_net->getDatasetMSE(t_set->validation_set); //get validation set MSE

	/*wrap up the log file*/
	log_file << epoch << "," << training_set_acc << "," << generalization_set_acc << "," << training_set_MSE << "," << generalization_set_MSE << std::endl;
	log_file << std::endl << "NN Training Completed!" << std::endl;
	log_file << "Number of Epochs: " << epoch << std::endl;
	log_file << "Validation Set Accuracy: " << validation_set_acc << std::endl;
	log_file << "Validation Set MSE: " << validation_set_MSE << std::endl;

	/*print validation acc and MSE to console as well*/
	std::cout << "NN Training Completed!" << std::endl;
	std::cout << "Number of Epochs: " << epoch << std::endl;
	std::cout << "Validation Set Accuracy: " << validation_set_acc << std::endl;
	std::cout << "Validation Set MSE: " << validation_set_MSE << std::endl;
};

void NNTrainer::runSingleEpoch(std::vector<dataSet*> t_set) {
	double num_bad_patterns = 0; //clear the incorrect patterns
	double mse = 0; //clear mean squared error

	for (int i = 0; i < (int)t_set.size(); i++) { //iterate through input_data
		neural_net->feedForward(t_set[i]->input_data); //feed inputs through the network
		backprop(t_set[i]->target); //backpropogate errors
		bool correct_pattern = true; //set the correct pattern flag

		//compare target values to the output values from the NN
		for (int j = 0; j < neural_net->num_outputs; j++) { //iterate through outputs
			if ((neural_net->squashOutput(neural_net->output_neurons[j])) != (t_set[i]->target[j])) {
				correct_pattern = false; //if the output =/= the target output it's incorrect!
			}
			mse += pow((neural_net->output_neurons[j] - t_set[i]->target[j]), 2); //calc mse
		}
		if (!correct_pattern) { //check correctness
			num_bad_patterns++; //inc # of incorrect patterns if input data is a dud
		}
	}

	if (use_batch) { //if we're using batch learning
		updateWeights(); //update the weights right away
	}

	//update training acc and MSE
	training_set_acc = (100 - ((num_bad_patterns / t_set.size()) * 100));
	training_set_MSE = (mse / (neural_net->num_outputs * t_set.size()));
};

/*backprop the error through the NN to get the delta values*/
void NNTrainer::backprop(double* goal_outputs) {
	//1. change deltas between hidden layers and output layers
	for (int o = 0; o < neural_net->num_outputs; o++) { //1.1 loop through all output neurons
		output_err_gradients[o] = getOutputErrGradient(goal_outputs[o], neural_net->output_neurons[o]); //1.2 get output node erorr gradient
		for (int h = 0; h <= neural_net->num_hidden; h++) { //1.3 loop through all hidden neurons (and bias neuron)
			if (!use_batch) { //if we're incorporating momentum
				hidden_output_delta[h][o] = (hidden_output_delta[h][o] * momentum) 
					+ ((neural_net->hidden_neurons[h] * output_err_gradients[o]) * learning_rate); //1.4 update the hidden-to-output deltas 
			} else { //if we're using batch learning
				hidden_output_delta[h][o] += ((neural_net->hidden_neurons[h] * output_err_gradients[o]) * learning_rate); //1.4 (alt) sum the hidden-to-output deltas 
			}
		}
	}

	//2. change deltas between input layers and hidden layers
	for (int h = 0; h < neural_net->num_outputs; h++) { //2.1 loop through all hidden neurons
		output_err_gradients[h] = getOutputErrGradient(goal_outputs[h], neural_net->output_neurons[h]); //2.2 get hidden node erorr gradien
		for (int i = 0; i <= neural_net->num_hidden; i++) { //2.3 loop through all input neurons (and bias neuron)
			if (!use_batch) { //if we're incorporating momentum
				hidden_output_delta[i][h] = (hidden_output_delta[i][h] * momentum)
					+ ((neural_net->hidden_neurons[i] * output_err_gradients[h]) * learning_rate); //2.4 update the input-to-hidden deltas 
			} else { //if we're using batch learning
				hidden_output_delta[i][h] += ((neural_net->hidden_neurons[i] * output_err_gradients[h]) * learning_rate); //2.4 (alt) sum the input-to-hidden deltas 
			}
		}
	}

	if (use_batch) { //3. if we're using batch learning
		updateWeights(); //update the weights right away
	}
};

/*update weights using delta values*/
void NNTrainer::updateWeights() {
	//1. update the input-to-hidden layer weights
	for (int i = 0; i <= neural_net->num_inputs; i++) {
		for (int h = 0; h <= neural_net->num_hidden; h++) {
			neural_net->input_hidden_weights[i][h] += input_hidden_delta[i][h]; //update weights
			if (use_batch) { //if we're using batch learning
				input_hidden_delta[i][h] = 0; //clear the delta (we don't need it for momentum)
			}
		}
	}

	//2. update the hidden-to-output layer weights
	for (int h = 0; h <= neural_net->num_hidden; h++) {
		for (int o = 0; h <= neural_net->num_outputs; o++) {
			neural_net->input_hidden_weights[h][o] += input_hidden_delta[h][o]; //update weights
			if (use_batch) { //if we're using batch learning
				input_hidden_delta[h][o] = 0; //clear the delta (we don't need it for momentum)
			}
		}
	}
};

/*print a header to console with training info*/
void NNTrainer::printHeader() {
	std::cout << "\nFeed-Forward Neural Net Training Initialized: \n"
		<< "********************************************************************************"
		<< "Learning Rate: " << learning_rate << ", Momentum: " << momentum << ", Max Epochs: " << max_epochs << "\n"
		<< "Input Neurons: " << neural_net->num_inputs << " Hidden Neurons: " 
			<< neural_net->num_hidden << " Output Neurons: " << neural_net->num_outputs << "\n"
		<< "*******************************************************************************" << "*\n";
};
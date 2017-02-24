/*Will scan in data from CSV files and create a Training Data Set. 
* This allows easy access and organization of the data items for use in a 
* neural network.
*
* Author @ Alex vanKooten
* Version: 1.0 (02.21.2017)                                                   */

#include "Scanner.h"
#include <string>
#include <math.h>
#include <fstream>
#include <iostream>
#include <algorithm>

/*default destructor*/
scanner::~scanner() {
	//clear all data
	for (int i = 0; i < (int)data.size(); i++) {
		delete data[i];
	}
	data.clear();
};

/*load in the data from a file in csv format*/
bool scanner::loadDataFile(const char* file_name, int load_num_inputs, int load_num_targets) {
	for (int i = 0; i < (int)data.size(); i++) {
		delete data[i]; //clear individual data
	}
	data.clear(); //clear data
	current_data_set.clear();  //clear the current data

	num_inputs = load_num_inputs; //set # of inputs
	num_targets = load_num_targets; //set # of outputs

	//open the csv data file to read it
	std::fstream input_file;
	input_file.open(file_name, std::ios::in);

	if (input_file.is_open()) { //if 
		std::string line = "";

		//read data while not at end of line
		while (!input_file.eof()) {
			getline(input_file, line);
			if (line.length() > 2) { //read the line
				readLine(line);
			}
		}

		//shuffle data (helps our NN to not memorize)
		random_shuffle(data.begin(), data.end());

		//split data set
		training_data_end_index = (int)(0.65 * data.size());
		int generalization_size = (int)(ceil(0.15 * data.size()));
		int validation_size = (int)(data.size() - training_data_end_index - generalization_size);

		//generalization set
		for (int i = training_data_end_index; i < training_data_end_index + generalization_size; i++) {
			current_data_set.generalization_set.push_back(data[i]);
		}

		//validation set
		for (int i = training_data_end_index + generalization_size; i < (int)data.size(); i++) {
			current_data_set.validation_set.push_back(data[i]);
		}

		//show successful read status
		std::cout << "File Name: " << file_name << ".\nRead Complete: " 
			<< data.size() << " Input Patterns Loaded.\n";
		input_file.close(); //close the file 
		return true; //stuff worked!
	} else { //something went wrong and the file couldn't be opened
		std::cout << "An error occurred attempting to open: " << file_name << "\n";
		return false;
	}
};


/*read through a single line in the data file (for us this will be a single "digit")*/
void scanner::readLine(std::string &line) {
	//create new input pattern and target output
	double* input_pattern = new double[num_inputs];
	double* target_output = new double[num_targets];

	//store the inputs		
	char* char_string = new char[line.size() + 1];
	char* token;
	strcpy_s(char_string, line.size() + 1, line.c_str());

	//tokenize
	int i = 0;
	char* nextToken = NULL;
	token = strtok_s(char_string, ",", &nextToken);

	while ((token != NULL) && (i < (num_inputs + num_targets))) {
		if (i < num_inputs) {
			input_pattern[i] = atof(token);
		}
		else {
			target_output[i - num_inputs] = atof(token);
		}

		//move token forward (token++)
		token = strtok_s(NULL, ",", &nextToken);
		i++;
	}

	/*std::cout << "INPUT_PATTERNS: ";	//DEBUGGIN'
	for (int i=0; i < num_inputs; i++) {
		std::cout << input_pattern[i] << ",";
	}
	cout << "\nTARGET_OUTPUTS: ";
	for (int i=0; i < num_targets; i++) {
		std::cout << target_output[i] << " ";
	}
	std::cout << endl;*/

	//add the read line to a data set!
	data.push_back(new dataSet(input_pattern, target_output));

};

//returns the number of data sets created
int scanner::getNumTrainingSets() {
	return num_training_sets;
};

/*determines how the data sets are going to be partitioned for testing*/
void scanner::setPartitionMethod(int method, double var_a, double var_b) {
	//static, singular training set
	if (method == STATIC) {
		partition_method = STATIC;
		num_training_sets = 1;
	}

	//growing, training set size begins small an dincreases until full training set used
	else if (method == GROW) {
		if (var_a <= 100.0 && var_b > 0) {
			partition_method = GROW;

			//step size
			grow_step_size = var_a / 100;
			grow_last_data_index = 0;

			//number of sets
			num_training_sets = (int)ceil(1 / grow_step_size);
		}
	}

	//windowing, a set "sliding window" of training data will be used
	else if (method == WINDOW) {
		//window size must be smaller than all the training data
		//and the step size must be smaller than window size
		if ((var_a < data.size()) && (var_b <= var_a)) {
			partition_method = WINDOW;

			//
			window_size = (int)var_a;		//set the window size
			window_step_size = (int)var_b;	//set the window step size
			window_start_index = 0;			//set where the window will begin

			//number of sets
			num_training_sets = (int)(ceil( (double)(training_data_end_index - window_size) / window_step_size ) + 1);
		}
	}
};

//returns the data set created by a selected method
trainDataSet* scanner::getTrainingDataSet() {
	switch (partition_method) {
		case STATIC: 
			createStaticDataSet();
			break;
		case GROW: 
			createGrowDataSet(); 
			break;
		case WINDOW: 
			createWindowDataSet(); 
			break;
		default:
			break;
	}

	return &current_data_set;
};

//returns all data entries
std::vector<dataSet*>& scanner::getAllData(){
	return data;
};

//creates a static data set (default partition method)
void scanner::createStaticDataSet() {
	for (int i = 0; i < training_data_end_index; i++) {
		current_data_set.training_set.push_back(data[i]); //use only the training set data
	}
};

//create a growing data set (uses a small window which expands until all training data is encompassed)
void scanner::createGrowDataSet() {
	//data set increases by a percentage of the grow step
	grow_last_data_index += (int)ceil(grow_step_size * training_data_end_index);

	if (grow_last_data_index > ((int)training_data_end_index)) {
		grow_last_data_index = training_data_end_index; //can only grow as large as training data!
	}
	
	current_data_set.training_set.clear(); //clear training set
	for (int i = 0; i < grow_last_data_index; i++) { //set training set
		current_data_set.training_set.push_back(data[i]);
	}
};

/*create a windowed data set (slides a window over training data until it hits the end*/
void scanner::createWindowDataSet() {
	//create the end point of the window
	int window_end_index = window_start_index + window_size;

	if (window_end_index > training_data_end_index) {
		window_end_index = training_data_end_index; //must end when training data does!
	}

	current_data_set.training_set.clear(); //clear training set
	for (int i = window_start_index; i < window_end_index; i++) { //set the training set
		current_data_set.training_set.push_back(data[i]);
	}

	//slide the window forward
	window_start_index += window_step_size;
};
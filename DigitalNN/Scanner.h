/*Define all the methods we'll need to scan in information from CSV files.
* The Scanner will also create a Total Data Set to allow easy access and 
* organization of the data items.
*
* Author @ Alex vanKooten
* Version: 1.0 (02.21.2017)                                                   */

#pragma once
#ifndef SCANNER
#define SCANNER
#include<vector>
#include <string>

/*defines and stores what constitutes as a set of data*/
class dataSet {

public:
	double* input_data;	//will hold all the data values
	double* target;		//will hold target values

	//default contructor
	dataSet(double* data_in, double* target_in) :
		input_data(data_in), target(target_in) {}

	//default destructor
	~dataSet() {
		delete[] input_data;
		delete[] target;
	}
};

/*this will store our classified data sets (training, validation and testing)*/
class trainDataSet {

public:
	std::vector<dataSet*> training_set;
	std::vector<dataSet*> validation_set;
	std::vector<dataSet*> generalization_set;

	trainDataSet() {
		//empty default contructor
	};
		
	void clear() { //clear all data set categories
		training_set.clear();
		validation_set.clear();
		generalization_set.clear();
	}

};

//enum for advanced partitioning methods
enum {
	NONE,	//no special partitioning applied to the training set
	STATIC,	//train on a fixed subset of the training set
	GROW,	//train on a growing subset of the training set
	WINDOW	//train on a sliding window over the training set
};	

/*scanner class with all of it's methods to get us our data*/
class scanner {

private:
	int num_inputs;		//specified number of inputs
	int num_targets;	//specified number of targets

	int partition_method;	//how we'll create a training data set
	int num_training_sets;	//number of training data sets we need to make
	int training_data_end_index; //line at which the training data ends

	std::vector<dataSet*> data; //stores the data
	trainDataSet current_data_set; //the set we're currently working on

	double grow_step_size;		//training set will increase by % of total size
	int grow_last_data_index;	//where the current data set stops

	int window_size;		//size of each window
	int window_step_size;	//how many entries we move the window each iteration
	int window_start_index;	//where the current window starts

public:
	scanner() : partition_method(NONE), num_training_sets(-1) {} //default constructor
	~scanner(); //default destructor

	bool loadDataFile(const char* file_name, int load_num_inputs, int load_num_targets); //loads in data
	void setPartitionMethod(int method, double var_a = -1, double var_b = -1); //sets the training data partition type
	int getNumTrainingSets(); //returns the number of training sets

	trainDataSet* getTrainingDataSet();		//returns the training data set
	std::vector<dataSet*>& getAllData();	//returns full data set

private:
		void createStaticDataSet();	//create a static data set
		void createGrowDataSet();	//create a growing data set
		void createWindowDataSet(); //create a window data set
		void readLine(std::string &line); //read a line of data
};

#endif
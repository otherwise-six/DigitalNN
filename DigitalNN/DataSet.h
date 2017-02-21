/*Will define what constitutes as a set of data for use with the Neural Net. 
* Basically a node-like structure without full OO baggage.
*
* Author @ Alex vanKooten
* Version: 1.0 (02.21.2017)                                                   */

#pragma once
#ifndef DATASET
#define DATASET
#include <vector>
#include <string>

class dataSet {		//this will store a set of data
	double* data;	//will hold all the data values
	double* target;	//will hold target values

	//default contructor
	dataSet(double* data_in, double* target_in) :
		data(data_in), target(target_in) {}

	//default destructor
	~dataSet() {
		delete[] data;
		delete[] target;
	}
};
#endif


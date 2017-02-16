/* Header File for Neuron.cpp.
*
* PORBABLY GETTING DEPRICATED
*
* Author @ Alex vanKooten
* Version: 1.0 (02.15.2017)                                                   */

#pragma once
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#define _USE_MATH_DEFINES //for mathematical e
#include <math.h>		//for powers
#include <Windows.h>	//for sleep function
#include <iomanip>		//for setprecision
#include <limits>		//for dbl accuracy

typedef std::numeric_limits< double > dbl;
using namespace std;

class Neuron {
private:
	vector<Neuron> inputs;	//Neurons connected to the input layer
	vector<Neuron> outputs;	//Neurons connected to the output layer
	vector<double> wgtList; //list of output connection weights
	vector<double> deltaWgtList; //list of weight delts to be used in backprop
	double e;		//summation of all (input values * input weights)
	double weight;	//weight of the Neuron
	double errCont; //error contribution of the Neuron (for back prop)
	double value;	//the value of the Neuron 
	string name;	//unique Neuron name (not positive I need this)

public:
	//vector<Neuron> inputs;	//Neurons connected to the input layer
	//vector<Neuron> outputs;	//Neurons connected to the output layer
	Neuron(); //default constructor
	~Neuron(); //default deconstructor
	Neuron(string); //constructor with random weight (-1 to 1) and default value of 0
	Neuron(string, int); //constructor with random weight (-1 to 1)
	Neuron(string, double, int); //basic contructor
	void setIO(vector<Neuron> in, vector<Neuron> out);
	void addInput(Neuron);
	void addInputVector(vector<Neuron>);
	void addOutput(Neuron);
	void addOutputVector(vector<Neuron>);
	void setE(int);
	void setWeight(int, double);
	void setDeltaWeight(int, double);
	void setErrCont(double);
	void randWeights(int);
	void setName(string);
	void setValue(double);
	int getNumInputs();
	vector<Neuron> getInputVector();
	int getNumOutputs();
	vector<Neuron> getOutputVector();
	vector<double> getWeightVector();
	Neuron getInput(int);
	Neuron getOutput(int);
	double getE();
	int getNumWeights();
	double getWeight(int);
	double getDeltaWeight(int);
	double getErrCont();
	string getName();
	double getValue();
};


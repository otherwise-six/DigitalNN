/* Neuron class to act as a neuron in the neural network.
*
* PROBABLY GETTING DEPRICATED
*
* Author @ Alex vanKooten
* Version: 2.2 (02.15.2017)                                                   */

#include "Neuron.h"

/*default constructor*/
Neuron::Neuron() {
	//Neuron stuffs!
}

/*default destructor*/
Neuron::~Neuron() {
	//destroy the Neuron!
}

/*create a Neuron with random initial and default startValue*/
Neuron::Neuron(string startName) {
	name = startName;
	value = 0;
	randWeights(0);
	errCont = 0;
	e = 0;
}

/*create a Neuron with random initial weight*/
Neuron::Neuron(string startName, int startValue) {
	name = startName;
	value = startValue;
	randWeights(0);
	errCont = 0;
	e = 0;
}

/*create a Neuron with basic info*/
Neuron::Neuron(string startName, double startWeight, int startValue) {
	name = startName;
	weight = startWeight;
	value = startValue;
	errCont = 0;
	e = 0;
}

/*assign inputs and outputs to Neurons*/
void Neuron::setIO(vector<Neuron> in, vector<Neuron> out) {
	for (int i = 0; i < (int)in.size(); i++) {
		addInput(in[i]);
	}
	for (int j = 0; j < (int)out.size(); j++) {
		addOutput(out[j]);
	}
} //setIO

  /*add an input to a Neuron*/
void Neuron::addInput(Neuron i) {
	inputs.push_back(i);
} //addInput

  /*add a vector of inputs to a Neuron*/
void Neuron::addInputVector(vector<Neuron> in) {
	inputs = in;
} //addInputVector

  /*add an output to a Neuron*/
void Neuron::addOutput(Neuron o) {
	outputs.push_back(o);
} //addOutput

  /*add an output to a Neuron*/
void Neuron::addOutputVector(vector<Neuron> out) {
	outputs = out;
} //addOutputVector

void Neuron::setE(int n) {
	for (int i = 0; i < inputs.size(); i++) {
		e += inputs[i].getWeight(n) * inputs[i].getValue();
	}
} //setE

  /*set (or reset) a connections weight value*/
void Neuron::setWeight(int i, double newWgt) {
	wgtList[i] = newWgt;
	//cout << "setwgt = " << newWgt << endl;
} //setWeight

  /*set (or reset) a connections delta weight value*/
void Neuron::setDeltaWeight(int i, double newDWgt) {
	deltaWgtList[i] = newDWgt;
} //setDeltaWeight

  /*set (or reset) a Neuron's error contribution*/
void Neuron::setErrCont(double e) {
	errCont = e;
} //setErrCont

  /*assigns a random weight value between -1 and 1*/
void Neuron::randWeights(int j) {
	for (int i = 0; i < j; i++) {
		weight = ((rand() % 1000) - 500) / 1000.00;
		while ((-0.05 < weight) & (weight < 0.05)) {
			weight = ((rand() % 1000) - 500) / 1000.00;
		}
		wgtList.push_back(weight);
		deltaWgtList.push_back(0);
		//cout << "wgtList[" << i << "] " << wgtList[i] << endl;
	}
} //randWeight

  /*set the Neuron's name*/
void Neuron::setName(string newName) {
	name = newName;
} //setName

  /*sets the value of a Neuron*/
void Neuron::setValue(double v) {
	value = v;
} //setValue

  /*returns the number of inputs attached to a Neuron*/
int Neuron::getNumInputs() {
	return inputs.size();
} //getNumInputs

  /*returns the Neurons input vector*/
vector<Neuron> Neuron::getInputVector() {
	return inputs;
} //getInputVector

  /*returns the number of outputs attached to a Neuron*/
int Neuron::getNumOutputs() {
	return outputs.size();
} //getNumOutputs

  /*returns the Neurons output vector*/
vector<Neuron> Neuron::getOutputVector() {
	return outputs;
} //getOutputVector

  /*returns a Neuron's current weight vector*/
vector<double> Neuron::getWeightVector() {
	return wgtList;
} //getWeightVector

  /*returns a specified input Neuron*/
Neuron Neuron::getInput(int i) {
	return inputs[i];
} //getInput

  /*returns a specified output Neuron*/
Neuron Neuron::getOutput(int o) {
	return outputs[o];
} //getIntputVector

  /*returns the summation of the all input weights and values*/
double Neuron::getE() {
	return e;
} //getE

  /*returns the number of weighted connection tails attached to a Neuron*/
int Neuron::getNumWeights() {
	return wgtList.size();
} //getWeight

  /*returns a Neuron's current weight value*/
double Neuron::getWeight(int i) {
	return wgtList[i];
} //getWeight

  /*returns a Neuron's delta weight value*/
double Neuron::getDeltaWeight(int i) {
	return deltaWgtList[i];
} //getDeltaWeight

  /*get a Neuron's error contribution*/
double Neuron::getErrCont() {
	return errCont;
} //getErrCont

  /*returns the value of a Neuron*/
double Neuron::getValue() {
	return value;
} //getValue

  /*returns a Neuron's name*/
string Neuron::getName() {
	return name;
} //getName

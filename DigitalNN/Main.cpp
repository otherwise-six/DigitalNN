/*DigitalNN will use a Feed Forward Neural Network (FFNNet) and a 
* Neural Net Trainer (NNTrainer) to recognize basic digits (0 to 9).
*
* A small dataset of around 700 training integers have been provided.
*
* What I'll be doing first is a feed forward neural network based on
* backpropagation learning with momentum.
*
* The learning rate, momentum and number of hidden nodes need to be variable
* so the network performance under different conditions can be recorded.
* The activation function will need to be toggle-able between the logistic
* and tanh functions.
*
* Eventually both "holdout" and "cross validation" techniques will be 
* used on the testing data.
*
* Author @ Alex vanKooten
* Version: 1.0 (02.24.2017)                                                   */

#include "FFNNet.h"
#include "NNTrainer.h"
#include <ctime>
#include <iostream>

void main() {
	srand((unsigned int)(time(0))); //seed our rand # generator

	scanner scan; //make a scanner for our data input
	scan.loadDataFile("letter-recognition-2.csv", 16, 3); //load data
	scan.setPartitionMethod(STATIC, 10); //set how data is broken up

	FFNNet net(16, 10, 3); //make the NN (# inputs, # hidden, # outputs)
	NNTrainer trainer(&net); //make the NN Trainer
	trainer.enableLog("log.csv", 5); //(log name, log resolution)
	trainer.setStopConditions(150, 90); //(max # epochs, goal % acc) 
	trainer.setTrainingVariables(0.001, 0.9, false); //(learning rate, momentum, batch learning on)

	for (int i = 0; i < scan.getNumTrainingSets(); i++) {
		trainer.trainNNet(scan.getTrainingDataSet()); //use data sets to train the NN
	}

	net.saveWeights("weights.csv"); //save the final weights
	std::cout << "\nNeural Net has been Trained!\n"; //let the user know
};
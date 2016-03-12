/*
 * neural_network.h
 *
 *  Created on: Oct 18, 2015
 *      Author: matt
 */

#ifndef NAVIGATION_INCLUDE_NAVIGATION_NEURAL_NETWORK_H_
#define NAVIGATION_INCLUDE_NAVIGATION_NEURAL_NETWORK_H_

#include <ros/ros.h>
#include <cmath>

class NeuralNetwork
{
private:
    int m_num_input;
    int m_num_hidden;
    int m_num_output;

    //Input to hidden arrays
    double* inputs;
    double** ih_weights;
    double* ih_sums;
    double* ih_biases;
    double* ih_outputs;

    //Output to hidden arrays
    double** ho_weights;
    double* ho_sums;
    double* ho_biases;
    double* outputs;

    double* o_grads; //output gradients
    double* h_grads; //hidden gradients

    //Momentum arrays
    double** ih_prev_weights_delta;
    double* ih_prev_biases_delta;
    double** ho_prev_weights_delta;
    double* ho_prev_biases_delta;

    double sigmoidFunction(double x);
    double hyperTanFunction(double x);

public:
    NeuralNetwork(int num_input, int num_hidden, int num_output);
    ~NeuralNetwork();

    void updateWeights(double weights[]);
    void setWeights(double weights[]);

    double* getWeights();
    double* computeOutputs();
};

#endif /* NAVIGATION_INCLUDE_NAVIGATION_NEURAL_NETWORK_H_ */

/*
 * neural_network.cpp
 *
 *  Created on: Oct 18, 2015
 *      Author: matt
 */

#include "neural_network.h"

double NeuralNetwork::sigmoidFunction(double x)
{
    double ret;

    if(x < -45.0)
        ret = 0.0;
    else if (x > 45.0)
        ret = 1.0;
    else
        ret = 1.0 / (1.0 + exp(-x));

    return ret;
}
double NeuralNetwork::hyperTanFunction(double x)
{
    double ret;

    if(x < -10.0)
        ret = -1.0;
    else if(x > 10.0)
        ret = 1.0;
    else
        ret = tanh(x);

    return ret;
}

NeuralNetwork::NeuralNetwork(int num_input, int num_hidden, int num_output)
{
    //Set member parameter variables
    m_num_input = num_input;
    m_num_hidden = num_hidden;
    m_num_output = num_output;

    inputs = new double[num_input];

    //Allocate memory for input to hidden arrays
    ih_weights = new double*[num_input];
    for(int i = 0; i < num_input; i++)
    {
        ih_weights[i] = new double[num_hidden];
    }
    ih_sums = new double[num_hidden];
    ih_biases = new double[num_hidden];
    ih_outputs = new double[num_output];

    //Allocate memory for hidden to output arrays
    ho_weights = new double*[num_hidden];
    for(int i = 0; i < num_hidden; i++)
    {
        ho_weights[i] = new double[num_output];
    }
    ho_sums = new double[num_output];
    ho_biases = new double[num_output];
    outputs = new double[num_output];

    //Allocate gradient array memory
    o_grads = new double[num_output];
    h_grads = new double[num_hidden];

    //Allocate momentum arrays
    ih_prev_weights_delta = new double*[num_input];
    for(int i = 0; i < num_input; i++)
    {
        ih_prev_weights_delta[i] = new double[num_hidden];
    }
    ih_prev_biases_delta = new double[num_hidden];
    ho_prev_weights_delta = new double*[num_hidden];
    for(int i = 0; i < num_hidden; i++)
    {
        ho_prev_weights_delta[i] = new double[num_output];
    }
    ho_prev_biases_delta = new double[num_output];
}

NeuralNetwork::~NeuralNetwork()
{
    delete[] inputs;
    for(int i = 0; i < m_num_input; i++)
    {
        delete[] ih_weights[i];
    }
    delete[] ih_weights;
    delete[] ih_sums;
    delete[] ih_biases;
    delete[] ih_outputs;

    for(int i = 0; i < m_num_hidden; i++)
    {
        delete[] ho_weights[i];
    }
    delete[] ho_weights;
    delete[] ho_sums;
    delete[] ho_biases;
    delete[] outputs;

    delete[] o_grads;
    delete[] h_grads;

    for(int i = 0; i < m_num_input; i++)
    {
        delete[] ih_prev_weights_delta[i];
    }
    delete[] ih_prev_weights_delta;
    delete[] ih_prev_biases_delta;
    for(int i = 0; i < m_num_hidden; i++)
    {
        delete[] ho_prev_weights_delta[i];
    }
    delete[] ho_prev_weights_delta;
    delete[] ho_prev_biases_delta;
}


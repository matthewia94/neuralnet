/*
 * neural_network.cpp
 *
 *  Created on: Oct 18, 2015
 *      Author: matt
 */

#include "neural_network.h"

template <typename T>
double NeuralNetwork<T>::sigmoidFunction(T x)
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

template <typename T>
NeuralNetwork<T>::NeuralNetwork(int num_input, int num_hidden, int num_output, double learning_rate)
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
    ih_outputs = new double[num_output];

    //Allocate memory for hidden to output arrays
    ho_weights = new double*[num_hidden];
    for(int i = 0; i < num_hidden; i++)
    {
        ho_weights[i] = new double[num_output];
    }
    outputs = new double[num_output];

    //Initialize the weights randomly
    resetWeights();

    m_learning_rate = learning_rate;
}

template <typename T>
NeuralNetwork<T>::~NeuralNetwork()
{
    delete[] inputs;
    for(int i = 0; i < m_num_input; i++)
    {
        delete[] ih_weights[i];
    }
    delete[] ih_weights;
    delete[] ih_outputs;

    for(int i = 0; i < m_num_hidden; i++)
    {
        delete[] ho_weights[i];
    }
    delete[] ho_weights;
    delete[] outputs;
}

template <typename T>
void NeuralNetwork<T>::printInput()
{
    for(size_t i = 0; i < m_num_input; i++)
    {
        std::cout << inputs[i] << " ";
    }
    std::cout << std::endl;
}

template <typename T>
void NeuralNetwork<T>::printHidden()
{
    for(size_t i = 0; i < m_num_hidden; i++)
    {
        std::cout << ih_outputs[i] << " ";
    }
    std::cout << std::endl;
}

template <typename T>
void NeuralNetwork<T>::printOutput()
{
    for(size_t i = 0; i < m_num_output; i++)
    {
        std::cout << outputs[i] << " ";
    }
    std::cout << std::endl;
}

template <typename T>
void NeuralNetwork<T>::resetWeights()
{
    std::default_random_engine rng;
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    // calculate the output from the hidden layer
    for(size_t i = 0; i < m_num_hidden; i++)
    {
        for(size_t j = 0; j < m_num_input; j++)
        {
            ih_weights[j][i] = distribution(rng);
        }
    }

    // calculate the overall output
    for(size_t i = 0; i < m_num_output; i++)
    {
        for(size_t j = 0; j < m_num_hidden; j++)
        {
            ho_weights[j][i] = distribution(rng);
        }
    }
}

template <typename T>
std::istream& operator>>(std::istream& is, const NeuralNetwork<T>& n)
{
    for(size_t i = 0; i < n.m_num_input; i++)
    {
        is >> n.inputs[i];
    }

    return is;
}

template <typename T>
double* NeuralNetwork<T>::computeOutputs()
{
    // calculate the output from the hidden layer
    for(size_t i = 0; i < m_num_hidden; i++)
    {
        ih_outputs[i] = 0;
        for(size_t j = 0; j < m_num_input; j++)
        {
            ih_outputs[i] += inputs[j] * ih_weights[j][i];
        }
        ih_outputs[i] = sigmoidFunction(ih_outputs[i]);
    }

    // calculate the overall output
   for(size_t i = 0; i < m_num_output; i++)
   {
       outputs[i] = 0;
       for(size_t j = 0; j < m_num_hidden; j++)
       {
           outputs[i] += ih_outputs[j] * ho_weights[j][i];
       }
   }

    return outputs;
}

template <typename T>
void NeuralNetwork<T>::backPropagation(const T target_outputs[])
{
    T* error_output = new T[m_num_output];
    T* error_hidden = new T[m_num_hidden];
    T sum = 0;

    for(size_t i = 0; i < m_num_output; i++)
    {
        error_output[i] = outputs[i] * (1 - outputs[i]) * (target_outputs[i] - outputs[i]);
    }

    // Find hidden layer errors
    for(size_t i = 0; i < m_num_hidden; i++)
    {
        sum = 0;
        for(size_t j = 0; j < m_num_output; j++)
        {
           sum += error_output[j] + ih_weights[j][i];
        }
        error_hidden[i] = ih_outputs[i] * (1 - ih_outputs[i]) * sum;
    }

    // Change output weights
    for(size_t i = 0; i < m_num_output; i++)
    {
        for (size_t j = 0; j < m_num_hidden; j++)
        {
            ho_weights[j][i] = ho_weights[j][i] + m_learning_rate * error_output[i] * ih_outputs[j];
        }
    }

    for(size_t i = 0; i < m_num_hidden; i++)
    {
        for(size_t j = 0; j < m_num_input; j++)
        {
            ih_weights[j][i] = ih_weights[j][i] + m_learning_rate * error_hidden[i] * inputs[j];
        }
    }

    delete[] error_output;
    delete[] error_hidden;
}

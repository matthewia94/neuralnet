/*
 * neural_network.h
 *
 *  Created on: Oct 18, 2015
 *      Author: matt
 */

#ifndef NEURAL_NETWORK_H_
#define NEURAL_NETWORK_H_

#include <cstdlib>
#include <cmath>
#include <random>
#include <iostream>

template <typename T>
class NeuralNetwork;

template <typename T>
std::istream& operator>>(std::istream& is, const NeuralNetwork<T>& n);

template <typename T>
class NeuralNetwork
{
    private:
        int m_num_input;
        int m_num_hidden;
        int m_num_output;
        double m_learning_rate;

        //Input to hidden arrays
        T* inputs;
        T** ih_weights;
        T* ih_outputs;

        //Output to hidden arrays
        T** ho_weights;
        T* outputs;

        double sigmoidFunction(T x);

    public:
        NeuralNetwork(int num_input, int num_hidden, int num_output, double learning_rate);
        ~NeuralNetwork();

        //Print functions
        void printInput();
        void printHidden();
        void printOutput();

        //Input functions
        friend std::istream& operator>> <T>(std::istream& is, const NeuralNetwork<T>& n);

        void resetWeights();
        double* computeOutputs();
        void backPropagation(const T outputs[]);
};

#include "neural_network.hpp"

#endif /* NEURAL_NETWORK_H_ */

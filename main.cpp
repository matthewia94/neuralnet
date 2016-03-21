#include "neural_network.h"
#include <iostream>

int main(int argc, char** argv)
{
    NeuralNetwork<double> n(2, 3, 1, .9);
    double target[1] = {0};
    std::cin >> n;
    n.computeOutputs();
    n.printInput();
    n.printHidden();
    n.printOutput();

    while(n.computeOutputs()[0] - target[0] > .00001)
    {
        std::cout << n.computeOutputs()[0] << std::endl;
        n.backPropagation(target);
    }
}

#include "OutputLayer.h"


OutputLayer::OutputLayer(int size)
    : Layer(size, size)
{ }

void OutputLayer::set_prev_layer(Layer* prev_layer) {
    throw std::logic_error("Output layer cannot be used externally");
}

void OutputLayer::set_next_layer(Layer* next_layer) {
    throw std::logic_error("Output layer can only be the last layer");
}

void OutputLayer::backpropagate(const arma::vec* layer_error) {

}

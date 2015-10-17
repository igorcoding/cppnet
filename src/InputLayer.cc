#include "InputLayer.h"


InputLayer::InputLayer(int size)
    : Layer(size, size)
{ }

void InputLayer::set_prev_layer(Layer* prev_layer) {
    throw std::logic_error("Input layer can only be the first layer");
}

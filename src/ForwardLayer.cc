#include "ForwardLayer.h"

ForwardLayer::ForwardLayer(int input_size, int output_size)
    : Layer(input_size, output_size),
      _w(nullptr),
      _prev_out(output_size)
{
}

ForwardLayer::ForwardLayer(int output_size)
    : Layer(output_size),
      _w(nullptr),
      _prev_out(output_size) {

}

ForwardLayer::~ForwardLayer() {
    delete _w;
    _w = nullptr;
}

void ForwardLayer::init() {
    Layer::init();
    delete _w;
    _w = new arma::mat(_output_size, _input_size, arma::fill::randu);
}

const arma::vec* ForwardLayer::activate() {
    auto activation = prev_layer()->activate();
    if (activation == nullptr) {
        return nullptr;
    }
    _prev_out = (*_w) * (*activation);
    return &_prev_out;
}

void ForwardLayer::backpropagate(const arma::vec* layer_error) {

}

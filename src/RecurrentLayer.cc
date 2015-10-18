#include "RecurrentLayer.h"

// === RecurrentPartLayer === //

RecurrentLayer::RecurrentPartLayer::RecurrentPartLayer(int input_size, int output_size)
        : ForwardLayer(input_size, output_size),
          _context_w(nullptr)
{
    init();
}

RecurrentLayer::RecurrentPartLayer::~RecurrentPartLayer() {
    delete _context_w;
    _context_w = nullptr;
}

void RecurrentLayer::RecurrentPartLayer::init() {
    ForwardLayer::init();
    delete _context_w;
    _context_w = new arma::mat(_output_size, _output_size, arma::fill::randu);
}

const arma::vec* RecurrentLayer::RecurrentPartLayer::activate(const arma::vec& context) {
    auto prev_layer_activation = prev_layer()->activate();
    if (prev_layer_activation == nullptr) {
        return nullptr;
    }
    *_prev_out = (*_w) * (*prev_layer_activation) + (*_context_w) * context;
    *_prev_activation = sigma(_prev_out);
    return _prev_activation;
}

void RecurrentLayer::RecurrentPartLayer::backpropagate(const arma::vec* layer_error) {

}

void RecurrentLayer::RecurrentPartLayer::backpropagate_apply() {

}


// === RecurrentLayer === //

RecurrentLayer::RecurrentLayer(int output_size, int parts_count)
    : ForwardLayer(output_size),
      _parts(static_cast<size_t>(parts_count)),
      _context(output_size, arma::fill::zeros) {

    for (RecurrentPartLayer*& part : _parts) {
        part = nullptr;
    }
}

RecurrentLayer::~RecurrentLayer() {
    for (RecurrentPartLayer*& part : _parts) {
        delete part;
        part = nullptr;
    }
}

void RecurrentLayer::init() {
    ForwardLayer::init();

    for (int i = 0; i < _parts.size(); ++i) {
        delete _parts[i];
        _parts[i] = new RecurrentPartLayer(input_size(), output_size());
        _parts[i]->set_prev_layer(prev_layer());
    }
}

const arma::vec* RecurrentLayer::activate() {
    const arma::vec* context = &_context;
    for (RecurrentPartLayer*& part : _parts) {
        auto activation = part->activate(*context);
        if (activation == nullptr) {
            break;
        }
        context = activation;
    }
    _context = *context;
    *_prev_out = _context;
    *_prev_activation = sigma(_prev_out);
    return _prev_activation;
}

void RecurrentLayer::backpropagate(const arma::vec* layer_error) {

}

void RecurrentLayer::backpropagate_apply() {

}

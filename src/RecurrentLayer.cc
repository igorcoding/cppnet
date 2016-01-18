#include "RecurrentLayer.h"

// === RecurrentPartLayer === //

RecurrentLayer::RecurrentPartLayer::RecurrentPartLayer(int input_size, int output_size)
        : ForwardLayer(input_size, output_size),
          _context_w(nullptr),
          _context_dw(nullptr),
          _context_w_last_change(nullptr)
{
    init();
}

RecurrentLayer::RecurrentPartLayer::~RecurrentPartLayer() {
    delete _context_w;
    _context_w = nullptr;

    delete _context_dw;
    _context_dw = nullptr;

    delete _context_w_last_change;
    _context_w_last_change = nullptr;
}

void RecurrentLayer::RecurrentPartLayer::init() {
    ForwardLayer::init();
    delete _context_w;
    _context_w = new arma::mat(_output_size, _output_size, arma::fill::randu);

    delete _context_dw;
    _context_dw = new arma::mat(_output_size, _output_size, arma::fill::randu);

    delete _context_w_last_change;
    _context_w_last_change = new arma::mat(_output_size, _output_size, arma::fill::randu);
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
    const auto& prev_layer_activation = prev_layer()->prev_activation();
    const auto& prev_layer_out = prev_layer()->prev_out();

    const auto& prev_part_layer_activation = prev_part_layer()->prev_activation();
    const auto& prev_part_layer_out = prev_part_layer()->prev_out();

    if (is_output()) {
        auto error = *_prev_activation - *layer_error;
        arma::vec delta = error % dsigma(_prev_out);

        *_dw += delta * prev_part_layer_activation->t();
        *_db += delta;

        *_context_dw += delta * prev_part_layer_activation->t();
        ++_m;

        arma::vec delta_prev_layer = (_w->t() * delta) % dsigma(prev_part_layer_out);
        return prev_part_layer()->backpropagate(&delta_prev_layer);
    } else {
        *_dw += (*layer_error) * prev_part_layer_activation->t();
        *_db += *layer_error;

        *_context_dw += (*layer_error) * prev_part_layer_activation->t();
        ++_m;

        arma::vec delta_prev_layer = (_w->t() * (*layer_error)) % dsigma(prev_part_layer_out);
        return prev_part_layer()->backpropagate(&delta_prev_layer);
    }
}

void RecurrentLayer::RecurrentPartLayer::backpropagate_apply() {
    double learning_rate = 0.7;
    double regularization = 0.00001;
    double momentum = 0.8;

    if (_m > 0) {
        *_w_last_change = -learning_rate * (*_dw / _m + regularization * (*_w)) + momentum * (*_w_last_change);
        *_b_last_change = -learning_rate * (*_db / _m) + momentum * (*_b_last_change);
        *_context_w_last_change = -learning_rate * (*_context_dw / _m + regularization * (*_context_w)) + momentum * (*_context_w_last_change);

        *_w += *_w_last_change;
        *_b += *_b_last_change;
        *_context_w += *_context_w_last_change;
    }
    clean_dw();
    prev_layer()->backpropagate_apply();
}


void RecurrentLayer::RecurrentPartLayer::clean_dw() {
    ForwardLayer::clean_dw();
    _context_dw->fill(arma::fill::zeros);
}


// === RecurrentLayer === //

RecurrentLayer::RecurrentLayer(int output_size, int parts_count)
    : ForwardLayer(output_size),
      _parts(static_cast<size_t>(parts_count)),
      _context(output_size, arma::fill::zeros),
      _w(nullptr),
      _context_w(nullptr) {

    for (RecurrentPartLayer*& part : _parts) {
        part = nullptr;
    }
}

RecurrentLayer::~RecurrentLayer() {
    for (RecurrentPartLayer*& part : _parts) {
        delete part;
        part = nullptr;
    }

    delete _w;
    _w = nullptr;

    delete _context_w;
    _context_w = nullptr;
}

void RecurrentLayer::init() {
    ForwardLayer::init();

    for (int i = 0; i < _parts.size(); ++i) {
        delete _parts[i];
        _parts[i] = new RecurrentPartLayer(input_size(), output_size());
        _parts[i]->set_prev_layer(prev_layer());
        if (i > 0) {
            _parts[i]->set_prev_part_layer(_parts[i-1]);
        }
    }

    delete _w;
    _w = nullptr;
    delete _context_w;
    _context_w = nullptr;
}

void RecurrentLayer::set_as_output() {
    ForwardLayer::set_as_output();
    _parts.back()->set_as_output();
}

arma::vec RecurrentLayer::eval(const arma::vec& input) {
    arma::vec res = (*_w) * input + (*_context_w) * _context;
    return res;
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
    // do we need to clean context?
    _context = *context;
    *_prev_out = _context;
    *_prev_activation = sigma(_prev_out);
    return _prev_activation;
}

void RecurrentLayer::backpropagate(const arma::vec* layer_error) {
    _parts.back()->backpropagate(layer_error);
}

void RecurrentLayer::backpropagate_apply() {
    _parts.back()->backpropagate_apply();

    delete _w;
    delete _context_w;
    _w = new arma::mat(_output_size, _input_size, arma::fill::zeros);
    _context_w = new arma::mat(_output_size, _output_size, arma::fill::zeros);

    for (RecurrentPartLayer*& part : _parts) {
        *_w += *(part->_w);
        *_context_w += *(part->_context_w);
    }
    *_w /= _parts.size();
    *_context_w /= _parts.size();
}

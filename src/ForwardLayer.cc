#include "ForwardLayer.h"

ForwardLayer::ForwardLayer(int input_size, int output_size)
    : Layer(input_size, output_size),
      _w(nullptr),
      _b(nullptr),
      _dw(nullptr),
      _db(nullptr),
      _w_last_change(nullptr),
      _b_last_change(nullptr),
      _m(0)
{
}

ForwardLayer::ForwardLayer(int output_size)
    : ForwardLayer(-1, output_size) {

}

ForwardLayer::~ForwardLayer() {
    delete _w;
    _w = nullptr;

    delete _b;
    _b = nullptr;

    delete _dw;
    _dw = nullptr;

    delete _db;
    _db = nullptr;

    delete _w_last_change;
    _w_last_change = nullptr;

    delete _b_last_change;
    _b_last_change = nullptr;
}

void ForwardLayer::init() {
    Layer::init();
    delete _w;
    _w = new arma::mat(_output_size, _input_size, arma::fill::randu);
    *_w -= 0.5;

    delete _b;
    _b = new arma::vec(_output_size, arma::fill::randu);
    *_b -= 0.5;

    delete _dw;
    _dw = new arma::mat(_output_size, _input_size, arma::fill::zeros);

    delete _db;
    _db = new arma::vec(_output_size, arma::fill::zeros);

    delete _w_last_change;
    _w_last_change = new arma::mat(_output_size, _input_size, arma::fill::zeros);

    delete _b_last_change;
    _b_last_change = new arma::vec(_output_size, arma::fill::zeros);

    _m = 0;
}

arma::vec ForwardLayer::eval(const arma::vec& input) {
    arma::vec out = (*_w) * input + *_b;
    return sigma(&out);
}

const arma::vec* ForwardLayer::activate() {
    auto activation = prev_layer()->activate();
    if (activation == nullptr) {
        return nullptr;
    }
//    std::cout << *activation << std::endl;
    *_prev_out = (*_w) * (*activation) + *_b;
    *_prev_activation = sigma(_prev_out);
    return _prev_activation;
}

void ForwardLayer::backpropagate(const arma::vec* layer_error) {
    const auto& prev_layer_activation = prev_layer()->prev_activation();
    const auto& prev_layer_out = prev_layer()->prev_out();

    if (is_output()) {
        auto delta = *_prev_activation - *layer_error;
        *_dw += delta * prev_layer_activation->t();
        *_db += delta;
        ++_m;

        arma::vec delta_prev_layer = (_w->t() * delta) % dsigma(prev_layer_out);
        return prev_layer()->backpropagate(&delta_prev_layer);
    } else {
        auto& delta = *layer_error;
        *_dw += delta * prev_layer_activation->t();
        *_db += delta;
        ++_m;

        arma::vec delta_prev_layer = (_w->t() * delta) % dsigma(prev_layer_out);
        return prev_layer()->backpropagate(&delta_prev_layer);
    }
}

void ForwardLayer::backpropagate_apply() {
    double learning_rate = 0.1;
    double regularization = 0.00001;
    double momentum = 0.0;

    if (_m > 0) {
        *_w_last_change = -learning_rate * (*_dw / _m + regularization * (*_w)) + momentum * (*_w_last_change);
        *_b_last_change = -learning_rate * (*_db / _m) + momentum * (*_b_last_change);

        *_w += *_w_last_change;
        *_b += *_b_last_change;
    }
    clean_dw();
    prev_layer()->backpropagate_apply();
//    std::cout << "_w: " << *_w << std::endl;
}

void ForwardLayer::clean_dw() {
    _dw->fill(arma::fill::zeros);
    _db->fill(arma::fill::zeros);
    _m = 0;
}

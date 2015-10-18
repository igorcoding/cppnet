#ifndef CPPNET_LAYER_H
#define CPPNET_LAYER_H

#include <vector>
#include <armadillo>

class Layer {
public:

    Layer(int input_size, int output_size)
            : _input_size(input_size),
              _output_size(output_size),
              _init(false),
              _is_output(false),
              _prev_out(nullptr),
              _prev_activation(nullptr),
              _prev_layer(nullptr),
              _next_layer(nullptr)
    { }

    Layer(int output_size)
            : Layer(-1, output_size)
    { }

    virtual ~Layer() {
        delete _prev_out;
        _prev_out = nullptr;

        delete _prev_activation;
        _prev_activation = nullptr;
    }


    void input_size(int input_size) {
        _input_size = input_size;
        init();
    }

    void output_size(int output_size) {
        _output_size = output_size;
        init();
    }

    int input_size() const { return _input_size; }
    int output_size() const { return _output_size; }

    const arma::vec* prev_out() const { return _prev_out; }
    const arma::vec* prev_activation() const { return _prev_activation; }

    Layer* prev_layer() { return _prev_layer; }
    Layer* next_layer() { return _next_layer; }

    virtual void set_prev_layer(Layer* prev_layer) { _prev_layer = prev_layer; }
    void set_prev_layer_force(Layer* prev_layer) { _prev_layer = prev_layer; }
    virtual void set_next_layer(Layer* next_layer) { _next_layer = next_layer; }
    void set_next_layer_force(Layer* next_layer) { _next_layer = next_layer; }

    bool is_init() { return _init; }
    virtual void init() {
        if (_output_size > 0) {
            _prev_out = new arma::vec(_output_size, arma::fill::zeros);
            _prev_activation = new arma::vec(_output_size, arma::fill::zeros);
        }
        _init = true;
    }

    bool is_output() { return _is_output; }
    void set_as_output() { _is_output = true; }

    virtual arma::vec eval(const arma::vec& input) {
        if (_prev_layer != nullptr) {
            return _prev_layer->eval(input);
        } else {
            throw std::logic_error("eval() is not implemented");
        }
    }

    virtual const arma::vec* activate() {
        if (_prev_layer != nullptr) {
            return _prev_layer->activate();
        } else {
            throw std::logic_error("activate() is not implemented");
        }
    }

    virtual void backpropagate(const arma::vec* layer_error) {
        if (_prev_layer != nullptr) {
            return _prev_layer->backpropagate(layer_error);
        } else {
            throw std::logic_error("backpropagate() is not implemented");
        }
    }

    virtual void backpropagate_apply() {
        if (_prev_layer != nullptr) {
            return _prev_layer->backpropagate_apply();
        } else {
            throw std::logic_error("backpropagate_apply() is not implemented");
        }
    }

    arma::vec sigma(const arma::vec* v) {
        if (v == nullptr) return nullptr;
        return 1.0 / (1 + arma::exp(-(*v)));
    }

    arma::vec dsigma(const arma::vec* v) {
        if (v == nullptr) return nullptr;
        auto s = sigma(v);
        return s % (1 - s);
    }

protected:
    int _input_size;
    int _output_size;
    bool _init;
    bool _is_output;

    arma::vec*  _prev_out;
    arma::vec*  _prev_activation;

    Layer* _prev_layer;
    Layer* _next_layer;


private:
    Layer(const Layer& rhs) = delete;
    Layer& operator =(const Layer& rhs) = delete;
};

#endif //CPPNET_LAYER_H

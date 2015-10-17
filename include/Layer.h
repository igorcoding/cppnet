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
              _prev_layer(nullptr),
              _next_layer(nullptr)
    { }

    Layer(int output_size)
            : _input_size(-1),
              _output_size(output_size),
              _init(false),
              _prev_layer(nullptr),
              _next_layer(nullptr)
    { }

    virtual ~Layer() { }


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

    Layer* prev_layer() {
        return _prev_layer;
    }

    Layer* next_layer() {
        return _next_layer;
    }

    virtual void set_prev_layer(Layer* prev_layer) {
        _prev_layer = prev_layer;
    }

    virtual void set_prev_layer_force(Layer* prev_layer) {
        _prev_layer = prev_layer;
    }

    void set_next_layer(Layer* next_layer) {
        _next_layer = next_layer;
    }

    void set_next_layer_force(Layer* next_layer) {
        _next_layer = next_layer;
    }

    bool is_init() { return _init; }
    virtual void init() { _init = true; }

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

protected:
    int _input_size;
    int _output_size;
    bool _init;

    Layer* _prev_layer;
    Layer* _next_layer;


private:
    Layer(const Layer& rhs) = delete;
    Layer& operator =(const Layer& rhs) = delete;
};

#endif //CPPNET_LAYER_H

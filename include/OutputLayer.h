#ifndef CPPNET_OUTPUTLAYER_H
#define CPPNET_OUTPUTLAYER_H

#include "Layer.h"

#include <armadillo>

class OutputLayer : public Layer {
public:
    OutputLayer(int size);

    virtual void set_prev_layer(Layer* prev_layer) override;
    virtual void set_next_layer(Layer* next_layer) override;
    virtual void backpropagate(const arma::vec* layer_error) override;
};

#endif //CPPNET_OUTPUTLAYER_H

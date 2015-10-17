#ifndef CPPNET_INPUTLAYER_H
#define CPPNET_INPUTLAYER_H

#include "Layer.h"

#include <armadillo>

class InputLayer : public Layer {
public:
    InputLayer(int size);

    virtual void set_prev_layer(Layer* prev_layer) override;
};

#endif //CPPNET_INPUTLAYER_H

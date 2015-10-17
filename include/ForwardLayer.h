#ifndef CPPNET_FORWARDLAYER_H
#define CPPNET_FORWARDLAYER_H

#include "Layer.h"

#include <armadillo>

class ForwardLayer : public Layer {
public:

    ForwardLayer(int input_size, int output_size);
    ForwardLayer(int output_size);
    ~ForwardLayer();
    virtual void init() override;

    virtual const arma::vec* activate() override;
    virtual void backpropagate(const arma::vec* layer_error) override;


protected:
    arma::mat* _w;
    arma::vec  _prev_out;
};

#endif //CPPNET_FORWARDLAYER_H

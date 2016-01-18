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

    virtual arma::vec eval(const arma::vec& input) override;
    virtual const arma::vec* activate() override;
    virtual void backpropagate(const arma::vec* layer_error) override;
    virtual void backpropagate_apply() override;

protected:
    arma::mat* _w;
    arma::vec* _b;
    arma::mat* _dw;
    arma::vec* _db;
    uint32_t   _m;

    arma::mat* _w_last_change;
    arma::vec* _b_last_change;


    virtual void clean_dw();
};

#endif //CPPNET_FORWARDLAYER_H

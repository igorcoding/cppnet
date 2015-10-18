#ifndef CPPNET_RECURRENTLAYER_H
#define CPPNET_RECURRENTLAYER_H

#include "ForwardLayer.h"

#include <armadillo>
#include <vector>

class RecurrentLayer : public ForwardLayer {
    class RecurrentPartLayer : public ForwardLayer {
    public:
        RecurrentPartLayer(int input_size, int output_size);
        ~RecurrentPartLayer();
        virtual void init() override;

        virtual const arma::vec* activate() override {}
        virtual const arma::vec* activate(const arma::vec& context);
        virtual void backpropagate(const arma::vec* layer_error) override;
        virtual void backpropagate_apply() override;

    private:
        arma::mat* _context_w;
    };

public:
    RecurrentLayer(int output_size, int parts_count = 1);
    ~RecurrentLayer();
    virtual void init() override;

    int parts_count() const { return _parts.size(); }

    virtual const arma::vec* activate() override;
    virtual void backpropagate(const arma::vec* layer_error) override;
    virtual void backpropagate_apply() override;

protected:


private:
    std::vector<RecurrentPartLayer*> _parts;
    arma::vec _context;
};

#endif //CPPNET_RECURRENTLAYER_H

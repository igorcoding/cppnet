#ifndef CPPNET_RECURRENTLAYER_H
#define CPPNET_RECURRENTLAYER_H

#include "ForwardLayer.h"

#include <armadillo>
#include <vector>

class RecurrentLayer : public ForwardLayer {
    class RecurrentPartLayer : public ForwardLayer {
        friend class RecurrentLayer;
    public:
        RecurrentPartLayer(int input_size, int output_size);
        ~RecurrentPartLayer();
        virtual void init() override;

        virtual const arma::vec* activate() override {}
        virtual const arma::vec* activate(const arma::vec& context);
        virtual void backpropagate(const arma::vec* layer_error) override;
        virtual void backpropagate_apply() override;

        virtual void clean_dw() override;

        RecurrentPartLayer* prev_part_layer() const { return _prev_part_layer; }
        void set_prev_part_layer(RecurrentPartLayer* part) { _prev_part_layer = part; }

    private:
        arma::mat* _context_w;
        arma::mat* _context_dw;
        arma::mat* _context_w_last_change;

        RecurrentPartLayer* _prev_part_layer;
    };

public:
    RecurrentLayer(int output_size, int parts_count = 1);
    ~RecurrentLayer();
    virtual void init() override;

    int parts_count() const { return _parts.size(); }
    virtual void set_as_output() override;

    virtual arma::vec eval(const arma::vec& input) override;
    virtual const arma::vec* activate() override;
    virtual void backpropagate(const arma::vec* layer_error) override;
    virtual void backpropagate_apply() override;

protected:


private:
    std::vector<RecurrentPartLayer*> _parts;
    arma::vec _context;
    arma::mat* _w;
    arma::mat* _context_w;
};

#endif //CPPNET_RECURRENTLAYER_H

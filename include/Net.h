#ifndef CPPNET_NET_H
#define CPPNET_NET_H

#include "TrainExample.h"
#include "Layer.h"
#include "StreamLayer.h"
#include "OutputLayer.h"
#include "exceptions/EndOfDataException.h"

#include <vector>

#define ARMA_USE_CXX11

class Net {
public:
    Net();
    ~Net();

    void add_layer(Layer* layer);

    template <typename T>
    void train(TrainData<T>&& train_data);

    template <typename T>
    arma::vec predict(arma::Col<T>&& input_list);

private:
    std::vector<Layer*> _layers;
};



template <typename T>
void Net::train(TrainData<T>&& train_data) {
    StreamLayer<T>* stream_layer = new StreamLayer<T>(std::move(train_data));
    _layers.front()->set_prev_layer_force(stream_layer);
    _layers.back()->set_as_output();

    const arma::vec* res = nullptr;
    int max_iterations = 1000;
    for (int iteration = 1; iteration <= max_iterations; ++iteration) {
        std::cout << "Iteration #" << iteration << std::endl;
        while (true) {
            try {
                stream_layer->state_capture();
                res = _layers.back()->activate();
                stream_layer->state_rewind();
                _layers.back()->backpropagate(stream_layer->get_current_output());
//                std::cout << *res << std::endl;
                stream_layer->state_next();
            } catch (EndOfDataException&) {
                break;
            }
        }
        _layers.back()->backpropagate_apply();
        stream_layer->start_over();
    }

    _layers.front()->set_prev_layer_force(nullptr);
    delete stream_layer;
}

template <typename T>
arma::vec Net::predict(arma::Col<T>&& input_list) {
    arma::vec input(input_list);
    std::vector<arma::vec> results;
    results.push_back(input);
    for (auto& l : _layers) {
        results.push_back(l->eval(results.back()));
    }
    return results.back();
}

#endif //CPPNET_NET_H

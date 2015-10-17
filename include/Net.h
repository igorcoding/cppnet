#ifndef CPPNET_NET_H
#define CPPNET_NET_H

#include "TrainExample.h"
#include "Layer.h"
#include "StreamLayer.h"
#include "exceptions/EndOfDataException.h"

#include <vector>

class Net {
public:
    Net();
    ~Net();

    void add_layer(Layer* layer);

    template <typename T>
    void train(TrainData<T>&& train_data);

    template <typename T>
    T predict(std::initializer_list<T>&& input_list);

private:
    std::vector<Layer*> _layers;
};



template <typename T>
void Net::train(TrainData<T>&& train_data) {
    StreamLayer<T>* stream_layer = new StreamLayer<T>(std::move(train_data));
    _layers.front()->set_prev_layer_force(stream_layer);

    while (true) {
        try {
            stream_layer->state_capture();
            auto res = _layers.back()->activate();
            arma::vec error = *res - *stream_layer->get_current_output();
            _layers.back()->backpropagate(&error);
            std::cout << *res << std::endl << std::endl;
            stream_layer->state_next();
        } catch (EndOfDataException&) {
            break;
        }
    }

    _layers.front()->set_prev_layer_force(nullptr);
    delete stream_layer;
}

template <typename T>
T Net::predict(std::initializer_list<T>&& input_list) {
    std::vector<T> input(input_list);
    return 0;
}

#endif //CPPNET_NET_H

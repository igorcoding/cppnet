#include "Net.h"

Net::Net() { }

Net::~Net() {
    for (Layer*& l : _layers) {
        delete l;
        l = nullptr;
    }
    _layers.clear();
}

void Net::add_layer(Layer* layer) {
    if (!_layers.empty()) {
        auto& last_layer = _layers.back();
        layer->set_prev_layer(last_layer);
        last_layer->set_next_layer(layer);
        if (layer->input_size() == -1) {
            layer->input_size(last_layer->output_size());
        }
    } else if (layer->input_size() == -1) {
        throw std::logic_error("Layer's input size should be specified either via a constructor or via an InputLayer preceding current layer");
    }
    if (!layer->is_init()) {
        layer->init();
    }
    _layers.push_back(layer);
}

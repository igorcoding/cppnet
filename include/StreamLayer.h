#ifndef CPPNET_STREAMLAYER_H
#define CPPNET_STREAMLAYER_H

#include "Layer.h"
#include "TrainData.h"
#include "exceptions/EndOfDataException.h"

#include <armadillo>
#include <vector>
#include <type_traits>

template <typename T>
class StreamLayer : public Layer {
public:
    typedef typename std::vector<TrainExample<T>>::const_iterator train_data_it_t;
    typedef typename std::remove_reference<
        typename std::result_of<
            decltype(&TrainExample<T>::inputs)(TrainExample<T>)
        >::type
    >::type::const_iterator train_example_iterator_t;


    StreamLayer(TrainData<T>&& data);

    virtual void set_prev_layer(Layer* prev_layer) override;
    virtual const arma::vec* activate() override;
    virtual void backpropagate(const arma::vec* layer_error) override;
    virtual void backpropagate_apply() override;

    virtual void start_over();
    const arma::Col<T>* get_current_output() const;

    void state_capture();
    void state_rewind();
    void state_next();

private:
    TrainData<T> _train_data;
    train_data_it_t _data_it;
    train_example_iterator_t _train_example_it;

    train_data_it_t _state_data_it;
    train_example_iterator_t _state_train_example_it;
};

template <typename T>
StreamLayer<T>::StreamLayer(TrainData<T>&& train_data)
    : Layer(-1, -1),
      _train_data(std::move(train_data)),
      _data_it(_train_data.data().begin()),
      _train_example_it(_data_it->inputs().begin()),
      _state_data_it(nullptr),
      _state_train_example_it(nullptr) {
    init();
}

template <typename T> inline
void StreamLayer<T>::set_prev_layer(Layer* prev_layer) {
    throw std::logic_error("Stream layer can only be the first layer");
}

template <typename T> inline
const arma::vec* StreamLayer<T>::activate() {
    if (_data_it == _train_data.data().end()) {
        throw EndOfDataException();
    }
    if (_train_example_it == _data_it->inputs().end()) {
        // end of example
        ++_data_it;
        _train_example_it = _data_it->inputs().begin();
        return nullptr; // signal that example ended
    } else {
        const auto& d = *(_train_example_it++);
        return &d;
    }
}

template <typename T> inline
void StreamLayer<T>::backpropagate(const arma::vec* layer_error) {

}

template <typename T> inline
void StreamLayer<T>::backpropagate_apply() {

}

template <typename T> inline
void StreamLayer<T>::start_over() {
    _data_it = _train_data.data().begin();
    _train_example_it = _data_it->inputs().begin();
}

template <typename T> inline
const arma::Col<T>* StreamLayer<T>::get_current_output() const {
    return &_data_it->output();
}

template <typename T> inline
void StreamLayer<T>::state_capture() {
    _state_data_it = _data_it;
    _state_train_example_it = _train_example_it;
}

template <typename T> inline
void StreamLayer<T>::state_rewind() {
    _data_it = _state_data_it;
    _train_example_it = _state_train_example_it;
}

template <typename T> inline
void StreamLayer<T>::state_next() {
    ++_state_train_example_it;
    if (_state_train_example_it == _state_data_it->inputs().end()) {
        ++_state_data_it;
        _state_train_example_it = _state_data_it->inputs().begin();
    }
    state_rewind();
}

#endif // CPPNET_STREAMLAYER_H

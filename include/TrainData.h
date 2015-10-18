#ifndef CPPNET_TRAINDATA_H
#define CPPNET_TRAINDATA_H

#include "TrainExample.h"

#include <initializer_list>
#include <armadillo>

template <typename T>
class TrainData {
public:
    TrainData();

    void add(const std::initializer_list<arma::Col<T>>& inputs, const arma::Col<T>& output);
    void add(std::initializer_list<arma::Col<T>> inputs, arma::Col<T>&& output);
    void add(arma::Col<T>&& input, arma::Col<T>&& output);

    const std::vector<TrainExample<T>>& data() const { return _data; }

private:
    std::vector<TrainExample<T>> _data;
};

typedef TrainExample<double> Sample;


template <typename T>
TrainData<T>::TrainData()
{ }

//template <typename T>
//void TrainData<T>::add(const std::initializer_list<arma::Col<T>>& inputs, const arma::Col<T>& output) {
//    add(std::move(inputs), std::move(output));
//}

template <typename T>
void TrainData<T>::add(std::initializer_list<arma::Col<T>> inputs, arma::Col<T>&& output) {
    _data.push_back(TrainExample<T>(std::move(inputs), std::move(output)));
}

template <typename T>
void TrainData<T>::add(arma::Col<T>&& input, arma::Col<T>&& output) {
    _data.push_back(TrainExample<T>({std::move(input)}, std::move(output)));
}

#endif //CPPNET_TRAINDATA_H

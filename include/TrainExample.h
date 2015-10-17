#ifndef CPPNET_TRAINEXAMPLE_H
#define CPPNET_TRAINEXAMPLE_H

#include <armadillo>
#include <vector>

template <typename T>
class TrainExample {
public:
    TrainExample(std::initializer_list<arma::Col<T>>&& inputs, arma::Col<T>&& output);

    const std::vector<arma::Col<T>>& inputs() const;
    const arma::Col<T>& output() const;

private:
    std::vector<arma::Col<T>> _inputs;
    arma::Col<T> _output;
};

typedef TrainExample<double> Sample;


template <typename T>
TrainExample<T>::TrainExample(std::initializer_list<arma::Col<T>>&& inputs, arma::Col<T>&& output)
        : _inputs(inputs),
          _output(output)
{ }

template <typename T> inline
const std::vector<arma::Col<T> > &TrainExample<T>::inputs() const {
    return _inputs;
}

template <typename T> inline
const arma::Col<T>& TrainExample<T>::output() const {
    return _output;
}

#endif //CPPNET_TRAINEXAMPLE_H

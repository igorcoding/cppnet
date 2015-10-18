#include <iostream>

#include "Net.h"
#include "InputLayer.h"
#include "ForwardLayer.h"
#include "RecurrentLayer.h"

using namespace std;

int main() {
//    arma::arma_rng::set_seed(100);
    arma::arma_rng::set_seed_random();
    Net net;

    net.add_layer(new InputLayer(2));
    net.add_layer(new ForwardLayer(3));
    net.add_layer(new ForwardLayer(1));


    TrainData<double> train_data;
    train_data.add({{0, 0}}, {0});
    train_data.add({{0, 1}}, {1});
    train_data.add({{1, 0}}, {1});
    train_data.add({{1, 1}}, {0});

    net.train(std::move(train_data));

    std::cout << "{0, 0} => " << net.predict(arma::vec({0, 0})) << std::endl;
    std::cout << "{0, 1} => " << net.predict(arma::vec({0, 1})) << std::endl;
    std::cout << "{1, 0} => " << net.predict(arma::vec({1, 0})) << std::endl;
    std::cout << "{1, 1} => " << net.predict(arma::vec({1, 1})) << std::endl;


    return 0;
}

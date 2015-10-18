#include <iostream>

#include "Net.h"
#include "InputLayer.h"
#include "ForwardLayer.h"
#include "RecurrentLayer.h"

using namespace std;

int main() {
    arma::arma_rng::set_seed(100);
    Net net;

    net.add_layer(new InputLayer(2));
    net.add_layer(new RecurrentLayer(3, 2));
    net.add_layer(new ForwardLayer(4));


    TrainData<double> train_data;
    train_data.add({
                       {1,2},
                       {3,4},
                       {5,6},
                   }, {1,0,0,1});

    train_data.add({
                       {7,8},
                       {9,10},
                       {11,12},
                   }, {0,1,1,0});
    net.train(std::move(train_data));

    return 0;
}

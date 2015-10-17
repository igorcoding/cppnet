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
    net.add_layer(new RecurrentLayer(3));
    net.add_layer(new ForwardLayer(4));


    TrainData<double> train_data;
    train_data.add({1,2}, {1,0,0,1});
    net.train(std::move(train_data));

    return 0;
}

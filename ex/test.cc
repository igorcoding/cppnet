#include <iostream>
#include <armadillo>

int main() {
    arma::vec v0(5, arma::fill::randn);
    arma::vec v = 1.0 / (1 + arma::exp(-v0));
    std::cout << v << std::endl;
}

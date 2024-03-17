#include <iostream>

using namespace std ;

#define SAMPLE_SIZE 5
#define DIM_SIZE 3


void backProPagation(const double x_train[][DIM_SIZE], const double y_train[][1], 
                     const double lr, double hyphothe[][1], double weight[][1], double bias[][1]) {

    double w_sum[DIM_SIZE][1] = {{}} ;
    double b_sum[SAMPLE_SIZE][1] = {{}} ;

    for (int i = 0; i < DIM_SIZE; i++) {
        for (int j = 0; j < SAMPLE_SIZE; j++) {
            w_sum[i][0] -= x_train[j][i] * (y_train[j][0] - hyphothe[j][0]) ;
            w_sum[i][0] = w_sum[i][0] * 2 / SAMPLE_SIZE ;
        }
    }

    for (int i = 0; i < SAMPLE_SIZE; i++) {
        b_sum[i][0] -= (y_train[i][0] - hyphothe[i][0]) ;
        b_sum[i][0] = b_sum[i][0] * 2 / SAMPLE_SIZE ;
    }

    // update
    for (int i = 0; i < DIM_SIZE; i++) {
        weight[i][0] -= lr * w_sum[i][0] ;
    }
    for (int i = 0; i < SAMPLE_SIZE; i++) {
        bias[i][0] -= lr * b_sum[i][0] ;
    }

}


double getCost(const double y_train[][1], double hyphothe[][1]) {
    // MSE
    double cost = 0. ;

    for (int i = 0; i < SAMPLE_SIZE; i++) {
        cost += (y_train[i][0] - hyphothe[i][0]) * (y_train[i][0] - hyphothe[i][0]) ;
    }

    cost /= SAMPLE_SIZE ;

    return cost ;
}


void fowardProPagation(const double x_train[][DIM_SIZE], double hyphothe[][1], 
                       double weight[][1], double bias[][1]) {

    for (int i = 0; i < SAMPLE_SIZE; i++) {
        for (int j = 0; j < DIM_SIZE; j++) {
            hyphothe[i][0] = x_train[i][j] * weight[j][0] ;
        }
    }

    for (int i = 0; i < SAMPLE_SIZE; i++) {
        hyphothe[i][0] += bias[i][0] ;
    }    

}


int main() {

    const double x_train[SAMPLE_SIZE][DIM_SIZE] = {{73., 80., 75.},
                                                   {93., 88., 93.},
                                                   {89., 91., 80.},
                                                   {96., 98., 100.},
                                                   {73., 66., 70.}} ;
    const double y_train[SAMPLE_SIZE][1] = {{152.}, {185.}, {180.}, {196.}, {142.}} ;

    double hyphothe[SAMPLE_SIZE][1] = {{}} ;
    double weight[DIM_SIZE][1]      = {{}} ;
    double bias[SAMPLE_SIZE][1]     = {{}} ;

    double cost ;
    const double lr = 1.0E-5 ;
    const int epoch = 100000 ;

    for (int i = 0; i < epoch+1; i++) {
        fowardProPagation(x_train, hyphothe, weight, bias) ;
        cost = getCost(y_train, hyphothe) ;
        backProPagation(x_train, y_train, lr, hyphothe, weight, bias) ;

        if (i % 10000 == 0) {
            cout << "Epoch " << i << "/" << epoch << " Cost : " << cost << '\n' ;
        }
    }

    return 0 ;
}
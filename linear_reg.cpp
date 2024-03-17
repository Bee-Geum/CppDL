#include <iostream>

using namespace std ;


void backProPagation(double* weight, double* bias, const double lr, 
                    const double* x_train, const double* y_train, double* hypothe) {
    double w_sum = 0. ;
    double b_sum = 0. ;

    for (int i = 0; i < 3; i++) {
        w_sum -= x_train[i] * (y_train[i] - hypothe[i]) ;
        b_sum -= (y_train[i] - hypothe[i]) ;
    } 

    w_sum = w_sum * 2 / 3 ;
    b_sum = b_sum * 2 / 3 ;

    weight[0] -= lr * w_sum ;
    bias[0]   -= lr * b_sum ;
}

double getCost(double* hypothe, const double* y_train) {
    // MSE
    double sum = 0. ;

    for (int i = 0; i < 3; i++) {
        sum += (y_train[i] - hypothe[i]) * (y_train[i] - hypothe[i]) ;
    }    

    double cost = sum / 3 ;

    return cost ;
}


void matAdd(double* bias, double* hypothe) {
    for (int i = 0; i < 3; i++) {
        hypothe[i] = hypothe[i] + bias[0]  ;
    }    
}


void matMul(const double* x_train, double* weight, double* hypothe) {
    for (int i = 0; i < 3; i++) {
        hypothe[i] = x_train[i] * weight[0]  ;
    }
}


int main() {

    const double x_train[3][1] = {{1.}, {2.}, {3.}} ;
    const double y_train[3][1] = {{2.}, {4.}, {6.}} ;
    double hypothe[3][1] = {} ;

    double weight[1][1]  = {{0.}} ;      // 그냥 0으로 초기화
    double bias[1][1]    = {{0.}} ;      // 그냥 0으로 초기화

    double cost ;
    const double lr = 0.01 ;
    const int epoch = 2000 ;

    for (int i = 0; i < epoch; i++) {
        matMul(&x_train[0][0], &weight[0][0], &hypothe[0][0]) ;
        matAdd(&bias[0][0], &hypothe[0][0]) ;
        cost = getCost(&hypothe[0][0], &y_train[0][0]) ;
        backProPagation(&weight[0][0], &bias[0][0], lr, 
                        &x_train[0][0], &y_train[0][0], &hypothe[0][0]) ; 

        if (i % 100 == 0) {
            cout << "Epoch" << i << "/" << epoch << " w: " << *weight[0] << ", "
            << "b: " << *bias[0] << ", " << "Cost: " << cost << '\n' ;
        }
    }

    return 0 ;
}
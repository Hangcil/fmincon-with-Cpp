#include <iostream>
#include <cmath>
#include "sci_arma.h"
int main() {
    auto f=[](vec& x)-> double {
        return sin(x(0)+x(1)-3)+2*pow(x(1)+2,2);
    };
    vec x0= zeros(2,1);
    auto result1=sci_arma::fmin(f,x0);
    mat A={{1,1},{2,3}};
    vec b={1,2};
    auto result2=sci_arma::fmincon(f,x0,A,b);
    std::cout<<"local solution: "<<endl<<result1.x<<"min:"<<endl<<"  "<<result1.fval<<endl;
    std::cout<<"local solution: "<<endl<<result2.x<<"min:"<<endl<<"  "<<result2.fval<<endl;
    return 0;
}

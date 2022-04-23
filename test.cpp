#include <iostream>
#include <cmath>
#include "sci_arma.h"
int main() {
    //define a obj_fun
    auto f=[](vec& x)-> double {
        return pow(1-x(0),2)+pow(x(1)-x(0)*x(0),2);
    };
    
    //define a start point
    vec x0= zeros(2,1);
    
    //optimization with no constraints
    auto result1=sci_arma::fmin(f,x0);
    
    //define linear constraints Ax<=b
    mat A={{1,1},{2,3}};
    vec b={1,2};
    
    //optimization with constraints
    auto result2= sci_arma::fmincon(f, x0, A, b);
    
    //output
    std::cout<<"empty constraints:" <<endl<<"local solution: "<<endl
    <<result1.x<<"min:"<<endl<<"  "<<result1.fval<<endl;
    std::cout<<endl;
    std::cout<<"linear constraints:"<<endl<<"local solution: "<<endl
    <<result2.x<<"min:"<<endl<<"  "<<result2.fval<<endl;
    
    return 0;
}

# Cpp-fmincon-based-on-armadillo
An easy optimization C++ lib based on armadillo. It just implements one of the frequently used function ```fmincon()```.
Example:
'''c++
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

    //optimization with inequality linear constraints
    auto result2= sci_arma::fmincon(f, x0, A, b);

    //define linear equality constraints
    mat Aeq={{1,2}};
    vec beq={0};

    //optimization with mixed linear constraints
    auto result3= sci_arma::fmincon(f, x0, A, b, Aeq, beq);

    //define non-linear inequality constraints c(x)<=0
    auto c=[](vec& x)->vec {
        vec temp={x(0)*x(0)+x(1)*x(1)-0.05};
        return temp;
    };

    //optimization with mixed constraints
    auto result4= sci_arma::fmincon(f, x0, A, b, Aeq, beq, c);

    //output
    std::cout<<"empty constraints:" <<endl<<"local solution: "<<endl
             <<result1.x<<"min:"<<endl<<"  "<<result1.fval<<endl;
    std::cout<<endl;
    std::cout<<"linear constraints:"<<endl<<"local solution: "<<endl
             <<result2.x<<"min:"<<endl<<"  "<<result2.fval<<endl;
    std::cout<<endl;
    std::cout<<"Mixed linear constraints:"<<endl<<"local solution: "<<endl
             <<result3.x<<"min:"<<endl<<"  "<<result3.fval<<endl;
    std::cout<<endl;
    std::cout<<"Mixed constraints:"<<endl<<"local solution: "<<endl
             <<result4.x<<"min:"<<endl<<"  "<<result4.fval<<endl;
    return 0;
}
'''

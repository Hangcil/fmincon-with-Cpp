#include <iostream>
#include <cmath>
#include "sci_arma.h"
int main() {
    //define a obj_fun
    obj_fun f=[](vec& x)-> double {
        return pow(1-0.1*x(0),2)+100*pow(x(1)-0.5*x(0)*x(0),2)
               +pow(1-0.1*x(2),2)+100*pow(x(3)-0.5*x(2)*x(2),2);
    };

    //define the gradient
    gradient g=[](vec& x)->vec{
        vec result= {-0.2*(1-0.1*x(0)) - 200*x(0)*(x(1)-0.5*x(0)*x(0)),
                     200*(x(1)-0.5*x(0)*x(0)),
                     -0.2*(1-0.1*x(2)) - 200*x(2)*(x(3)-0.5*x(2)*x(2)),
                     200*(x(3)-0.5*x(2)*x(2))};
        return result;
    };

    options opt;
    opt.gra=g;
    opt.enable_self_defined_gra= true;
    opt.max_ite=5000;
    vec x0= {0,0,0,0};
    auto result1= sci_arma::fmincon(f, x0,opt);
    std::cout<<result1<<endl;
    return 0;
}
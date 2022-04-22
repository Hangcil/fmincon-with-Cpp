#ifndef ARM_TEST_SCI_ARMA_H
#define ARM_TEST_SCI_ARMA_H

#include "armadillo"
#include <functional>
using namespace arma;
using obj_fun=std::function<double(vec&)>;

//x_fval performs as a wrapper for the returned value.
struct x_fval
{
public:
    vec x;
    double fval;
};

//two simple static member functions
class sci_arma
{
public:
    static x_fval fmin(const obj_fun& f, vec& x0);
    static x_fval fmincon(const obj_fun& f, vec& x0, mat& Aeq, mat& beq);
};

#endif //ARM_TEST_SCI_ARMA_H

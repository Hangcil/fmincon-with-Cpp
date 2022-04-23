#ifndef ARM_TEST_SCI_ARMA_H
#define ARM_TEST_SCI_ARMA_H

#include "armadillo"
#include <functional>
#include <algorithm>
using namespace arma;
using obj_fun=std::function<double(vec&)>;

//x_fval defines local solution and minimal value
struct x_fval
{
public:
    vec x;
    double fval;
};

//class sci_arma wraps two functions
class sci_arma
{
public:
    static x_fval fmin(const obj_fun& f, vec& x0);
    static x_fval fmincon(const obj_fun& f, vec& x0, mat& A, mat& b);
};

#endif //ARM_TEST_SCI_ARMA_H

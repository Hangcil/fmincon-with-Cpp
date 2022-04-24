#ifndef ARM_TEST_SCI_ARMA_H
#define ARM_TEST_SCI_ARMA_H

#include "armadillo"
#include <functional>
#include <algorithm>
using namespace arma;

//defines special functions needed when optimizing
using obj_fun=std::function<double(vec&)>;
using non_linear_con=std::function<vec(vec&)>;

//x_fval defines local minimal returned by fmincon()
struct x_fval
{
public:
    vec x;
    double fval;
};

//sci_arma wraps optimizing functions
class sci_arma
{
public:
    static x_fval fmin(const obj_fun& f, vec& x0);
    static x_fval fmincon(const obj_fun& f, vec& x0, mat& A, mat& b);
    static x_fval fmincon(const obj_fun& f, vec& x0, mat& A, mat& b, mat& Aeq, mat& beq);
    static x_fval fmincon(const obj_fun& f, vec& x0, mat& A, mat& b, mat& Aeq, mat& beq, const non_linear_con& c, long n_non_lin);
};

#endif //ARM_TEST_SCI_ARMA_H

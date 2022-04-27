#ifndef ARM_TEST_SCI_ARMA_H
#define ARM_TEST_SCI_ARMA_H

#include "armadillo"
#include <functional>
#include <algorithm>
#include <vector>
using namespace arma;


//defines special functions needed when optimizing
using obj_fun=std::function<double(vec&)>;
using non_linear_con=std::function<vec(vec&)>;


//x_fval defines local minimal returned by fmincon()
struct x_fval
{
public:
    vec x;
    double fval=0;
};


//defines algorithms
//Powell is faster at most cases
enum algorithm
{
    Rosenbrock,
    Powell
};


//options set for fmincon
//max_ite defines max iterations in 'exterior-point' method
struct options
{
public:
    algorithm algo=Powell;
    long long max_ite=5000;
};


//sci_arma wraps optimizing functions
class sci_arma
{
public:
    //Rosenbrock method
    static x_fval rosenbrock(const obj_fun& f, vec& x0);

    //the following three functions implement Powell method
    //as for functions line_search() and include_min(),
    //they were designed to be nested in powell(),
    //but you can use them as standalone methods without any input check
    static x_fval powell(const obj_fun& f, vec& x0);
    static vec line_search(const obj_fun& f, vec& a, vec& b);
    static std::vector<vec> include_min(const obj_fun& f, vec& x0, vec& dir);

    //the followings are fmincon()-s;
    static x_fval fmincon(const obj_fun& f, vec& x0, mat& A, mat& b, const options& opt);
    static x_fval fmincon(const obj_fun& f, vec& x0, mat& A, mat& b, mat& Aeq, mat& beq, const options& opt);
    static x_fval fmincon(const obj_fun& f, vec& x0, mat& A, mat& b, mat& Aeq, mat& beq, const non_linear_con& c, const options& opt);
    static x_fval fmincon(const obj_fun& f, vec& x0, mat& A, mat& b);
    static x_fval fmincon(const obj_fun& f, vec& x0, mat& A, mat& b, mat& Aeq, mat& beq);
    static x_fval fmincon(const obj_fun& f, vec& x0, mat& A, mat& b, mat& Aeq, mat& beq, const non_linear_con& c);
};

#endif //ARM_TEST_SCI_ARMA_H

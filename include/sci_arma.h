#ifndef ARM_TEST_SCI_ARMA_H
#define ARM_TEST_SCI_ARMA_H

#include "armadillo"
#include <functional>
#include <algorithm>
#include <vector>
#include <string>
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
    std::string if_forced_terminated;
    std::string algorithm;
    long long ite_times;
    friend std::ostream& operator<< (std::ostream& out, x_fval& result);
};


//defines algorithms
//default algorithms:
//constraint-free: BFGS
//non-constraint-free: Powell_modified
enum algorithm
{
    BFGS,
    Powell,
    Powell_modified,
    Rosenbrock
};


//options set for fmincon
//max_ite defines max iterations in most cases
struct options
{
public:
    algorithm algo=Powell_modified;
    long long max_ite=1000;
};


//sci_arma wraps optimizing functions
class sci_arma
{
protected:
    //Rosenbrock method
    static x_fval rosenbrock(const obj_fun& f, vec& x0, const options& opt);

    //the following three functions implement Powell method
    static x_fval powell(const obj_fun& f, vec& x0, const options& opt);
    static x_fval powell_m(const obj_fun& f, vec& x0, const options& opt);
    static vec line_search(const obj_fun& f, vec& a, vec& b);
    static std::vector<vec> include_min(const obj_fun& f, vec& x0, vec& dir);

    //the following two functions implement BFGS method
    static vec gra(const obj_fun& f, vec& pos);
    static x_fval bfgs(const obj_fun& f, vec& x0, const options& opt);


public:
    //the followings are fmincon()-s;
    static x_fval fmincon(const obj_fun& f, vec& x0, const options& opt);
    static x_fval fmincon(const obj_fun& f, vec& x0, mat& A, mat& b, const options& opt);
    static x_fval fmincon(const obj_fun& f, vec& x0, mat& A, mat& b, mat& Aeq, mat& beq, const options& opt);
    static x_fval fmincon(const obj_fun& f, vec& x0, mat& A, mat& b, mat& Aeq, mat& beq, const non_linear_con& c, const options& opt);
    static x_fval fmincon(const obj_fun& f, vec& x0);
    static x_fval fmincon(const obj_fun& f, vec& x0, mat& A, mat& b);
    static x_fval fmincon(const obj_fun& f, vec& x0, mat& A, mat& b, mat& Aeq, mat& beq);
    static x_fval fmincon(const obj_fun& f, vec& x0, mat& A, mat& b, mat& Aeq, mat& beq, const non_linear_con& c);
};

inline std::ostream& operator<< (std::ostream& out, x_fval& result)
{
    cout<<"local minimal: "<<endl<<result.x
    <<"value:"<<endl<<"   "<<result.fval<<endl
    <<"algorithm:"<<endl<<"   "<<result.algorithm<<endl
    <<"iteration times:"<<endl<<"   "<<result.ite_times<<endl
    <<"if_forced_terminated"<<endl<<"   "<<result.if_forced_terminated<<endl;
    return out;
}

#endif //ARM_TEST_SCI_ARMA_H

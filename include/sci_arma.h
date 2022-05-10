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
using gradient=std::function<vec(vec&)>;


//x_fval defines local minimal returned by fmincon()
struct x_fval
{
public:
    vec x;
    double fval=0;
    bool if_forced_terminated= false;
    std::string algorithm;
    std::string warning;
    long long ite_times=0;
    friend std::ostream& operator<< (std::ostream& out, x_fval& result);
};


//defines algorithms
//default algorithms:
//constraint-free: BFGS
//non-constraint-free: Powell_modified
enum algorithm
{
    preset,
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
    algorithm algo=preset;
    long long max_ite=1000;
    bool enable_self_defined_gra= false;
    gradient gra;
    double tolerance=1.0/double(max_ite)/1000;
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

    //the following three functions implement BFGS method
    static vec gra(const obj_fun& f, vec& pos);
    static double line_search_imprecise(const obj_fun& f, const gradient& g, vec& x0, vec& dir);
    static x_fval bfgs(const obj_fun& f, vec& x0, const options& opt);


public:
    //the followings are fmincon()-s;
    static x_fval fmincon(const obj_fun& f, vec& x0, const options& opt);
    static x_fval fmincon(const obj_fun& f, vec& x0, vec& lb, vec& ub, const options& opt);
    static x_fval fmincon(const obj_fun& f, vec& x0, mat& A, mat& b, const options& opt);
    static x_fval fmincon(const obj_fun& f, vec& x0, mat& A, mat& b, vec& lb, vec& ub, const options& opt);
    static x_fval fmincon(const obj_fun& f, vec& x0, mat& A, mat& b, mat& Aeq, mat& beq, const options& opt);
    static x_fval fmincon(const obj_fun& f, vec& x0, mat& A, mat& b, mat& Aeq, mat& beq, vec& lb, vec& ub, const options& opt);
    static x_fval fmincon(const obj_fun& f, vec& x0, mat& A, mat& b, mat& Aeq, mat& beq, const non_linear_con& c, const options& opt);
    static x_fval fmincon(const obj_fun& f, vec& x0, mat& A, mat& b, mat& Aeq, mat& beq, vec& lb, vec& ub, const non_linear_con& c, const options& opt);//
    static x_fval fmincon(const obj_fun& f, vec& x0);
    static x_fval fmincon(const obj_fun& f, vec& x0, vec& lb, vec& ub);
    static x_fval fmincon(const obj_fun& f, vec& x0, mat& A, mat& b);
    static x_fval fmincon(const obj_fun& f, vec& x0, mat& A, mat& b, vec& lb, vec& ub);
    static x_fval fmincon(const obj_fun& f, vec& x0, mat& A, mat& b, mat& Aeq, mat& beq);
    static x_fval fmincon(const obj_fun& f, vec& x0, mat& A, mat& b, mat& Aeq, mat& beq, vec& lb, vec& ub);
    static x_fval fmincon(const obj_fun& f, vec& x0, mat& A, mat& b, mat& Aeq, mat& beq, const non_linear_con& c);
    static x_fval fmincon(const obj_fun& f, vec& x0, mat& A, mat& b, mat& Aeq, mat& beq, vec& lb, vec& ub, const non_linear_con& c);
};

inline std::ostream& operator<< (std::ostream& out, x_fval& result)
{
    if(!result.warning.empty())
        cout<<"warning: "<<endl<<"   "<<result.warning<<endl;
    cout<<"local minimal: "<<endl<<result.x
    <<"value:"<<endl<<"   "<<result.fval<<endl
    <<"algorithm:"<<endl<<"   "<<result.algorithm<<endl
    <<"iteration times:"<<endl<<"   "<<result.ite_times<<endl
    <<"if_forced_terminated"<<endl<<"   ";
    if(result.if_forced_terminated)
        cout<<"true"<<endl;
    else
        cout<<"false"<<endl;
    return out;
}

#endif //ARM_TEST_SCI_ARMA_H

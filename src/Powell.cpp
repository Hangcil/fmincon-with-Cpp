#include "sci_arma.h"


std::vector<vec> sci_arma::include_min(const obj_fun &f, vec &x0, vec &dir)
//successfull-failure method
//as a nested optimizing function in powell(), it does not check for any input errors
//this function returns a vector of x0 and x1
//such that [x0,x1] (or [x1,x0]) includes a local minimal
{
    double h=0.01;
    vec x1=x0, x2, x3, x4;
    double f1=f(x1), f4;
    int k=0;
    long long ite=0;
    while (true)
    {
        ite++;
        x4=x1+h*dir;
        f4=f(x4);
        k++;
        if(f4<f1)
        {
            x2=x1;
            x1=x4;
            f1=f4;
            h*=1.5;
            continue;
        }else
        {
            if(k==1 && ite<50)
            {
                h*=-1;
                x2=x4;
                continue;
            }else
            {
                x3=x2;
                x1=x4;
                std::vector<vec>result;
                result.push_back(x1);
                result.push_back(x3);
                return result;
            }
        }
    }
}


vec sci_arma::line_search(const obj_fun &f, vec &a, vec &b)
//0.618 method
//as a nested optimizing function in powell(), it does not check for any input errors
//this function returns a vec that is a local minimal on the line strictly between 'a' and 'b'
{
    auto g=[&](double x)->double {
        vec temp=(1-x)*a+x*b;
        return f(temp);
    };
    double a1=0.0, b1=1.0, l=0.001,
    lambda=a1+0.382*(b1-a1), mu=a1+0.618*(b1-a1);
    long long ite=0;
    while (true)
    {
        ite++;
        if(b1-a1<=l || ite>=50)
        {
            vec result=(1-(b1+a1)/2)*a+(b1+a1)/2*b;
            return result;
        }else
        {
            if(g(lambda)>g(mu))
            {
                a1=lambda;
                lambda=mu;
                mu=a1+0.618*(b1-a1);
                continue;
            }else
            {
                b1=mu;
                mu=lambda;
                lambda=a1+0.382*(b1-a1);
                continue;
            }
        }
    }
}



x_fval sci_arma::powell(const obj_fun &f, vec &x0, const options& opt)
//Powell method
{
    auto nvar=x0.n_rows;
    try{
        if((int)x0.n_cols!=1)
            throw std::logic_error("ERROR: powell(): x0 must be a column vector.");
    }
    catch(std::exception& e){
        std::cerr<<e.what()<<std::endl;
        std::terminate();
    }
    mat E= eye(nvar,nvar);
    vec x1=x0, d[nvar];
    for(auto i=0;i<nvar;i++){
        d[i]=E.col(i);
    }
    double eps=opt.tolerance<=0.0001 ? opt.tolerance:0.0001;
    long long ite=0;
    while(true)
    {
        ite++;
        auto x_kn=x1;
        for(auto i=0;i<nvar;i++)
        {
            auto range= include_min(f, x_kn, d[i]);
            x_kn= line_search(f, range[0], range[1]);
        }
        vec d_n_1=x_kn-x1;
        auto range= include_min(f, x_kn, d_n_1);
        auto x_k= line_search(f, range[0], range[1]);
        if(norm(x_k-x1)<=eps || ite>=5000000)
        {
            x_fval result;
            if(norm(x_k-x1)> eps){
                result.if_forced_terminated=true;
            }else{
                result.if_forced_terminated=false;
            }
            result.algorithm="Powell";
            result.ite_times=ite;
            result.x=x_k;
            result.fval=f(x_k);
            return result;
        }else
        {
            for(auto i=0;i<nvar-1;i++){
                d[i]=d[i+1];
            }
            x1=x_k;
            d[nvar-1]=d_n_1;
            continue;
        }
    }
}
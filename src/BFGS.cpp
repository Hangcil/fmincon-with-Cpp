#include "sci_arma.h"


vec sci_arma::gra(const obj_fun &f, vec &pos)
{
    double alpha=0.9, eps=0.0001;
    auto nvar=pos.n_rows;
    vec result= zeros(nvar,1);
    for(auto i=0;i<nvar;i++)
    {
        auto par=[&](double x)->double {
            vec x1=pos;
            x1(i)+=x;
            return f(x1);
        };
        double h=0.1;
        double p=(par(h)-par(-h))/2/h;
        long long ite=0;
        while (true)
        {
            ite++;
            h*=alpha;
            double p_=(par(h)-par(-h))/2/h;
            if(std::abs(p-p_)<eps || ite>=50){
                result(i)=p_;
                break;
            }else{
                p=p_;
            }
        }
    }
    return result;
}


/*
double sci_arma::line_search_imprecise(const obj_fun& f, vec& x0, vec& dir)
//Wolfe-Powell's law for imprecise line search
//this method needs further developments, so it's used nowhere
{
    double lambda=1, rho=0.3, delta=0.5, a=0, b=50000000;
    long long ite=0;
    while(true)
    {
        ite++;
        vec x1=x0+lambda*dir;
        double temp= sum(gra(f,x0).t()*dir);
        if(f(x1)>f(x0)+rho*lambda*temp)
        {
            b=lambda;
            lambda=(a+lambda)/2;
        }else
        {
            double temp1= sum(gra(f,x1).t()*dir);
            double temp2= delta*temp;
            if(temp1<temp2 && ite<50){
                a=lambda;
                lambda=std::min(2*lambda,(a+b)/2);
            }else
            {
                return lambda;
            }
        }
    }
}
*/

x_fval sci_arma::bfgs(const obj_fun &f, vec &x0, const options& opt)
{
    auto nvar=x0.n_rows;
    try{
        if((int)x0.n_cols!=1)
            throw std::logic_error("ERROR: bfgs(): x0 must be a column vector.");
    }
    catch(std::exception& e){
        std::cerr<<e.what()<<std::endl;
        std::terminate();
    }
    vec x1=x0, x11=x0;
    double eps=0.00001;
    int ite=0;
    while (true)
    {
        ite++;
        mat H= eye(nvar,nvar);
        auto g= gra(f,x1);
        for(auto k=0;k<nvar-1;k++)
        {
            vec d=-H*g;
            auto range= include_min(f, x1,d);
            auto x_kn_1= line_search(f, range[0], range[1]);
            vec temp_=x_kn_1-x1;
            vec temp=solve(d,temp_);
            double lambda= std::abs(sum(temp));
            x11=x1+lambda*d;
            auto g1= gra(f,x11);
            if(norm(g1)<= eps || ite>=opt.max_ite)
            {
                x_fval result;
                if(norm(g1)> eps){
                    result.if_forced_terminated="true";
                }else{
                    result.if_forced_terminated="false";
                }
                result.algorithm="BFGS";
                result.ite_times=ite;
                result.x=x11;
                result.fval=f(x11);
                return result;
            }else
            {
                auto save=g;
                g=g1;
                vec p=x11-x1, q=g1-save;
                double temp1=sum(q.t()*H*q), temp2=sum(p.t()*q);
                mat temp3=p*p.t(), temp4=p*q.t()*H, temp5=H*q*p.t();
                H=H+(1+temp1/temp2)/temp2*temp3-(1/temp2)*(temp4+temp5);
            }
            x1=x11;
        }
        x1=x11;
    }
}
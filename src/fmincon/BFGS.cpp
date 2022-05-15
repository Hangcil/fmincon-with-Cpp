#include "sci_arma.h"
#include "sci_arma.h"


vec sci_arma::gra(const obj_fun &f, vec &pos)
//numerical gradient calculation
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



double sci_arma::line_search_imprecise(const obj_fun& f, const gradient& g, vec& x0, vec& dir)
//Wolfe-Powell's law for imprecise line search
//require self-defined gradient
{
    double lambda=1, rho=0.1, delta=0.2, alpha=1.5, beta=0.5;
    long long ite=0;
    while(true)
    {
        ite++;
        vec x1=x0+lambda*dir;
        double temp= sum(g(x0).t()*dir);
        if(f(x1)<=f(x0)+rho*lambda*temp || ite>=50)
        {
            double temp1=sum(g(x1).t()*dir);
            if(rho*lambda*temp1>=delta*temp*lambda || ite>=50)
                return lambda;
            else
            {
                lambda*=alpha;
                continue;
            }
        }else
        {
            lambda*=beta;
            continue;
        }
    }
}


x_fval sci_arma::bfgs(const obj_fun &f, vec &x0, const options& opt)
{
    auto nvar=x0.n_rows;
    try{
        if((int)x0.n_cols!=1)
            throw std::logic_error("ERROR: fmincon(): x0 must be a column vector.");
    }
    catch(std::exception& e){
        std::cerr<<e.what()<<std::endl;
        std::terminate();
    }
    vec x1=x0, x11=x0;
    double eps=opt.tolerance<=0.0001 ? opt.tolerance:0.0001;
    int ite=0;
    while (true)
    {
        ite++;
        mat H= eye(nvar,nvar);
        vec g;
        if(!opt.enable_self_defined_gra)
        { g= gra(f, x1);}
        else
        {g= opt.gra(x1);}
        for(auto k=0;k<nvar;k++)
        {
            vec d=-H*g;
            double lambda;
            if(!opt.enable_self_defined_gra)
            {
                auto range= include_min(f, x1,d);
                auto x_kn_1= line_search(f, range[0], range[1]);
                vec temp_=x_kn_1-x1;
                vec temp=solve(d,temp_);
                lambda= std::abs(sum(temp));
            }else{
                lambda= line_search_imprecise(f,opt.gra,x1,d);
            }
            x11=x1+lambda*d;
            vec g1;
            if(!opt.enable_self_defined_gra) { g1= gra(f, x11);}
            else{g1= opt.gra(x11);}
            if(norm(g1-g)<0.000001* norm(g))
            {
                auto result= powell_m(f,x0,opt);
                result.warning="non-decreasing direction occurred. BFGS failed.";
                return result;
            }
            if(norm(g1)<= eps || ite>=opt.max_ite)
            {
                x_fval result;
                if(norm(g1)> eps){
                    result.if_forced_terminated=true;
                }else{
                    result.if_forced_terminated=false;
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



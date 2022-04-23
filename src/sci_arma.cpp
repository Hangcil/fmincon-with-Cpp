#include "sci_arma.h"

x_fval sci_arma::fmin(const obj_fun& f, vec& x0)
//Rosenbrock Method
{
    auto nvar=x0.n_rows;
    try{
        if((int)x0.n_cols!=1)
            throw std::logic_error("ERROR: x0 must be a column vector.");
    }
    catch(std::exception& e){
        std::cerr<<e.what()<<std::endl;
        std::terminate();
    }
    mat E= eye(nvar,nvar);
    vec x=x0, d[nvar];
    for(auto i=0;i<nvar;i++){
        d[i]=E.col(i);
    }
    int j=1;
    double alpha=1.05, beta=-0.95, eps=0.00001;
    vec y1=x, y=y1, delta0= ones(nvar,1),
    lambda= zeros(nvar,1), delta=delta0;

    while(true)
    {
        auto y_=(vec)(y+delta(j-1)*d[j-1]);
        if(f(y_)<f(y))
        {
            y=y_;
            delta(j-1)=alpha*delta(j-1);
            lambda(j-1)=delta(j-1);
        }
        else
        {
            delta(j-1)=beta*delta(j-1);
        }
        if(j<nvar)
        {
            j++;
            continue;
        }
        else
        {
            if(f(y)<f(y1))
            {
                y1=y;
                j=1;
                continue;
            }
            else
            {
                if(f(y)<f(x))
                {
                    auto x_=y;
                    if(norm(x_-x)<=eps)
                    {
                        x_fval result;
                        result.x=x_;
                        result.fval=f(x_);
                        return result;
                    }
                    else
                    {
                        x=y;
                        vec p[nvar],q[nvar];
                        for(auto i=0;i<nvar;i++)
                        {
                            if(lambda[i]==0)
                            {
                                p[i]=d[i];
                            }else
                            {
                                p[i]= zeros(nvar,1);
                                for(auto l=i;l<nvar;l++)
                                {
                                    p[i]+=lambda[l]*d[l];
                                }
                            }
                        }
                        q[0]=p[0];
                        for(auto i=1;i<nvar;i++)
                        {
                            vec temp= zeros(nvar,1);
                            for(auto l=0;l<=i-1;l++)
                            {
                                temp+=sum(q[l].t()*p[i-1])/sum(q[l].t()*q[l])*q[l];
                            }
                            q[i]=p[i]-temp;
                        }
                        for(auto i=0;i<nvar;i++)
                        {
                            d[i]=1/(norm(q[i]))*q[i];
                        }
                        delta=delta0;
                        continue;
                    }
                }
                else
                {
                    int flag=0;
                    umat c=(abs(delta)-eps<=zeros(nvar,1));
                    for(auto i=0;i<nvar;i++)
                    {
                        if(c(i,0)==0)
                            flag++;
                    }
                    if(flag==0)
                    {
                        x_fval result;
                        result.x=x;
                        result.fval=f(x);
                        return result;
                    }
                    else
                    {
                        j=1;
                        continue;
                    }
                }
            }
        }
    }
}



//fmin with linear constraints
//exterior-point
x_fval sci_arma::fmincon(const obj_fun &f, vec &x0, mat &A, mat &b)
{
    auto nvar=x0.n_rows;
    try{
        if((int)x0.n_cols!=1)
            throw std::logic_error("ERROR: x0 must be a column vector.");
    }
    catch(std::exception& e){
        std::cerr<<e.what()<<std::endl;
        std::terminate();
    }
    umat c=A*x0-b < zeros(nvar,1);
    int flag=0;
    for(auto i=0;i<nvar;i++)
    {
        if(c(i,0)==0)
            flag++;
    }
    try{
        if(flag>0)
            throw std::logic_error("ERROR: x0 is not in the constraints, or on the boundary.");
    }
    catch(std::exception& e){
        std::cerr<<e.what()<<std::endl;
        std::terminate();
    }
    double delta=1, C=1.1, eps=0.00001;
    auto x_=x0;
    auto P=[&](vec& x)->double {
        vec x1=b-A*x;
        double B=0;
        for(auto i=0;i<nvar;i++)
        {
            double temp=std::max(0.0,-x1(i));
            B+=temp*temp;
        }
        return delta*B;
    };
    while (true)
    {
        auto g=[&](vec& x)-> double {
            return f(x) + P(x);
        };
        auto result_=fmin(g,x_);
        x_=result_.x;
        if(P(x_) < eps)
        {
            x_fval result;
            result.x=x_;
            result.fval=f(x_);
            return result;
        }
        else
        {
            delta*=C;
        }
    }
}



//fmin with linear constraints
//'interior-point'
//this method is not well-implemented
/*
x_fval sci_arma::fmincon_i(const obj_fun &f, vec &x0, mat &A, mat &b)
{
    auto nvar=x0.n_rows;
    try{
        if((int)x0.n_cols!=1)
            throw std::logic_error("ERROR: x0 must be a column vector.");
    }
    catch(std::exception& e){
        std::cerr<<e.what()<<std::endl;
        std::terminate();
    }
    umat c=A*x0-b < zeros(nvar,1);
    int flag=0;
    for(auto i=0;i<nvar;i++)
    {
        if(c(i,0)==0)
            flag++;
    }
    try{
        if(flag>0)
            throw std::logic_error("ERROR: x0 is not in constraints, or on the boundary.");
    }
    catch(std::exception& e){
        std::cerr<<e.what()<<std::endl;
        std::terminate();
    }
    double r=1, beta=0.9, eps=0.0001;
    auto x_=x0;
    while (true)
    {
        auto g=[&](vec& x)-> double {
            return f(x)+r*std::abs(sum(ones(nvar,1)/(b-A*x)));
        };
        auto result_=fmin(g,x_);
        x_=result_.x;
        double temp=sum(ones(nvar,1)/(b-A*x_));
        if(r*std::abs(temp)<eps)
        {
            x_fval result;
            result.x=x_;
            result.fval=f(x_);
            return result;
        }
        else
        {
            r=beta*r;
        }
    }
}
*/

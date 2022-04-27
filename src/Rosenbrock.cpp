#include "sci_arma.h"

x_fval sci_arma::rosenbrock(const obj_fun& f, vec& x0)
//Rosenbrock Method
{
    auto nvar=x0.n_rows;
    try{
        if((int)x0.n_cols!=1)
            throw std::logic_error("ERROR: rosenbrock(): x0 must be a column vector.");
    }
    catch(std::exception& e){
        std::cerr<<e.what()<<std::endl;
        std::terminate();
    }
    mat E= eye(nvar,nvar);
    vec x = x0;
    auto d=new vec[nvar];
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
                        auto p=new vec[nvar],q=new vec[nvar];
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






#include "sci_arma.h"

x_fval sci_arma::powell_m(const obj_fun &f, vec &x0, const options& opt)
//modified Powell method in Sargent form
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
    double eps=0.000001;
    long long ite=0;
    while (true)
    {
        ite++;
        std::vector<vec>xn;
        std::vector<double>diff;
        xn.push_back(x1);
        auto x_kn=x1;
        for(auto i=0;i<nvar;i++)
        {
            auto range= include_min(f, x_kn, d[i]);
            x_kn= line_search(f, range[0], range[1]);
            xn.push_back(x_kn);
        }
        for(auto i=0;i<nvar;i++){
            diff.push_back(f(xn[i])-f(xn[i+1]));
        }
        auto max=std::max_element(diff.begin(), diff.end());
        auto m=std::distance(diff.begin(),max);
        vec d_n_1=x_kn-x1;
        if(norm(x_kn-x1)<=eps || ite>=opt.max_ite)
        {
            x_fval result;
            if(norm(x_kn-x1)> eps){
                result.if_forced_terminated=true;
            }else{
                result.if_forced_terminated=false;
            }
            result.algorithm="Powell_modified";
            result.ite_times=ite;
            result.x=x_kn;
            result.fval=f(x_kn);
            return result;
        }else
        {
            auto range= include_min(f, x1, d_n_1);
            auto x_kn_1= line_search(f, range[0], range[1]);
            vec temp_=x_kn_1-x1;
            vec temp=solve(d_n_1,temp_);
            double lambda= -sum(temp);
            temp=x1;
            x1=x_kn_1;
            if(norm(temp-x1)<=eps)
            {
                x_fval result;
                result.x=x1;
                result.fval=f(x1);
                return result;
            }else
            {
                double flag=(f(temp)-f(x1))/(*max);
                if(std::abs(lambda) > std::sqrt(flag))
                {
                    for(auto j=m;j<nvar-1;j++){
                        d[j]=d[j+1];
                    }
                    d[nvar-1]=d_n_1;
                }
            }
        }
    }
}
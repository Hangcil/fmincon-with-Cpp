#include "sci_arma.h"

x_fval sci_arma::karmarkar(vec& x0, vec& f, mat& A, vec& b, const options& opt)
{
    auto m = A.n_cols;
    try {
        if ((int)x0.n_cols != 1)
            throw std::logic_error("ERROR: linprog(): x0 must be a column vector.");
        if (f.n_rows != x0.n_rows)
            throw std::logic_error("ERROR: linprog(): x0 and f must be the same size.");
        if (A.n_rows != b.n_rows)
            throw std::logic_error("ERROR: linprog(): A and b have inconsistent sizes.");
        if (A.n_cols != x0.n_rows)
            throw std::logic_error("ERROR: linprog(): A and x0 have inconsistent sizes.");
    }
    catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
        std::terminate();
    }
    long long ite = 0;
    double gamma = 0.2;
    vec x = x0;
    while (true)
    {
        ite++;
        vec v = b - A * x;
        mat D = diagmat(ones(v.n_rows, 1) / v);
        mat temp1 = A.t() * D * D * A;
        mat temp = inv(A.t() * D * D * A);
        vec dx = temp * f;
        vec dv = -A * dx;
        std::vector<double> set;
        for (auto i = 0; i < m; i++)
        {
            if (dv(i) < 0)
                set.push_back(-v(i) / dv(i));
        }
        double lambda = 0;
        if (!set.empty())
        {
            lambda = gamma * *(std::min_element(set.begin(), set.end()));
        }
        vec x1 = x + lambda * dx;
        if (std::abs(norm(f.t() * x1 - f.t() * x) / norm(f.t() * x)) < opt.tolerance || ite >= opt.max_ite)
        {
            x_fval result;
            result.x = x1;
            result.fval = double(sum(f.t() * x1));
            result.ite_times = ite;
            if (std::abs(norm(f.t() * x1 - f.t() * x) / norm(f.t() * x)) >= opt.tolerance)
            {
                result.if_forced_terminated = true;
            }
            result.algorithm = "Karmarkar";
            return result;
        }
        x = x1;
    }
}
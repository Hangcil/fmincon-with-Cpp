#include "sci_arma.h"

x_fval sci_arma::linprog(vec& f, mat& A, vec& b, const options& opt)
{
	x_fval result0 = find_x0(f, A, b);
	if (!result0.warning.empty())
	{
		return result0;
	}
	else
	{
		auto result1 = karmarkar(result0.x, vec(-f), A, b, opt);
		result1.fval = double(sum(f.t() * result1.x));
		return result1;
	}
}


x_fval sci_arma::linprog(vec& f, mat& A, vec& b)
{
	options opt;
	return linprog(f, A, b, opt);
}
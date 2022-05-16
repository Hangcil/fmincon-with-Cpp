#include "sci_arma.h"

x_fval sci_arma::find_x0(vec& f, mat& A, vec& b)
{
	vec x0 = norm(b) / norm(A * f) * f;
	vec v = b - A * x0;
	umat c = v > zeros(v.n_rows, 1);
	int flag = 0;
	for (auto i = 0; i < v.n_rows; i++)
	{
		if (c(i, 0) == 0)
		{
			flag++;
		}
	}
	if (flag == 0)
	{
		x_fval result;
		result.x = x0;
		return result;
	}
	else
	{
		double M = 100000000;
		vec e = ones(v.n_rows, 1);
		std::vector<double> v_set;
		options opt;
		long long ite = 0;
		while (true)
		{
			ite++;
			v = b - A * x0;
			for (auto i = 0; i < v.n_rows; i++)
			{
				v_set.push_back(v(i));
			}
			double x_alpha = *std::min_element(v_set.begin(), v_set.end());
			x_alpha = 2 * std::abs(x_alpha) + 2;
			vec temp = { x_alpha };
			vec xa = join_cols(x0, temp);
			mat A_ = join_rows(A, -e);
			temp = { M };
			vec f_ = join_cols(-f, temp);
			auto result = karmarkar(xa, vec(-f_), A_, b, opt);
			xa = result.x;
			x0 = xa.rows(0, xa.n_rows - 2);
			c = x0 < zeros(x0.n_rows, 1);
			flag = 0;
			for (auto i = 0; i < x0.n_rows; i++)
			{
				if (c(i, 0) == 0)
				{
					flag++;
				}
			}
			if (flag == 0)
			{
				x_fval result;
				result.x = x0;
				return result;
			}
			c = x0 <= zeros(x0.n_rows, 1);
			flag = 0;
			for (auto i = 0; i < x0.n_rows; i++)
			{
				if (c(i, 0) == 0)
				{
					flag++;
				}
			}
			if (flag == x0.n_rows)
			{
				x_fval result;
				result.x = x0;
				result.warning = "no feasible inner point";
				return result;
			}
		}
	}
}
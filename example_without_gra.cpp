#include <iostream>
#include <cmath>
#include "sci_arma.h"
int main()
{
	// define a obj_fun
	obj_fun f = [](vec &x) -> double
	{
		return pow(1 - 0.1 * x(0), 2) + 100 * pow(x(1) - 0.5 * x(0) * x(0), 2) + pow(1 - 0.1 * x(2), 2) + 100 * pow(x(3) - 0.5 * x(2) * x(2), 2);
	};

	// define a start point
	vec x0 = {1, 1, 1, 1};

	// optimization without constraints
	// using default algorithm BFGS
	auto result1 = sci_arma::fmincon(f, x0);

	// define linear constraints Ax<=b
	mat A = {{1, 1, 1, 0}, {2, 3, 4, 5}};
	vec b = {3, 15};

	// optimization with inequality linear constraints
	auto result2 = sci_arma::fmincon(f, x0, A, b);

	// define linear equality constraints
	mat Aeq = {{1, 0, 2, 0}};
	vec beq = {3};

	// optimization with mixed linear constraints
	auto result3 = sci_arma::fmincon(f, x0, A, b, Aeq, beq);

	// define non-linear inequality constraints c(x)<=0
	auto c = [](vec &x) -> vec
	{
		vec temp = {25 * x(0) * x(0) + x(1) * x(1) + x(2) * x(2) + x(3) * x(3) - 50 * x(3)};
		return temp;
	};

	// optimization with mixed constraints
	// set options::algorithm from preset(Powell_modified) to Powell
	options opt;
	opt.algo = Powell;
	auto result4 = sci_arma::fmincon(f, x0, A, b, Aeq, beq, c, opt);

	// output
	std::cout << result1;
	std::cout << endl;
	std::cout << result2;
	std::cout << endl;
	std::cout << result3;
	std::cout << endl;
	std::cout << result4;
	return 0;
}

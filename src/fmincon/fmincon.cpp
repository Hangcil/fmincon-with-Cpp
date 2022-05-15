#include "sci_arma.h"


//fmin with linear constraints
//multiplier method
x_fval sci_arma::fmincon(const obj_fun& f, vec& x0, mat& A, mat& b, const options& opt)
{
	try {
		if ((int)x0.n_cols != 1 || (int)b.n_cols != 1)
			throw std::logic_error("ERROR: fmincon(): x0 and b must be a column vector.");
	}
	catch (std::exception& e) {
		std::cerr << e.what() << std::endl;
		std::terminate();
	}
	umat c = A * x0 - b <= zeros(A.n_rows, 1);
	int flag = 0;
	for (auto i = 0; i < A.n_rows; i++)
	{
		if (c(i, 0) == 0)
			flag++;
	}
	try {
		if (flag > 0)
			throw std::logic_error("ERROR: fmincon(): x0 does not meet inequality linear constraints.");
	}
	catch (std::exception& e) {
		std::cerr << e.what() << std::endl;
		std::terminate();
	}
	double alpha = 1.2, delta = 2, beta = 0.5, eps = opt.tolerance <= 0.0001 ? opt.tolerance : 0.0001;
	long long ite = 0;
	auto x = x0;
	auto nvar = x0.n_rows, ncon = b.n_rows;
	vec w = ones(ncon, 1);

	double exit_flag_pre = 0, exit_flag = 0;
	for (auto i = 0; i < ncon; i++)
	{
		vec a = A.row(i).t();
		exit_flag_pre += std::pow(std::max(-w(i) / delta, double(sum(-b(i) + a.t() * x))), 2.0);
	}
	exit_flag_pre = std::sqrt(exit_flag_pre);

	while (true)
	{
		ite++;
		auto phi = [&](vec x)->double {
			double temp = 0;
			for (auto i = 0; i < ncon; i++)
			{
				vec a = A.row(i).t();
				double temp_ = sum(b(i) - a.t() * x);
				temp += std::pow(std::max(0.0, w(i) - delta * temp_), 2.0) - std::pow(w(i), 2.0);
			}
			return f(x) + 0.5 * delta * temp;
		};
		x_fval result_;
		if (opt.algo == Powell)
		{
			result_ = powell(phi, x, opt);
		}
		else if (opt.algo == Rosenbrock)
		{
			result_ = rosenbrock(phi, x, opt);
		}
		else if (opt.algo == Powell_modified || opt.algo == preset || opt.algo == BFGS)
		{
			result_ = powell_m(phi, x, opt);
		}
		exit_flag = 0;
		auto x_ = result_.x;
		for (auto i = 0; i < ncon; i++)
		{
			vec a = A.row(i).t();
			exit_flag += std::pow(std::max(-w(i) / delta, double(sum(-b(i) + a.t() * x_))), 2.0);
		}
		exit_flag = std::sqrt(exit_flag);
		if (exit_flag <= eps || ite >= opt.max_ite)
		{
			x_fval result;
			if (exit_flag > eps) {
				result.if_forced_terminated = true;
			}
			else {
				result.if_forced_terminated = false;
			}
			if (opt.algo == Powell)
			{
				result.algorithm = "multiplier_Powell";
			}
			else if (opt.algo == Rosenbrock)
			{
				result.algorithm = "multiplier_Rosenbrock";
			}
			else if (opt.algo == Powell_modified || opt.algo == preset || opt.algo == BFGS)
			{
				result.algorithm = "multiplier_Powell_modified";
			}
			if (opt.algo == BFGS)
				result.warning = "BFGS is not suitable for multiplier method, changed with Powell_modified.";
			result.ite_times = ite;
			result.x = x_;
			result.fval = f(x_);
			return result;
		}
		else
		{
			if (exit_flag >= exit_flag_pre * beta)
			{
				delta *= alpha;
			}
			for (auto i = 0; i < ncon; i++)
			{
				vec a = A.row(i).t();
				w(i) = std::max(0.0, w(i) - delta * sum(b(i) - a.t() * x));
			}
			x = x_;
			exit_flag_pre = exit_flag;
		}
	}
}


x_fval sci_arma::fmincon(const obj_fun& f, vec& x0, vec& lb, vec& ub, const options& opt)
{
	auto nvar = x0.n_rows;
	umat c1 = x0 <= ub;
	int flag = 0;
	for (auto i = 0; i < nvar; i++)
	{
		if (c1(i, 0) == 0)
			flag++;
	}
	try {
		if (flag > 0)
			throw std::logic_error("ERROR: fmincon(): x0 exceeds the upper bound.");
	}
	catch (std::exception& e) {
		std::cerr << e.what() << std::endl;
		std::terminate();
	}
	c1 = x0 >= lb;
	flag = 0;
	for (auto i = 0; i < nvar; i++)
	{
		if (c1(i, 0) == 0)
			flag++;
	}
	try {
		if (flag > 0)
			throw std::logic_error("ERROR: fmincon(): x0 falls beneath the lower bound.");
	}
	catch (std::exception& e) {
		std::cerr << e.what() << std::endl;
		std::terminate();
	}
	mat E1 = eye(nvar, nvar);
	mat E2 = -E1;
	vec b = join_cols(-lb, ub);
	mat A = join_cols(E2, E1);
	return fmincon(f, x0, A, b, opt);
}


x_fval sci_arma::fmincon(const obj_fun& f, vec& x0, mat& A, mat& b, vec& lb, vec& ub, const options& opt)
{
	auto nvar = x0.n_rows;
	umat c1 = x0 <= ub;
	int flag = 0;
	for (auto i = 0; i < nvar; i++)
	{
		if (c1(i, 0) == 0)
			flag++;
	}
	try {
		if (flag > 0)
			throw std::logic_error("ERROR: fmincon(): x0 exceeds the upper bound.");
	}
	catch (std::exception& e) {
		std::cerr << e.what() << std::endl;
		std::terminate();
	}
	c1 = x0 >= lb;
	flag = 0;
	for (auto i = 0; i < nvar; i++)
	{
		if (c1(i, 0) == 0)
			flag++;
	}
	try {
		if (flag > 0)
			throw std::logic_error("ERROR: fmincon(): x0 falls beneath the lower bound.");
	}
	catch (std::exception& e) {
		std::cerr << e.what() << std::endl;
		std::terminate();
	}
	mat E1 = eye(nvar, nvar);
	mat E2 = -E1;
	vec b_ = join_cols(-lb, ub);
	mat A_ = join_cols(E2, E1);
	A_ = join_cols(A_, A);
	b_ = join_cols(b_, b);
	return fmincon(f, x0, A_, b_, opt);
}


x_fval sci_arma::fmincon(const obj_fun& f, vec& x0, vec& lb, vec& ub, const nonl_con& c, const options& opt)
{
	auto nvar = x0.n_rows;
	umat c1 = x0 <= ub;
	int flag = 0;
	for (auto i = 0; i < nvar; i++)
	{
		if (c1(i, 0) == 0)
			flag++;
	}
	try {
		if (flag > 0)
			throw std::logic_error("ERROR: fmincon(): x0 exceeds the upper bound.");
	}
	catch (std::exception& e) {
		std::cerr << e.what() << std::endl;
		std::terminate();
	}
	c1 = x0 >= lb;
	flag = 0;
	for (auto i = 0; i < nvar; i++)
	{
		if (c1(i, 0) == 0)
			flag++;
	}
	try {
		if (flag > 0)
			throw std::logic_error("ERROR: fmincon(): x0 falls beneath the lower bound.");
	}
	catch (std::exception& e) {
		std::cerr << e.what() << std::endl;
		std::terminate();
	}
	mat E1 = eye(nvar, nvar);
	mat E2 = -E1;
	vec b = join_cols(-lb, ub);
	mat A = join_cols(E2, E1);
	mat Aeq = zeros(1, nvar);
	vec beq = { 0 };
	return fmincon(f, x0, A, b, Aeq, beq, c, opt);
}


x_fval sci_arma::fmincon(const obj_fun& f, vec& x0, mat& A, mat& b, mat& Aeq, mat& beq, const options& opt)
{
	try {
		if ((int)x0.n_cols != 1)
			throw std::logic_error("ERROR: fmincon(): x0 must be a column vector.");
	}
	catch (std::exception& e) {
		std::cerr << e.what() << std::endl;
		std::terminate();
	}
	umat c = A * x0 - b <= zeros(A.n_rows, 1);
	int flag = 0;
	for (auto i = 0; i < A.n_rows; i++)
	{
		if (c(i, 0) == 0)
			flag++;
	}
	try {
		if (flag > 0)
			throw std::logic_error("ERROR: fmincon(): x0 does not meet inequality linear constraints.");
	}
	catch (std::exception& e) {
		std::cerr << e.what() << std::endl;
		std::terminate();
	}
	c = Aeq * x0 - beq == zeros(Aeq.n_rows, 1);
	flag = 0;
	for (auto i = 0; i < Aeq.n_rows; i++)
	{
		if (c(i, 0) == 0)
			flag++;
	}
	try {
		if (flag > 0)
			throw std::logic_error("ERROR: fmincon(): x0 does not meet equality linear constraints.");
	}
	catch (std::exception& e) {
		std::cerr << e.what() << std::endl;
		std::terminate();
	}
	double alpha = 1.2, delta = 2, beta = 0.5, eps = opt.tolerance <= 0.0001 ? opt.tolerance : 0.0001;
	long long ite = 0;
	auto x = x0;
	auto nvar = x0.n_rows, m = b.n_rows, l = beq.n_rows;
	vec w = ones(m, 1), v = ones(l, 1);

	//exit flag for inequality constraints
	double exit_flag_pre = 0, exit_flag = 0;
	for (auto i = 0; i < m; i++)
	{
		vec a = A.row(i).t();
		exit_flag_pre += std::pow(std::max(-w(i) / delta, double(sum(-b(i) + a.t() * x))), 2.0);
	}
	exit_flag_pre = std::sqrt(exit_flag_pre);

	while (true)
	{
		ite++;
		auto phi = [&](vec x)->double {
			double temp1 = 0;
			for (auto i = 0; i < m; i++)
			{
				vec a = A.row(i).t();
				double temp_ = sum(b(i) - a.t() * x);
				temp1 += std::pow(std::max(0.0, w(i) - delta * temp_), 2.0) - std::pow(w(i), 2.0);
			}
			double temp2 = 0;
			for (auto i = 0; i < l; i++)
			{
				vec a = Aeq.row(i).t();
				double temp_ = -v(i) * sum(beq(i) - a.t() * x);
				double temp__ = 0.5 * delta * std::pow(sum(beq(i) - a.t() * x), 2.0);
				temp2 += temp_ + temp__;
			}
			return f(x) + 0.5 / delta * temp1 + temp2;
		};
		x_fval result_;
		if (opt.algo == Powell)
		{
			result_ = powell(phi, x, opt);
		}
		else if (opt.algo == Rosenbrock)
		{
			result_ = rosenbrock(phi, x, opt);
		}
		else if (opt.algo == Powell_modified || opt.algo == preset || opt.algo == BFGS)
		{
			result_ = powell_m(phi, x, opt);
		}
		exit_flag = 0;
		auto x_ = result_.x;
		for (auto i = 0; i < m; i++)
		{
			vec a = A.row(i).t();
			exit_flag += std::pow(std::max(-w(i) / delta, double(sum(-b(i) + a.t() * x_))), 2.0);
		}
		exit_flag = std::sqrt(exit_flag);
		if ((exit_flag <= eps && norm(beq - Aeq * x_) <= eps) || ite >= opt.max_ite)
		{
			x_fval result;
			if (exit_flag > eps) {
				result.if_forced_terminated = true;
			}
			else {
				result.if_forced_terminated = false;
			}
			if (opt.algo == Powell)
			{
				result.algorithm = "multiplier_Powell";
			}
			else if (opt.algo == Rosenbrock)
			{
				result.algorithm = "multiplier_Rosenbrock";
			}
			else if (opt.algo == Powell_modified || opt.algo == preset || opt.algo == BFGS)
			{
				result.algorithm = "multiplier_Powell_modified";
			}
			if (opt.algo == BFGS)
				result.warning = "BFGS is not suitable for multiplier method, changed with Powell_modified.";
			result.ite_times = ite;
			result.x = x_;
			result.fval = f(x_);
			return result;
		}
		else
		{
			if (exit_flag >= exit_flag_pre * beta && norm(beq - Aeq * x_) >= norm(beq - Aeq * x) * beta)
			{
				delta *= alpha;
			}
			for (auto i = 0; i < m; i++)
			{
				vec a = A.row(i).t();
				w(i) = std::max(0.0, w(i) - delta * sum(b(i) - a.t() * x));
			}
			x = x_;
			exit_flag_pre = exit_flag;
		}
	}
}


x_fval sci_arma::fmincon(const obj_fun& f, vec& x0, mat& A, mat& b, mat& Aeq, mat& beq, vec& lb, vec& ub, const options& opt)
{
	auto nvar = x0.n_rows;
	umat c1 = x0 <= ub;
	int flag = 0;
	for (auto i = 0; i < nvar; i++)
	{
		if (c1(i, 0) == 0)
			flag++;
	}
	try {
		if (flag > 0)
			throw std::logic_error("ERROR: fmincon(): x0 exceeds the upper bound.");
	}
	catch (std::exception& e) {
		std::cerr << e.what() << std::endl;
		std::terminate();
	}
	c1 = x0 >= lb;
	flag = 0;
	for (auto i = 0; i < nvar; i++)
	{
		if (c1(i, 0) == 0)
			flag++;
	}
	try {
		if (flag > 0)
			throw std::logic_error("ERROR: fmincon(): x0 falls beneath the lower bound.");
	}
	catch (std::exception& e) {
		std::cerr << e.what() << std::endl;
		std::terminate();
	}
	mat E1 = eye(nvar, nvar);
	mat E2 = -E1;
	vec b_ = join_cols(-lb, ub);
	mat A_ = join_cols(E2, E1);
	A_ = join_cols(A_, A);
	b_ = join_cols(b_, b);
	return fmincon(f, x0, A_, b_, Aeq, beq, opt);
}


x_fval sci_arma::fmincon(const obj_fun& f, vec& x0, mat& A, mat& b, mat& Aeq, mat& beq, const nonl_con& c, const options& opt)
{
	try {
		if ((int)x0.n_cols != 1)
			throw std::logic_error("ERROR: fmincon(): x0 must be a column vector.");
	}
	catch (std::exception& e) {
		std::cerr << e.what() << std::endl;
		std::terminate();
	}
	umat com = A * x0 - b <= zeros(A.n_rows, 1);
	int flag = 0;
	for (auto i = 0; i < A.n_rows; i++)
	{
		if (com(i, 0) == 0)
			flag++;
	}
	try {
		if (flag > 0)
			throw std::logic_error("ERROR: fmincon(): x0 does not meet inequality linear constraints.");
	}
	catch (std::exception& e) {
		std::cerr << e.what() << std::endl;
		std::terminate();
	}
	com = Aeq * x0 - beq == zeros(Aeq.n_rows, 1);
	flag = 0;
	for (auto i = 0; i < Aeq.n_rows; i++)
	{
		if (com(i, 0) == 0)
			flag++;
	}
	try {
		if (flag > 0)
			throw std::logic_error("ERROR: fmincon(): x0 does not meet equality liear constraints.");
	}
	catch (std::exception& e) {
		std::cerr << e.what() << std::endl;
		std::terminate();
	}
	double alpha = 1.2, delta = 2, beta = 0.5, eps = opt.tolerance <= 0.0001 ? opt.tolerance : 0.0001;
	long long ite = 0;
	auto x = x0;
	vec test = c(x);
	auto nvar = x0.n_rows, m = b.n_rows, l = beq.n_rows;
	vec w = ones(m, 1), v = ones(l, 1);

	//exit flag for inequality constraints
	double exit_flag_pre = 0, exit_flag = 0;
	for (auto i = 0; i < m; i++)
	{
		vec a = A.row(i).t();
		exit_flag_pre += std::pow(std::max(-w(i) / delta, double(sum(-b(i) + a.t() * x))), 2.0);
	}
	for (auto i = 0; i < test.n_rows; i++)
	{
		exit_flag_pre += std::pow(std::max(-w(i) / delta, test(i)), 2.0);
	}
	exit_flag_pre = std::sqrt(exit_flag_pre);

	while (true)
	{
		ite++;
		auto phi = [&](vec x)->double {
			double temp1 = 0;
			vec nonl = c(x);
			for (auto i = 0; i < m; i++)
			{
				vec a = A.row(i).t();
				double temp_ = sum(b(i) - a.t() * x);
				temp1 += std::pow(std::max(0.0, w(i) - delta * temp_), 2.0) - std::pow(w(i), 2.0);
			}
			for (auto i = 0; i < nonl.n_rows; i++)
			{
				temp1 += std::pow(std::max(0.0, w(i) + delta * nonl(i)), 2.0) - std::pow(w(i), 2.0);
			}
			double temp2 = 0;
			for (auto i = 0; i < l; i++)
			{
				vec a = Aeq.row(i).t();
				double temp_ = -v(i) * sum(beq(i) - a.t() * x);
				double temp__ = 0.5 * delta * std::pow(sum(beq(i) - a.t() * x), 2.0);
				temp2 += temp_ + temp__;
			}
			return f(x) + 0.5 / delta * temp1 + temp2;
		};
		x_fval result_;
		if (opt.algo == Powell)
		{
			result_ = powell(phi, x, opt);
		}
		else if (opt.algo == Rosenbrock)
		{
			result_ = rosenbrock(phi, x, opt);
		}
		else if (opt.algo == Powell_modified || opt.algo == preset || opt.algo == BFGS)
		{
			result_ = powell_m(phi, x, opt);
		}
		exit_flag = 0;
		auto x_ = result_.x;
		vec nonl = c(x_);
		for (auto i = 0; i < m; i++)
		{
			vec a = A.row(i).t();
			exit_flag += std::pow(std::max(-w(i) / delta, double(sum(-b(i) + a.t() * x_))), 2.0);
		}
		for (auto i = 0; i < test.n_rows; i++)
		{
			exit_flag += std::pow(std::max(-w(i) / delta, nonl(i)), 2.0);
		}
		exit_flag = std::sqrt(exit_flag);
		if ((exit_flag <= eps && norm(beq - Aeq * x_) <= eps) || ite >= opt.max_ite)
		{
			x_fval result;
			if (exit_flag > eps) {
				result.if_forced_terminated = true;
			}
			else {
				result.if_forced_terminated = false;
			}
			if (opt.algo == Powell)
			{
				result.algorithm = "multiplier_Powell";
			}
			else if (opt.algo == Rosenbrock)
			{
				result.algorithm = "multiplier_Rosenbrock";
			}
			else if (opt.algo == Powell_modified || opt.algo == preset || opt.algo == BFGS)
			{
				result.algorithm = "multiplier_Powell_modified";
			}
			if (opt.algo == BFGS)
				result.warning = "BFGS is not suitable for multiplier method, changed with Powell_modified.";
			result.ite_times = ite;
			result.x = x_;
			result.fval = f(x_);
			return result;
		}
		else
		{
			if (exit_flag >= exit_flag_pre * beta && norm(beq - Aeq * x_) >= norm(beq - Aeq * x) * beta)
			{
				delta *= alpha;
			}
			for (auto i = 0; i < m; i++)
			{
				vec a = A.row(i).t();
				w(i) = std::max(0.0, w(i) - delta * sum(b(i) - a.t() * x));
			}
			x = x_;
			exit_flag_pre = exit_flag;
		}
	}
}


x_fval sci_arma::fmincon(const obj_fun& f, vec& x0, mat& A, mat& b, mat& Aeq, mat& beq, vec& lb, vec& ub, const nonl_con& c, const options& opt)
{
	auto nvar = x0.n_rows;
	umat c1 = x0 <= ub;
	int flag = 0;
	for (auto i = 0; i < nvar; i++)
	{
		if (c1(i, 0) == 0)
			flag++;
	}
	try {
		if (flag > 0)
			throw std::logic_error("ERROR: fmincon(): x0 exceeds the upper bound.");
	}
	catch (std::exception& e) {
		std::cerr << e.what() << std::endl;
		std::terminate();
	}
	c1 = x0 >= lb;
	flag = 0;
	for (auto i = 0; i < nvar; i++)
	{
		if (c1(i, 0) == 0)
			flag++;
	}
	try {
		if (flag > 0)
			throw std::logic_error("ERROR: fmincon(): x0 falls beneath the lower bound.");
	}
	catch (std::exception& e) {
		std::cerr << e.what() << std::endl;
		std::terminate();
	}
	mat E1 = eye(nvar, nvar);
	mat E2 = -E1;
	vec b_ = join_cols(-lb, ub);
	mat A_ = join_cols(E2, E1);
	A_ = join_cols(A_, A);
	b_ = join_cols(b_, b);
	return fmincon(f, x0, A_, b_, Aeq, beq, c, opt);
}


x_fval sci_arma::fmincon(const obj_fun& f, vec& x0, const options& opt)
{
	if (opt.algo == Rosenbrock)
	{
		return rosenbrock(f, x0, opt);
	}
	else if (opt.algo == Powell)
	{
		return powell(f, x0, opt);
	}
	else if (opt.algo == Powell_modified)
	{
		return powell_m(f, x0, opt);
	}
	else if (opt.algo == BFGS || opt.algo == preset)
	{
		return bfgs(f, x0, opt);
	}
}


x_fval sci_arma::fmincon(const obj_fun& f, vec& x0)
{
	options opt;
	return fmincon(f, x0, opt);
}


x_fval sci_arma::fmincon(const obj_fun& f, vec& x0, vec& lb, vec& ub)
{
	options opt;
	return fmincon(f, x0, lb, ub, opt);
}


x_fval sci_arma::fmincon(const obj_fun& f, vec& x0, mat& A, mat& b)
{
	options opt;
	return fmincon(f, x0, A, b, opt);
}


x_fval sci_arma::fmincon(const obj_fun& f, vec& x0, vec& lb, vec& ub, const nonl_con& c)
{
	options opt;
	return fmincon(f, x0, lb, ub, c, opt);
}


x_fval sci_arma::fmincon(const obj_fun& f, vec& x0, mat& A, mat& b, vec& lb, vec& ub)
{
	options opt;
	return fmincon(f, x0, A, b, lb, ub, opt);
}


x_fval sci_arma::fmincon(const obj_fun& f, vec& x0, mat& A, mat& b, mat& Aeq, mat& beq)
{
	options opt;
	return fmincon(f, x0, A, b, Aeq, beq, opt);
}


x_fval sci_arma::fmincon(const obj_fun& f, vec& x0, mat& A, mat& b, mat& Aeq, mat& beq, vec& lb, vec& ub)
{
	options opt;
	return fmincon(f, x0, A, b, Aeq, beq, lb, ub, opt);
}


x_fval sci_arma::fmincon(const obj_fun& f, vec& x0, mat& A, mat& b, mat& Aeq, mat& beq, const nonl_con& c)
{
	options opt;
	return fmincon(f, x0, A, b, Aeq, beq, c, opt);
}


x_fval sci_arma::fmincon(const obj_fun& f, vec& x0, mat& A, mat& b, mat& Aeq, mat& beq, vec& lb, vec& ub, const nonl_con& c)
{
	options opt;
	return fmincon(f, x0, A, b, Aeq, beq, lb, ub, c, opt);
}
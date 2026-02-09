#pragma once

#include <vector>
#include <stdexcept>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <cmath>
#include <string>

class Matrix {

private:

	std::vector<std::vector<double>> data;
	int rows;
	int cols;

public:

	Matrix(const std::vector<std::vector<double>>& matrix_data);
	Matrix(int m = 0, int n = 0);

	int getRows() const;
	int getCols() const;

	double& operator()(int i, int j);

	const double& operator()(int i, int j) const;

	void readFromFile(const std::string& filename);
	void readFromConsole();
	void print(const std::string& name = "") const;
	void printVector(const std::vector<double>& vec, const std::string& name = "");

	Matrix multiply(const Matrix& other) const;
	Matrix operator*(const Matrix& other) const;

	std::pair<Matrix, Matrix> luFactorization() const;
	std::vector<double> solve(const std::vector<double>& b) const;

	Matrix inverse() const;
	Matrix transpose() const;
	
	std::vector<double> solveUsingInverse(const std::vector<double>& b) const;
};
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

	std::vector<std::vector<int>> data;
	int rows;
	int cols;

public:

	Matrix(const std::vector<std::vector<int>>& matrix_data);
	Matrix(int m = 0, int n = 0);

	int getRows() const;
	int getCols() const;

	int& operator()(int i, int j);

	const int& operator()(int i, int j) const;

	void readFromFile(const std::string& filename);
	void readFromConsole();
	void print(const std::string& name = "") const;
	void printVector(const std::vector<int>& vec, const std::string& name = "");

	Matrix multiply(const Matrix& other) const;
	Matrix operator*(const Matrix& other) const;

	std::pair<Matrix, Matrix> luFactorization() const;
	std::vector<int> solve(const std::vector<int>& b) const;

	Matrix inverse() const;
	Matrix transpose() const;
	
	std::vector<int> solveUsingInverse(const std::vector<int>& b) const;
};
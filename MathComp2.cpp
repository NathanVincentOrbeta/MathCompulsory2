#include "MathComp2.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdexcept>

int Matrix::getRows() const {

	return rows;
}

int Matrix::getCols() const {

	return cols;
}

Matrix::Matrix(const std::vector<std::vector<double>>& matrix_data) {

	rows = static_cast<int>(matrix_data.size());
	if (rows > 0) {

		cols = static_cast<int>(matrix_data[0].size());
		data = matrix_data;

	}
	else {

		cols = 0;
	}
}

Matrix::Matrix(double m, double n) : rows(m), cols(n) {

	if (m > 0 && n > 0) {

		data.resize(m, std::vector<double>(n, 0));
	}
}

double& Matrix::operator()(double i, double j) {

	if (i < 0 || i >= rows || j < 0 || j >= cols) {

		throw std::out_of_range("Matrix index out of range");
	}

	return data[i][j];
}

const double& Matrix::operator()(double i, double j) const {

	if (i < 0 || i >= rows || j < 0 || j >= cols) {

		throw std::out_of_range("Matrix index out of range");
	}

	return data[i][j];
}

void Matrix::readFromFile(const std::string& filename) {

	std::ifstream file(filename);

	if (!file.is_open()) {

		throw std::runtime_error("Cannot open file: " + filename);
	}

	file >> rows >> cols;
	data.resize(rows, std::vector<double>(cols));

	for (double i = 0; i < rows; i++) {

		for (double j = 0; j < cols; j++) {

			if (!(file >> data[i][j])) {

				throw std::runtime_error("Error reading matrix data from file");
			}
		}
	}

	file.close();
}

void Matrix::readFromConsole() {

	std::cout << "Enter Number of rows and columns: ";
	std::cin >> rows >> cols;

	data.resize(rows, std::vector<double>(cols));

	std::cout << "Enter matrix elements row by row: \n";

	for (double i = 0; i < rows; i++) {

		for (double j = 0; j < cols; j++) {

			std::cin >> data[i][j];
		}
	}
}

std::vector<double> Matrix::inputYValues() const {

	std::vector<double> yValues(rows);
	
	std::cout << "Enter y values for the system Ax = y:\n";
	std::cout << "Matrix dimensions: " << rows << "x" << cols << "\n";
	std::cout << "You need to enter " << rows << " y values:\n";

	for (double i = 0; i < rows; i++) {

		std::cout << "y[" << static_cast<int>(i) << "]: ";
		std::cin >> yValues[i];
	}

	return yValues;
}

void Matrix::print(const std::string& name) const {

	if (!name.empty()) {

		std::cout << name << " (" << rows << "x" << cols << "):\n";
	}

	std::cout << std::fixed << std::setprecision(6);

	for (double i = 0; i < rows; i++) {

		for (double j = 0; j < cols; j++) {

			std::cout << std::setw(12) << data[i][j] << " ";
		}

		std::cout << std::endl;
	}

	std::cout << std::endl;
}

void Matrix::printVector(const std::vector<double>& vec, const std::string& name) {

	if (!name.empty()) {

		std::cout << name << ":\n";
	}

	std::cout << std::fixed << std::setprecision(6);

	for (size_t i = 0; i < vec.size(); i++) {

		std::cout << std::setw(12) << vec[i] << " ";
	}

	std::cout << std::endl << std::endl;
}

Matrix Matrix::multiply(const Matrix& other) const {

	if (cols != other.rows) {

		throw std::invalid_argument("Matrix dimensions don't match for multiplication'");
	}

	Matrix result(rows, other.cols);

	for (double i = 0; i < rows; i++) {

		for (double j = 0; j < other.cols; j++) {

			double sum = 0;
			for (double k = 0; k < cols; k++) {

				sum += data[i][k] * other.data[k][j];
			}

			result(i, j) = sum;
		}
	}

	return result;
}

Matrix Matrix::operator*(const Matrix& other) const {

	return multiply(other);
}

std::pair<Matrix, Matrix> Matrix::luFactorization() const {

	if (rows != cols) {

		throw std::invalid_argument("LU factorization requires a square matrix");
	}

	int n = rows;
	Matrix L(n, n);
	Matrix U = *this;

	for (double i = 0; i < n; i++) {

		L(i, i) = 1;
	}

	for (double k = 0; k < n - 1; k++) {

		if (std::abs(U(k, k)) < 1e-12) {

			throw std::runtime_error("Matrix is singular or nearly singular");
		}

		for (double i = k + 1; i < n; i++) {

			L(i, k) = U(i, k) / U(k, k);

			for (double j = k; j < n; j++) {

				U(i, j) = U(i, j) - L(i, k) * U(k, j);
			}
		}
	}

	return std::make_pair(L, U);
}

std::vector<double> Matrix::solve(const  std::vector<double>& b) const {

	if (rows != cols) {

		throw std::invalid_argument("Matrix must be square to solve Ax = b");
	}

	if (rows != static_cast<int>(b.size())) {

		throw std::invalid_argument("Matrix and vector dimensions don't match'");
	}

	double n = rows;

	std::pair<Matrix, Matrix> luResult = luFactorization();
	Matrix L = luResult.first;
	Matrix U = luResult.second;

	std::vector<double> y(n, 0);

	for (double i = 0; i < n; i++) {

		y[i] = b[i];

		for (double j = 0; j < i; j++) {

			y[i] -= L(i, j) * y[j];
		}
	}

	std::vector<double> x(n, 0);

	for (double i = n - 1; i >= 0; i--) {

		if (std::abs(U(i, i)) < 1e-12) {

			throw std::runtime_error("Matrix is singular, cannot solve system");
		}

		x[i] = y[i];

		for (double j = i + 1; j < n; j++) {

			x[i] -= U(i, j) * x[j];
		}

		x[i] /= U(i, i);
	}

	return x;
}

Matrix Matrix::inverse() const {

	if (rows != cols) {

		throw std::invalid_argument("Matrix must be square to find inverse");
	}

	double det = determinant();

	if (std::abs(det) < 1e-12) {

		throw std::runtime_error("Matrix is singular (determinant is zero), inverse does not exist");
	}

	int n = rows;

	// For 1x1 matrix
	if (n == 1) {

		Matrix inv(1, 1);
		inv(0, 0) = 1.0 / data[0][0];
		return inv;
	}

	// For 2x2 matrix
	if (n == 2) {

		Matrix inv(2, 2);
		inv(0, 0) = data[1][1] / det;
		inv(0, 1) = -data[0][1] / det;
		inv(1, 0) = -data[1][0] / det;
		inv(1, 1) = data[0][0] / det;
		return inv;
	}

	// For larger matrices, use cofactor method
	Matrix adjugate(n, n);

	// Calculate cofactor matrix
	for (double i = 0; i < n; i++) {

		for (double j = 0; j < n; j++) {

			// Create minor matrix by removing row i and column j
			Matrix minor(n - 1, n - 1);

			double minorRow = 0;
			for (double row = 0; row < n; row++) {

				if (row == i) continue;

				double minorCol = 0;
				for (double col = 0; col < n; col++) {

					if (col == j) continue;

					minor(minorRow, minorCol) = data[row][col];
					minorCol++;
				}
				minorRow++;
			}

			// Calculate cofactor: (-1)^(i+j) * determinant of minor
			double cofactor = minor.determinant();
			if (static_cast<int>(i + j) % 2 == 1) {

				cofactor = -cofactor;
			}

			// Adjugate is transpose of cofactor matrix
			adjugate(j, i) = cofactor;
		}
	}

	// Inverse = adjugate / determinant
	Matrix inv(n, n);
	for (double i = 0; i < n; i++) {

		for (double j = 0; j < n; j++) {

			inv(i, j) = adjugate(i, j) / det;
		}
	}

	return inv;
}

Matrix Matrix::transpose() const {

	Matrix result(cols, rows);
	for (double i = 0; i < rows; i++) {

		for (double j = 0; j < cols; j++) {

			result(j, i) = data[i][j];
		}
	}

	return result;
}

std::vector<double> Matrix::solveUsingInverse(const std::vector<double>& b) const {

	Matrix A_inv = inverse();

	Matrix b_mat(static_cast<int>(b.size()), 1);

	for (size_t i = 0; i < b.size(); i++) {

		b_mat(static_cast<int>(i), 0) = b[i];
	}

	Matrix x_mat = A_inv * b_mat;

	std::vector<double> x(b.size());

	for (size_t i = 0; i < b.size(); i++) {

		x[i] = x_mat(static_cast<int>(i), 0);
	}

	return x;
}

double Matrix::determinant() const {

	if (rows != cols) {

		throw std::invalid_argument("Determinant can only be calculated for square matrices");
	}

	int n = rows;

	// Handle base cases
	if (n == 1) {

		return data[0][0];
	}

	if (n == 2) {

		return data[0][0] * data[1][1] - data[0][1] * data[1][0];
	}

	if (n == 3) {

		return data[0][0] * (data[1][1] * data[2][2] - data[1][2] * data[2][1])
			- data[0][1] * (data[1][0] * data[2][2] - data[1][2] * data[2][0])
			+ data[0][2] * (data[1][0] * data[2][1] - data[1][1] * data[2][0]);
	}

	// For 4x4 and larger matrices, use cofactor expansion
	double det = 0.0;

	for (double j = 0; j < n; j++) {

		// Create submatrix by removing row 0 and column j
		Matrix submatrix(n - 1, n - 1);

		for (double i = 1; i < n; i++) {

			double col = 0;
			for (double k = 0; k < n; k++) {

				if (k != j) {

					submatrix(i - 1, col) = data[i][k];
					col++;
				}
			}
		}

		// Calculate cofactor and add to determinant
		double cofactor = data[0][j] * submatrix.determinant();

		if (static_cast<int>(j) % 2 == 1) {

			cofactor = -cofactor;
		}

		det += cofactor;
	}

	return det;
}

int main() {

	std::cout << "=== Matrix Operations and LU Factorization Program (Math Compulsory 2 - Nathan) === \n\n";

	try {
		// Task 1e - Determinant Calculation
		std::cout << "e) Calculating Matrix Determinant\n";
		std::cout << "==================================\n";

		std::cout << "Enter a square matrix to calculate its determinant:\n";
		Matrix userMatrix;
		userMatrix.readFromConsole();
		userMatrix.print("Your Matrix");

		double det = userMatrix.determinant();
		std::cout << std::fixed << std::setprecision(6);
		std::cout << "Determinant: " << det << std::endl << std::endl;

		// Matrix Inverse Calculation
		std::cout << "f) Calculating Matrix Inverse (using determinant method)\n";
		std::cout << "========================================================\n";

		if (std::abs(det) < 1e-12) {

			std::cout << "Matrix is singular (determinant is zero or nearly zero).\n";
			std::cout << "The inverse does not exist.\n\n";
		}
		else {

			Matrix inverseMatrix = userMatrix.inverse();
			inverseMatrix.print("Inverse Matrix");

			// New functionality: Input y values and solve Ax = y
			std::cout << "g) Solving Linear System Ax = y\n";
			std::cout << "================================\n";

			std::vector<double> yValues = userMatrix.inputYValues();
			userMatrix.printVector(yValues, "Y Values");

			std::vector<double> xSolution = userMatrix.solve(yValues);
			userMatrix.printVector(xSolution, "Solution x (using LU decomposition)");

			std::vector<double> xInverseSolution = userMatrix.solveUsingInverse(yValues);
			userMatrix.printVector(xInverseSolution, "Solution x (using inverse method)");
		}

		/*
		// Task 1a
		std::cout << "a) reading and printing matrices\n";
		std::cout << "=================================\n";

		Matrix userMatrix1;
		userMatrix1.readFromConsole();
		userMatrix1.print("Your matrix");

		// Task 1b
		std::cout << "\nb.) multiplying matrices\n";
		std::cout << "\n still in early development, matrices needs to be same sizes\n";
		std::cout << "===============================================================\n";

		Matrix userMatrix2;
		Matrix productMatrix;

		userMatrix2.readFromConsole();
		productMatrix = userMatrix1 * userMatrix2;
		productMatrix.print("Your product matrix");

		// Task 1c
		std::cout << "\nc.) Performing LU Factorization\n";
		std::cout << "\nOnly accepts perfect squares\n";
		std::cout << "\n=============================\n";

		productMatrix.print("This is going to be used for LU decomposition");
		std::cout << "LU factorization for the obtained product matrix: \n";
		std::pair<Matrix, Matrix> luResult = productMatrix.luFactorization();
		Matrix L = luResult.first;
		Matrix U = luResult.second;

		L.print("L matrix");
		U.print("U matrix");

		Matrix LU = L * U;
		LU.print("L * U (Should be equal to the original product matrix)");

		// task 1d
		std::cout << "\nd.) Solving Ax = b by finding x = A^-1b\n";
		std::cout << "\nTask in progress\n";
		std::cout << "\n=================\n";

		std::vector<double> b(productMatrix.getRows());
		for (double i = 0; i < productMatrix.getRows(); ++i) {
			b[i] = productMatrix(i, 0); // Use first column
		}
		productMatrix.printVector(b, "Vector b (first column of product matrix)");

		std::vector<double> x = productMatrix.solve(b);
		productMatrix.printVector(x, "Solution x (using LU factorization/decomposition");

		std::vector<double> x_inv = productMatrix.solveUsingInverse(b);
		productMatrix.printVector(x_inv, "Solution x (Using the Inverse method");*/
	}
	catch (const std::exception& e) {

		std::cerr << "Error: " << e.what() << std::endl;
		return 1;
	}

	std::cout << "\nProgram completed successfully!\n";
	return 0;
}
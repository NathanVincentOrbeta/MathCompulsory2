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

Matrix::Matrix(const std::vector<std::vector<int>>& matrix_data) {

	rows = static_cast<int>(matrix_data.size());
	if (rows > 0) {

		cols = static_cast<int>(matrix_data[0].size());
		data = matrix_data;

	}
	else {

		cols = 0;
	}
}

Matrix::Matrix(int m, int n) : rows(m), cols(n) {

	if (m > 0 && n > 0) {

		data.resize(m, std::vector<int>(n, 0));
	}
}

int& Matrix::operator()(int i, int j) {

	if (i < 0 || i >= rows || j < 0 || j >= cols) {

		throw std::out_of_range("Matrix index out of range");
	}

	return data[i][j];
}

const int& Matrix::operator()(int i, int j) const {

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
	data.resize(rows, std::vector<int>(cols));

	for (int i = 0; i < rows; i++) {

		for (int j = 0; j < cols; j++) {

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

	data.resize(rows, std::vector<int>(cols));

	std::cout << "Enter matrix elements row by row: \n";

	for (int i = 0; i < rows; i++) {

		for (int j = 0; j < cols; j++) {

			std::cin >> data[i][j];
		}
	}
}

void Matrix::print(const std::string& name) const {

	if (!name.empty()) {

		std::cout << name << " (" << rows << "x" << cols << "):\n";
	}

	std::cout << std::fixed << std::setprecision(6);

	for (int i = 0; i < rows; i++) {

		for (int j = 0; j < cols; j++) {

			std::cout << std::setw(12) << data[i][j] << " ";
		}

		std::cout << std::endl;
	}

	std::cout << std::endl;
}

void Matrix::printVector(const std::vector<int>& vec, const std::string& name) {

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

	for (int i = 0; i < rows; i++) {

		for (int j = 0; j < other.cols; j++) {

			double sum = 0.0;
			for (int k = 0; k < cols; k++) {

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

	for (int i = 0; i < n; i++) {

		L(i, i) = 1.0;
	}

	for (int k = 0; k < n - 1; k++) {

		if (std::abs(U(k, k)) < 1e-12) {

			throw std::runtime_error("Matrix is singular or nearly singular");
		}

		for (int i = k + 1; i < n; i++) {

			L(i, k) = U(i, k) / U(k, k);

			for (int j = k; j < n; j++) {

				U(i, j) = U(i, j) - L(i, k) * U(k, j);
			}
		}
	}

	return std::make_pair(L, U);
}

std::vector<int> Matrix::solve(const  std::vector<int>& b) const {

	if (rows != cols) {

		throw std::invalid_argument("Matrix must be square to solve Ax = b");
	}

	if (rows != static_cast<int>(b.size())) {

		throw std::invalid_argument("Matrix and vector dimensions don't match'");
	}

	int n = rows;

	std::pair<Matrix, Matrix> luResult = luFactorization();
	Matrix L = luResult.first;
	Matrix U = luResult.second;

	std::vector<double> y(n, 0.0);

	for (int i = 0; i < n; i++) {

		y[i] = b[i];

		for (int j = 0; j < i; j++) {

			y[i] -= L(i, j) * y[j];
		}
	}

	std::vector<int> x(n, 0);

	for (int i = n - 1; i >= 0; i--) {

		if (std::abs(U(i, i)) < 1e-12) {

			throw std::runtime_error("Matrix is singular, cannot solve system");
		}

		x[i] = y[i];

		for (int j = i + 1; j < n; j++) {

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

	int n = rows;
	Matrix inv(n, n);

	for (int col = 0; col < n; col++) {

		std::vector<int> b(n, 0);
		b[col] = 1;

		std::vector<int> x = solve(b);

		for (int row = 0; row < n; row++) {

			inv(row, col) = x[row];
		}
	}

	return inv;
}

Matrix Matrix::transpose() const {

	Matrix result(cols, rows);
	for (int i = 0; i < rows; i++) {

		for (int j = 0; j < cols; j++) {

			result(j, i) = data[i][j];
		}
	}

	return result;
}

std::vector<int> Matrix::solveUsingInverse(const std::vector<int>& b) const {

	Matrix A_inv = inverse();

	Matrix b_mat(static_cast<int>(b.size()), 1);

	for (size_t i = 0; i < b.size(); i++) {

		b_mat(static_cast<int>(i), 0) = b[i];
	}

	Matrix x_mat = A_inv * b_mat;

	std::vector<int> x(b.size());

	for (size_t i = 0; i < b.size(); i++) {

		x[i] = x_mat(static_cast<int>(i), 0);
	}

	return x;
}

int main() {

	std::cout << "=== Matrix Operations and LU Factorization Program (Math Compulsory 2 - Nathan) === \n\n";

	try {

		std::cout << "a) reading and printing matrices\n";
		std::cout << "=================================\n";

		Matrix userMatrix;
		userMatrix.readFromConsole();
		userMatrix.print("Your matrix");
		
	
		std::cout << "Example 1: Basic matrix operations\n";
		std::cout << "===================================\n";

		std::vector<std::vector<int>> A_data = {

			{4, 3, 2},
			{1, 2, 3},
			{2, 1, 4}
		};

		Matrix A(A_data);
		A.print("Matrix A");

		std::cout << "LU factorization of A:\n";
		std::pair<Matrix, Matrix> luResult = A.luFactorization();
		Matrix L = luResult.first;
		Matrix U = luResult.second;

		L.print("L matrix");
		U.print("U matrix");

		Matrix LU = L * U;
		LU.print("L * U (should equal A)");

		std::cout << "\nExample 2: Solving linear system Ax = b\n";
		std::cout << "==========================================\n";

		std::vector<int> b = { 1, 2, 3 };
		A.printVector(b, "Vector b");

		std::vector<int> x = A.solve(b);
		A.printVector(x, "Solution x (using LU factorization");

		std::vector<int> x_inv = A.solveUsingInverse(b);
		A.printVector(x_inv, "Solution x (using inverse)");

		std::cout << "\nExample 3: Matrix inverse\n";
		std::cout << "============================\n";

		Matrix A_inv = A.inverse();
		A_inv.print("Inverse of A");

		Matrix I = A * A_inv;
		I.print("A * A^{-1} (should be identity matrix");

		std::cout << "\nExample 4: User input demonstration\n";
		std::cout << "======================================\n";

		char choice;
		std::cout << "Do you want to enter your own matrix? (y/n): ";
		std::cin >> choice;

		if (choice == 'y' || choice == 'Y') {

			Matrix userMatrix;
			userMatrix.readFromConsole();
			userMatrix.print("Your matrix");

			if (userMatrix.getRows() == userMatrix.getCols()) {

				int n = userMatrix.getRows();
				std::vector<int> b_user(n);

				std::cout << "Enter right-hand side vector b (size " << n << "):\n";

				for (int i = 0; i < n; i++) {

					std::cin >> b_user[i];
				}

				A.printVector(b_user, "Vector b");

				std::vector<int> x_user = userMatrix.solve(b_user);
				A.printVector(x_user, "Solution x");

				Matrix user_inv = userMatrix.inverse();
				user_inv.print("Inverse of your matrix");
			}
		}

		std::cout << "\nExample 5: File input demonstration\n";
		std::cout << "======================================\n";

		std::ofstream sampleFile("sample_matrix.txt");

		if (sampleFile.is_open()) {

			sampleFile << "3 3\n";
			sampleFile << "2 1 -1\n";
			sampleFile << "-3 -1 2\n";
			sampleFile << "-2 1 2\n";
			sampleFile.close();

			std::cout << "Created sample file: sample_matrix.txt\n";

			Matrix fileMatrix;
			fileMatrix.readFromFile("sample_matrix.txt");
			fileMatrix.print("Matrix from file");

			std::vector<int> b_file = { 8, -11, -3 };
			std::vector<int> x_file = fileMatrix.solve(b_file);
			A.printVector(x_file, "Solution for file matrix");
		}

	}
	catch (const std::exception& e) {

		std::cerr << "Error: " << e.what() << std::endl;
		return 1;
	}

	std::cout << "\nProgram completed successfully!\n";
	return 0;
}
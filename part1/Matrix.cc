#include "Matrix.h"

// 默认构造函数：初始化全零矩阵
Matrix::Matrix(size_t rows, size_t cols)
    : rows_(rows), cols_(cols), data_(rows, std::vector<float>(cols, 0.0f)) {}

// 使用二维数组初始化
Matrix::Matrix(size_t rows, size_t cols, const std::vector<std::vector<float>> &data)
    : rows_(rows), cols_(cols), data_(data)
{
    if (data.size() != rows || data[0].size() != cols)
        throw std::invalid_argument("Data dimension mismatch");
}

// 拷贝构造函数
Matrix::Matrix(const Matrix &other)
    : rows_(other.rows_), cols_(other.cols_), data_(other.data_) {}

//重载赋值运算符
Matrix &Matrix::operator=(const Matrix &other)
{
    if (this != &other)
    {
        rows_ = other.rows_;
        cols_ = other.cols_;
        data_ = other.data_;
    }
    return *this;
}

// 元素的访问
float &Matrix::operator()(size_t row, size_t col)
{
    if (row >= rows_ || col >= cols_)
        throw std::out_of_range("Matrix index out of range");
    return data_[row][col];
}

const float &Matrix::operator()(size_t row, size_t col) const
{
    if (row >= rows_ || col >= cols_)
        throw std::out_of_range("Matrix index out of range");
    return data_[row][col];
}

// 矩阵的加法
Matrix Matrix::operator+(const Matrix &other) const
{
    if (rows_ != other.rows_ || cols_ != other.cols_)
        throw std::invalid_argument("Matrix dimensions do not match for addition");
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < rows_; ++i)
        for (size_t j = 0; j < cols_; ++j)
            result(i, j) = data_[i][j] + other.data_[i][j];
    return result;
}

// 矩阵的乘法
Matrix Matrix::operator*(const Matrix &other) const
{
    if (cols_ != other.rows_)
        throw std::invalid_argument("Matrix dimensions do not match for multiplication");
    Matrix result(rows_, other.cols_);
    for (size_t i = 0; i < rows_; ++i)
        for (size_t j = 0; j < other.cols_; ++j)
            for (size_t k = 0; k < cols_; ++k)
                result(i, j) += data_[i][k] * other.data_[k][j];
    return result;
}

// 打印矩阵
void Matrix::print() const
{
    for (const auto &row : data_)
    {
        for (const auto &val : row)
        {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}

// relu函数
Matrix Matrix::relu() const
{
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < rows_; ++i)
    {
        for (size_t j = 0; j < cols_; ++j)
        {
            result(i, j) = std::max(0.0f, data_[i][j]);
        }
    }
    return result;
}

// softmax函数
Matrix Matrix::softmax() const
{
    Matrix result(rows_, cols_);
    float sumExp = 0.0f;
    for (size_t i = 0; i < rows_; ++i)
    {

        for (size_t j = 0; j < cols_; ++j)
        {
            result(i, j) = std::exp(data_[i][j]);
            sumExp += result(i, j);
        }
    }
    for (size_t i = 0; i < rows_; ++i)
    {
        for (size_t j = 0; j < cols_; ++j)
        {
            result(i, j) /= sumExp; // 归一化
        }
    }
    return result;
}

// // 构造函数
// model::model(const std::vector<size_t> &layer_sizes)
// {
//     num_layers = layer_sizes.size();
//     weights.resize(num_layers - 1);
//     biases.resize(num_layers - 1);
//     weights[0] = Matrix(784, 500);
//     weights[1] = Matrix(500, 10);
//     biases[0] = Matrix(500, 1);
//     biases[1] = Matrix(10, 1);

//     // for (size_t i = 0; i < num_layers - 1; ++i) {
//     //     weights[i] = Matrix(layer_sizes[i + 1], layer_sizes[i]);
//     //     biases[i] = Matrix(layer_sizes[i + 1], 1);
//     //     // 初始化权重和偏置（这里简单初始化为随机值，可以根据需要改进）
//     //     for (size_t r = 0; r < weights[i].rows(); ++r) {
//     //         for (size_t c = 0; c < weights[i].cols(); ++c) {
//     //             weights[i](r, c) = static_cast<float>(rand()) / RAND_MAX; // 随机初始化
//     //         }
//     //         biases[i](r, 0) = static_cast<float>(rand()) / RAND_MAX; // 随机初始化
//     //     }
//     // }
// }

// 构造函数
model::model()
{

    weights.insert(weights.end(), {Matrix(784, 500), Matrix(500, 10)});
    biases.insert(biases.end(), {Matrix(1,500), Matrix(1,10)});

    // for (size_t i = 0; i < num_layers - 1; ++i) {
    //     weights[i] = Matrix(layer_sizes[i + 1], layer_sizes[i]);
    //     biases[i] = Matrix(layer_sizes[i + 1], 1);
    //     // 初始化权重和偏置（这里简单初始化为随机值，可以根据需要改进）
    //     for (size_t r = 0; r < weights[i].rows(); ++r) {
    //         for (size_t c = 0; c < weights[i].cols(); ++c) {
    //             weights[i](r, c) = static_cast<float>(rand()) / RAND_MAX; // 随机初始化
    //         }
    //         biases[i](r, 0) = static_cast<float>(rand()) / RAND_MAX; // 随机初始化
    //     }
    // }
}

//拷贝构造函数
model::model(const model &other)
    : weights(other.weights), biases(other.biases) {} 

Matrix model::forward(const Matrix &input)
{
    if (input.rows() != 1 || input.cols() != 784)
    {
        throw std::invalid_argument("Input dimension must be 1x784");
    }
    Matrix activation = input;
    activation = (activation * weights[0]  + biases[0]).relu();
    activation = (activation * weights[1] + biases[1]).softmax();
    return activation;
}

void model::print_parameters() const
{
    for (size_t i = 0; i < weights.size(); ++i)
    {
        std::cout << "Weights Layer " << i + 1 << ":\n";
        weights[i].print();
        std::cout << "Biases Layer " << i + 1 << ":\n";
        biases[i].print();
    }
}
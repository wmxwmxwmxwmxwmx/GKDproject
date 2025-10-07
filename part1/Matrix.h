#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>
using namespace std;
class Matrix
{
private:
    std::vector<std::vector<float>> data_;
    size_t rows_; // 行数
    size_t cols_; // 列数

public:
    // 构造函数
    Matrix(size_t rows, size_t cols);
    Matrix(size_t rows, size_t cols, const std::vector<std::vector<float>> &data);
    Matrix(const Matrix &other); // 拷贝构造

    // 元素访问
    float &operator()(size_t row, size_t col);
    const float &operator()(size_t row, size_t col) const;

    // 获取维度
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

    // 矩阵运算
    Matrix operator+(const Matrix &other) const;
    Matrix operator-(const Matrix &other) const;
    Matrix operator*(const Matrix &other) const;
    Matrix operator*(float scalar) const; // 标量乘法

    // 重载赋值运算符
    Matrix &operator=(const Matrix &other);

    // 打印矩阵
    void print() const;

    // RELU函数
    Matrix relu() const;

    // SoftMax函数
    Matrix softmax() const;
};

class model
{
private:
    std::vector<Matrix> weights;
    std::vector<Matrix> biases;
public:
    model();
    model(vector<vector<float>> fc1_weight, vector<vector<float>> fc1_bias, vector<vector<float>> fc2_weight, vector<vector<float>> fc2_bias);
    model(const model &other);
    Matrix forward(const Matrix &input);
    void print_parameters() const;
};
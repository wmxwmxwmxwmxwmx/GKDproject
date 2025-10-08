#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>
using namespace std;

// typedef float ValType;

template <typename T>
class Matrix
{
private:
    std::vector<std::vector<T>> data_;
    size_t rows_; // 行数
    size_t cols_; // 列数

public:
    // 构造函数
    Matrix(size_t rows, size_t cols);
    Matrix(size_t rows, size_t cols, const std::vector<std::vector<T>> &data);
    Matrix(const Matrix &other); // 拷贝构造

    // 元素访问
    T &operator()(size_t row, size_t col);
    const T &operator()(size_t row, size_t col) const;

    // 获取维度
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

    // 矩阵运算
    Matrix<T> operator+(const Matrix<T> &other) const;
    Matrix<T> operator-(const Matrix<T> &other) const;
    Matrix<T> operator*(const Matrix<T> &other) const;

    // 重载赋值运算符
    Matrix<T> &operator=(const Matrix<T> &other);

    // 打印矩阵
    void print() const;

    // RELU函数
    Matrix<T> relu() const;

    // SoftMax函数
    Matrix<T> softmax() const;
};

// 基础模型类
class modelbase
{

public:
    modelbase(const string &path = "") : _path(path) {}
    virtual ~modelbase() = default;
    virtual const void predict() const = 0; // 纯虚函数
    std::string _path;
};

template <typename T>
class model : public modelbase
{
private:
    std::vector<Matrix<T>> weights;
    std::vector<Matrix<T>> biases;

public:
    model(const string &path = "");
    model(const model &other);
    Matrix<T> _predict(const Matrix<T> &input) const; // 预测函数
    virtual const void predict() const;               // 包装函数
};

// 函数实现

// 默认构造函数：初始化全零矩阵
template <typename T>
Matrix<T>::Matrix(size_t rows, size_t cols)
    : rows_(rows), cols_(cols), data_(rows, std::vector<T>(cols, static_cast<T>(0))) {}

// 使用二维数组初始化
template <typename T>
Matrix<T>::Matrix(size_t rows, size_t cols, const std::vector<std::vector<T>> &data)
    : rows_(rows), cols_(cols), data_(data)
{
    if (data.size() != rows || data[0].size() != cols)
        throw std::invalid_argument("Data dimension mismatch");
}

// 拷贝构造函数
template <typename T>
Matrix<T>::Matrix(const Matrix &other)
    : rows_(other.rows_), cols_(other.cols_), data_(other.data_) {}

// 重载赋值运算符
template <typename T>
Matrix<T> &Matrix<T>::operator=(const Matrix &other)
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
template <typename T>
T &Matrix<T>::operator()(size_t row, size_t col)
{
    if (row >= rows_ || col >= cols_)
        throw std::out_of_range("Matrix index out of range");
    return data_[row][col];
}

template <typename T>
const T &Matrix<T>::operator()(size_t row, size_t col) const
{
    if (row >= rows_ || col >= cols_)
        throw std::out_of_range("Matrix index out of range");
    return data_[row][col];
}

// 矩阵的加法
template <typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T> &other) const
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
template <typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T> &other) const
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
template <typename T>
void Matrix<T>::print() const
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
template <typename T>
Matrix<T> Matrix<T>::relu() const
{
    Matrix<T> result(rows_, cols_);
    for (size_t i = 0; i < rows_; ++i)
    {
        for (size_t j = 0; j < cols_; ++j)
        {
            result(i, j) = std::max(static_cast<T>(0), data_[i][j]);
        }
    }
    return result;
}

// softmax函数
template <typename T>
Matrix<T> Matrix<T>::softmax() const
{
    Matrix<T> result(rows_, cols_);
    T sumExp = static_cast<T>(0);
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

// 构造函数
template <typename T>
model<T>::model(const string &path)
    : modelbase(path)
{

    // 读取二进制文件
    vector<vector<T>> data[4];
    vector<int> row, col;
    if (path.ends_with("plus"))
    {
        row = {784, 1, 1000, 1};
        col = {1000, 1000, 10, 10};
    }
    else
    {
        row = {784, 1, 500, 1};
        col = {500, 500, 10, 10};
    }

    string filename[] = {_path + "/fc1.weight", _path + "/fc1.bias", _path + "/fc2.weight", _path + "/fc2.bias"};
    for (int k = 0; k < 4; k++)
    {
        FILE *pf = fopen(filename[k].c_str(), "rb");
        if (!pf)
        {
            printf("fopen error!\n");
            exit(-1);
        }

        data[k].resize(row[k], vector<T>(col[k], static_cast<T>(0))); // 初始化为0
        for (int i = 0; i < row[k]; i++)                              // 按行读取
        {
            fread(data[k][i].data(), sizeof(T), col[k], pf); // 读取一行数据
        }
        fclose(pf);
    }
    weights.insert(weights.end(), {Matrix<T>(row[0], col[0], data[0]), Matrix<T>(row[2], col[2], data[2])});
    biases.insert(biases.end(), {Matrix<T>(row[1], col[1], data[1]), Matrix<T>(row[3], col[3], data[3])});
}

// 拷贝构造函数
template <typename T>
model<T>::model(const model &other)
    : weights(other.weights), biases(other.biases) {}

// 预测函数
template <typename T>
Matrix<T> model<T>::_predict(const Matrix<T> &input) const
{
    if (input.rows() != 1 || input.cols() != 784)
    {
        throw std::invalid_argument("Input dimension must be 1x784");
    }
    Matrix<T> activation = input;
    activation = (activation * weights[0] + biases[0]).relu();
    activation = (activation * weights[1] + biases[1]).softmax();
    return activation;
}

// 包装函数
template <typename T>
const void model<T>::predict() const
{
    // 读取图像
    cv::Mat image = cv::imread("/home/wmx/桌面/project/GKDproject/project/num/0.png");
    if (image.empty())
    {
        cout << "无法打开图片！" << endl;
        exit(-1);
    }

    // 转换为灰度图像
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    // 图像缩放
    cv::Mat resized_down;
    int down_width = 28;
    int down_height = 28;
    cv::resize(grayImage, resized_down, cv::Size(down_width, down_height), cv::INTER_LINEAR);

    // 转换为Matrix
    Matrix<T> input(1, 784);
    for (int i = 0; i < down_height; i++)
    {
        for (int j = 0; j < down_width; j++)
        {
            input(0, i * down_width + j) = resized_down.at<uchar>(i, j) / static_cast<T>(255.0); // 归一化到0~1
        }
    }
    _predict(input).print();
}
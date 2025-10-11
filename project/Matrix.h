#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <thread>


#define SERVER_IP "127.0.0.1"
#define SERVER_PORT 8080
#define BUFFER_SIZE 1024

using namespace std;
template <typename T>
class Matrix
{
private:
    vector<vector<T>> data_;
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
    virtual const void predict(cv::Mat image) const = 0; // 纯虚函数
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
    Matrix<float> socket_predict(const Matrix<float> &input) const; // 有socket通信的预测函数
    Matrix<T> _predict(const Matrix<T> &input) const; // 预测函数
    virtual const void predict(cv::Mat image) const;  // 包装预测函数
    void drawBarChart(const std::vector<T> &values, const std::string &windowName = "predict", int displayWidth = 800, int displayHeight = 600) const;
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
    // 基础版
    // for (size_t i = 0; i < rows_; ++i)
    //     for (size_t j = 0; j < other.cols_; ++j)
    //         for (size_t k = 0; k < cols_; ++k)
    //             result(i, j) += data_[i][k] * other.data_[k][j];

    // 优化版
    int ThreadNum = std::thread::hardware_concurrency(); // 获取CPU核心数
    std::vector<std::thread> threads;                    // 存放线程

    // 将任务分配给多个线程
    int rowsPerThread = rows_ / ThreadNum; // 每个线程处理的行数
    for (int i = 0; i < ThreadNum; ++i)    // 创建线程
    {
        int startRow = i * rowsPerThread;                                     // 起始行
        int endRow = (i == ThreadNum - 1) ? rows_ : startRow + rowsPerThread; // 结束行,最后一个线程处理剩余行

        // 创建线程并传递参数,创建时就会立即启动子线程
        // lambda表达式传递参数
        threads.emplace_back([this, startRow, endRow, &other, &result]()
                             {
                for (size_t i = startRow; i < endRow; ++i) {
                    for (size_t j = 0; j < other.cols(); ++j) {
                        for (size_t k = 0; k < this->cols_; ++k) {
                            result(i, j) += this->data_[i][k] * other.data_[k][j];
                        }
                    }
                } });
    }

    // 按顺序等待所有子线程完成
    for (auto &t : threads)
    {
        if (t.joinable()) // 检查线程是否可连接
            t.join();     // 等待线程完成
    }

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

// 预测函数(无socket通信)
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

//预测函数(有socket通信)
template <>
Matrix<float> model<float>::socket_predict(const Matrix<float> &input) const
{

    if (input.rows() != 1 || input.cols() != 784)
    {
        throw std::invalid_argument("Input dimension must be 1x784");
    }


    // socket通信部分

    int clientSocket;
    struct sockaddr_in serverAddr;
    float buffer[BUFFER_SIZE];

    // 创建客户端套接字
    clientSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (clientSocket == -1) {
        perror("Failed to create socket");
        exit(EXIT_FAILURE);
    }

    // 设置服务器地址信息
    memset(&serverAddr, 0, sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(SERVER_PORT);
    if (inet_pton(AF_INET, SERVER_IP, &(serverAddr.sin_addr)) <=0) {
        perror("Failed to set server IP");
        exit(EXIT_FAILURE);
    }

    // 连接到服务器
    if (connect(clientSocket, (struct sockaddr *)&serverAddr, sizeof(serverAddr)) == -1) {
        perror("Failed to connect to server");
        exit(EXIT_FAILURE);
    }

    printf("Connected to server %s:%d\n", SERVER_IP, SERVER_PORT);

  
    // 复制数据到缓冲区
    for (size_t i = 0; i < 784; ++i) {
        buffer[i] = input(0, i);
    }
    // 发送数据
    if (send(clientSocket, buffer, 784 * sizeof(float), 0) == -1) {
        perror("Failed to send data");
        exit(EXIT_FAILURE);
    }

    // 接收响应
    memset(buffer, 0, BUFFER_SIZE*sizeof(float));
    if (recv(clientSocket, buffer, BUFFER_SIZE*sizeof(float), 0) == -1) {
        perror("Failed to receive data");
        exit(EXIT_FAILURE);
    }

    // 处理接收到的数据
    Matrix<float> output(1,10);
    for (size_t i = 0; i < 10; ++i) {
        output(0, i) = buffer[i];
    }
    printf("Received response from server.\n");
    // 关闭套接字
    close(clientSocket);

    return output;
}



// 包装预测函数
template <typename T>
const void model<T>::predict(cv::Mat image) const
{

    // 转换为灰度图像
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    // 图像缩放
    cv::Mat resized_down;
    int down_width = 28;
    int down_height = 28;
    cv::resize(grayImage, resized_down, cv::Size(down_width, down_height), cv::INTER_LINEAR);

    // 转换为Matrix<T>
    Matrix<T> input(1, 784);
    for (int i = 0; i < down_height; i++)
    {
        for (int j = 0; j < down_width; j++)
        {
            input(0, i * down_width + j) = resized_down.at<uchar>(i, j) / static_cast<T>(255.0); // 归一化到0~1
        }
    }

    // auto start = std::chrono::high_resolution_clock::now(); // 记录开始时间

    //Matrix<T> output = _predict(input);
    Matrix<float> output = socket_predict(input);

    // auto end = std::chrono::high_resolution_clock::now();                                          // 记录结束时间
    // auto duration_us = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); // 计算时间差，单位为毫秒
    // std::cout << "计算用时: " << duration_us << " 毫秒" << std::endl;

    drawBarChart(std::vector<T>{
                     output(0, 0), output(0, 1), output(0, 2), output(0, 3), output(0, 4),
                     output(0, 5), output(0, 6), output(0, 7), output(0, 8), output(0, 9)},
                 "predict", 800, 600);
}

// 绘制柱状图函数
template <typename T>
void model<T>::drawBarChart(const std::vector<T> &values, const std::string &windowName, int displayWidth, int displayHeight) const
{
    // 参数检查
    if (values.size() != 10)
    {
        std::cerr << "Error: Input vector size must be 10." << std::endl;
        return;
    }

    // 1. 创建一个白色背景的画布
    cv::Mat image(displayHeight, displayWidth, CV_8UC3, cv::Scalar(255, 255, 255));

    // 2. 定义柱状图参数
    int numBars = values.size(); // 柱条数量，这里是10
    int margin = 50;             // 画布的边距，为顶部和底部留出空间写标签
    int chartTop = margin;
    int chartBottom = displayHeight - margin;
    int chartHeight = chartBottom - chartTop;
    int totalBarAreaWidth = displayWidth - 2 * margin; // 柱条区域的总宽度
    int barWidth = totalBarAreaWidth / (numBars * 2);  // 每个柱条的宽度 (相邻柱条间会有间隔)
    int barSpacing = barWidth;                         // 柱条之间的间隔，这里设置为与柱宽相等

    // 3. 查找向量中的最大值（用于缩放柱条高度）
    float maxValue = *std::max_element(values.begin(), values.end());
    // 如果最大值为0，避免除以0，并设置一个最小缩放
    if (maxValue == static_cast<T>(0))
        maxValue = static_cast<T>(1);

    // 4. 绘制每个柱条和标签
    for (int i = 0; i < numBars; ++i)
    {
        // 计算当前柱条的水平起始位置（x坐标）
        int x = margin + i * (barWidth + barSpacing);

        // 根据值计算柱条高度（缩放至图表高度）
        int barHeight = static_cast<int>((values[i] / maxValue) * chartHeight);
        // 计算柱条在图像上的垂直起始位置（y坐标），OpenCV中y轴向下为正
        int y = chartBottom - barHeight;

        // 选择颜色：例如蓝色柱条，你也可以根据值的大小映射颜色
        cv::Scalar color = cv::Scalar(255, 0, 0); // BGR格式: 此处为蓝色

        // 绘制柱条（填充矩形）
        cv::rectangle(image,
                      cv::Point(x, y),                      // 矩形左上角
                      cv::Point(x + barWidth, chartBottom), // 矩形右下角
                      color,                                // 颜色
                      -1);                                  // 厚度-1表示填充

        // 5. 在柱条上方绘制数值标签
        std::string valueLabel = std::to_string(values[i]);
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(valueLabel, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseline);
        cv::putText(image,
                    valueLabel,
                    cv::Point(x + (barWidth - textSize.width) / 2, y - 5), // 位置：柱条上方居中
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.4,                 // 字体大小
                    cv::Scalar(0, 0, 0), // 黑色文字
                    1);                  // 线宽

        // 6. 在柱条下方绘制对应的数字标签（0-9）
        std::string numberLabel = std::to_string(i);
        cv::Size numTextSize = cv::getTextSize(numberLabel, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::putText(image,
                    numberLabel,
                    cv::Point(x + (barWidth - numTextSize.width) / 2, chartBottom + numTextSize.height + 5),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,                 // 字体大小
                    cv::Scalar(0, 0, 0), // 黑色文字
                    1);
    }

    // 7. 显示图像
    cv::imshow(windowName, image);
}
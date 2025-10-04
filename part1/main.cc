#include "Matrix.h"

int main()
{
    // // 创建两个矩阵
    // Matrix A(2, 3, {{1, 2, 3}, {4, 5, 6}});
    // Matrix B(3, 2, {{7, 8}, {9, 10}, {11, 12}});

    // // 矩阵乘法
    // Matrix C = A * B;

    // // 打印结果
    // std::cout << "Matrix A:" << std::endl;
    // A.print();
    // std::cout << "Matrix B:" << std::endl;
    // B.print();
    // std::cout << "Matrix C = A * B:" << std::endl;
    // C.print();

    //  Matrix A(3, 1, {{1}, {2}, {3}});
    //  Matrix D = A.softmax();
    //  std::cout << "softmax(D):" << std::endl;
    //  D.print();
    model my_model;
    Matrix ret = my_model.forward(Matrix(1, 784));
    ret.print();
    return 0;
}
#include "Matrix.h"
 

// #include <opencv2/opencv.hpp>
// int main()
// {
//     // // 创建两个矩阵
//     // Matrix A(2, 3, {{1, 2, 3}, {4, 5, 6}});
//     // Matrix B(3, 2, {{7, 8}, {9, 10}, {11, 12}});

//     // // 矩阵乘法
//     // Matrix C = A * B;

//     // // 打印结果
//     // std::cout << "Matrix A:" << std::endl;
//     // A.print();
//     // std::cout << "Matrix B:" << std::endl;
//     // B.print();
//     // std::cout << "Matrix C = A * B:" << std::endl;
//     // C.print();

//     //  Matrix A(3, 1, {{1}, {2}, {3}});
//     //  Matrix D = A.softmax();
//     //  std::cout << "softmax(D):" << std::endl;
//     //  D.print();
//     // model my_model;
//     // Matrix ret = my_model.forward(Matrix(1, 784));
//     // ret.print();
    
//     //读取二进制文件
//     vector<vector<float>> data[4];
//     int row[] = {784, 1, 500, 1};
//     int col[] = {500, 500, 10, 10};
//     char *filename[] = {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"};
//     for (int k = 0; k < 4; k++)
//     {
//         FILE *pf = fopen(filename[k], "rb");
//         if (!pf)
//         {
//             printf("fopen error!\n");
//             return -1;
//         }

//         data[k].resize(row[k], vector<float>(col[k], 0));
//         for (int i = 0; i < row[k]; i++)
//         {
//             fread(data[k][i].data(), sizeof(float), col[k], pf);
//         }
//         fclose(pf);
//     }
//     model my_model(data[0], data[1], data[2], data[3]);
//     Matrix ret = my_model.forward(Matrix(1, 784));
//     ret.print();
//     //my_model.print_parameters();
//     //    for(int i=0;i<row;i++)
//     //    {
//     //        for(int j=0;j<col;j++)
//     //        {
//     //            printf("%f ",data[i][j]);
//     //        }
//     //        printf("\n");
//     //    }
//     return 0;
// }
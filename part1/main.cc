#include "Matrix.h"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
int main()
{
    // 读取图像
    Mat image = imread("/home/wmx/桌面/project/GKDproject/part1/num/0.png");
    if (image.empty())
    { 
         cout << "无法打开图片！" << endl;
        return -1;
    }

    // 转换为灰度图像
    Mat grayImage;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);

    // 图像缩放
    Mat resized_down;
    int down_width = 28;
    int down_height = 28;
    resize(grayImage, resized_down, Size(down_width, down_height), INTER_LINEAR);


    // // 显示结果
    // imshow("Original Image", image);
    // //imshow("Blurred Image", blurredImage);
    // imshow("Resized Image", resized_down);
    // waitKey(0);
    // 转换为Matrix
    Matrix input(1, 784);
    for (int i = 0; i < down_height; i++)
    {
        for (int j = 0; j < down_width; j++)
        {
            input(0, i * down_width + j) = resized_down.at<uchar>(i, j) / 255.0; // 归一化到0~1
        }
    }
    //读取二进制文件
    vector<vector<float>> data[4];
    int row[] = {784, 1, 500, 1};
    int col[] = {500, 500, 10, 10};
   const char *filename[] = {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"};
    for (int k = 0; k < 4; k++)
    {
        FILE *pf = fopen(filename[k], "rb");
        if (!pf)
        {
            printf("fopen error!\n");
            return -1;
        }

        data[k].resize(row[k], vector<float>(col[k], 0));
        for (int i = 0; i < row[k]; i++)
        {
            fread(data[k][i].data(), sizeof(float), col[k], pf);
        }
        fclose(pf);
    }
    model my_model(data[0], data[1], data[2], data[3]);
    Matrix ret = my_model.forward(input);
    ret.print();
    // //my_model.print_parameters();
    // //    for(int i=0;i<row;i++)
    // //    {
    // //        for(int j=0;j<col;j++)
    // //        {
    // //            printf("%f ",data[i][j]);
    // //        }
    // //        printf("\n");
    // //    }

    

    return 0;
}
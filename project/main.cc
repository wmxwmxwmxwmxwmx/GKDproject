#include "Matrix.h"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// 全局变量用于存储绘图状态和图像
Mat gCanvas;          // 画布矩阵
bool Drawing = false; // 标记是否正在绘制
Point PreviousPoint;  // 记录上一个鼠标位置，用于画连续线

// 鼠标回调函数
void onMouse(int event, int x, int y, int flags, void *userdata)
{
    // 将userdata转换为Mat指针，获取我们的画布
    Mat *pCanvas = (Mat *)(userdata);

    switch (event)
    {
    // 左键按下 - 开始绘制
    case EVENT_LBUTTONDOWN:
        Drawing = true;
        PreviousPoint = Point(x, y); // 记录起始点
        break;

    // 左键释放 - 结束绘制
    case EVENT_LBUTTONUP:
        if (Drawing)
        {
            Drawing = false;
            // 在起点和终点之间画一条线（一次性线段）
            line(*pCanvas, PreviousPoint, Point(x, y), Scalar(0, 0, 0),10, LINE_AA);
            imshow("drawing", *pCanvas);
        }
        break;

    // 鼠标移动(并且左键按下) - 如果正在绘制，则画线
    case EVENT_MOUSEMOVE:
        if (Drawing && (flags & EVENT_FLAG_LBUTTON))
        {
            Point currentPoint(x, y);
            // 在上一个点和当前点之间画线，实现笔触效果
            line(*pCanvas, PreviousPoint, currentPoint, Scalar(0, 0, 0), 10, LINE_AA);
            PreviousPoint = currentPoint;      // 更新上一个点
            imshow("drawing", *pCanvas); // 实时更新显示
        }
        break;

    // 右键按下 - 清除画布
    case EVENT_RBUTTONDOWN:
        *pCanvas = Scalar(255, 255, 255); // 用白色填充，清除画布
        imshow("drawing", *pCanvas);
        break;
    }
    const modelbase& mb= model<float>("/home/wmx/桌面/project/GKDproject/project/mnist-fc");
    mb.predict(gCanvas);
}

int main()
{
    //    const modelbase& mb1= model<double>("/home/wmx/桌面/project/GKDproject/project/mnist-fc-plus");
    //    const modelbase& mb2= model<float>("/home/wmx/桌面/project/GKDproject/project/mnist-fc");

    //    mb1.predict();
    //    mb2.predict();

    // 初始化一个白色画布
    const int width = 120;
    const int height = 100;
    gCanvas = Mat(height, width, CV_8UC3, Scalar(255, 255, 255));

    // 创建窗口
    namedWindow("drawing", WINDOW_AUTOSIZE);

    // 设置鼠标回调函数，并将画布作为userdata传入
    setMouseCallback("drawing", onMouse, &gCanvas);

    // 显示初始画布
    imshow("drawing", gCanvas);

    // 主循环：处理键盘输入
    while (true)
    {
        int key = waitKey(0); // 等待按键

        switch (key)
        {
        case 's': // 保存图像
            imwrite("my_drawing.png", gCanvas);
            cout << "Drawing saved to my_drawing.png" << endl;
            break;

        case 27: // ESC键退出
            cout << "Exiting..." << endl;
            return 0;
        }
    }

    return 0;
}

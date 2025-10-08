#include "Matrix.h"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;


int main()
{
   const modelbase& mb1= model<double>("/home/wmx/桌面/project/GKDproject/project/mnist-fc-plus");
   const modelbase& mb2= model<float>("/home/wmx/桌面/project/GKDproject/project/mnist-fc");

   mb1.predict();
   mb2.predict();

    return 0;
}
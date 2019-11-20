#include "opencv2/opencv.hpp"
#include<iostream>
#include <opencv2\imgproc\types_c.h> 
using namespace std;
using namespace cv;
//查找最小外接矩形

void test5()
{
	Mat srcImg = imread("C:\\OpenCV\\Test1\\11111.jpg");
	imshow("src", srcImg);
	Mat result;
	result = srcImg.clone();
	Mat medImg;
	medianBlur(srcImg, medImg, 5);//中值滤波操作
	imshow("medianBlur", medImg);

	GaussianBlur(srcImg, srcImg, Size(3, 3), 0, 0); //高斯滤波
	Mat rayImg;
	cvtColor(srcImg, rayImg, CV_BGR2GRAY);//灰度图
	imshow("灰度图", rayImg);
	Mat binImg;
	threshold(rayImg, binImg, 200, 255, CV_THRESH_BINARY_INV); //INV是因为背景白色，物体黑色，需要反转一下
	imshow("二值化", binImg);

	//查找边缘
	Mat canImg;
	Canny(binImg, canImg, 50, 150);
	imshow("边缘查找", canImg);
	//绘制直线
	vector<Vec4i> lines;
	HoughLinesP(canImg, lines, 1, CV_PI / 180, 50, 50, 10);
	//fast
	vector<KeyPoint>detectKeyPoint;
	Mat keyPointImage1, keyPointImage2;
	//Ptr<FastFeatureDetector> fast = FastFeatureDetector::create();
	//FAST(srcImg, detectKeyPoint, 20);
	//drawKeypoints(result, detectKeyPoint, keyPointImage1, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	//drawKeypoints(result, detectKeyPoint, keyPointImage2, Scalar(0, 0, 255), DrawMatchesFlags::DEFAULT);
	//imshow("keyPoint image1", keyPointImage1);
	for (size_t i = 0; i < lines.size(); i++)
	{
		Vec4i l = lines[i];
		line(result, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 2);
	}
	imshow("lines", result);
	
	Mat dstImage;
	cornerHarris(rayImg, dstImage, 2, 3, 0.04, BORDER_DEFAULT);
	// 归一化与转换  
	Mat normImage;//归一化后的图  
	Mat scaledImage;//线性变换后的八位无符号整型的图  
	normalize(dstImage, normImage, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(normImage, scaledImage);
	// 将检测到的，且符合阈值条件的角点绘制出来  
	for (int j = 0; j < normImage.rows; j++)
	{
		for (int i = 0; i < normImage.cols; i++)
		{
			//Mat::at<float>(j,i)获取像素值，并与阈值比较
			if ((int)normImage.at<float>(j, i) > 500 + 80)
			{
				circle(srcImg, Point(i, j), 5, Scalar(10, 10, 255), 2, 8, 0);
				circle(scaledImage, Point(i, j), 5, Scalar(0, 10, 255), 2, 8, 0);
			}
		}
	}

	imshow("角点检测", srcImg);
	
	
	waitKey(0);
}
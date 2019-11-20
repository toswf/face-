#include "opencv2/opencv.hpp"
#include<iostream>
#include <opencv2\imgproc\types_c.h> 
using namespace std;
using namespace cv;
//������С��Ӿ���

void test5()
{
	Mat srcImg = imread("C:\\OpenCV\\Test1\\11111.jpg");
	imshow("src", srcImg);
	Mat result;
	result = srcImg.clone();
	Mat medImg;
	medianBlur(srcImg, medImg, 5);//��ֵ�˲�����
	imshow("medianBlur", medImg);

	GaussianBlur(srcImg, srcImg, Size(3, 3), 0, 0); //��˹�˲�
	Mat rayImg;
	cvtColor(srcImg, rayImg, CV_BGR2GRAY);//�Ҷ�ͼ
	imshow("�Ҷ�ͼ", rayImg);
	Mat binImg;
	threshold(rayImg, binImg, 200, 255, CV_THRESH_BINARY_INV); //INV����Ϊ������ɫ�������ɫ����Ҫ��תһ��
	imshow("��ֵ��", binImg);

	//���ұ�Ե
	Mat canImg;
	Canny(binImg, canImg, 50, 150);
	imshow("��Ե����", canImg);
	//����ֱ��
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
	// ��һ����ת��  
	Mat normImage;//��һ�����ͼ  
	Mat scaledImage;//���Ա任��İ�λ�޷������͵�ͼ  
	normalize(dstImage, normImage, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(normImage, scaledImage);
	// ����⵽�ģ��ҷ�����ֵ�����Ľǵ���Ƴ���  
	for (int j = 0; j < normImage.rows; j++)
	{
		for (int i = 0; i < normImage.cols; i++)
		{
			//Mat::at<float>(j,i)��ȡ����ֵ��������ֵ�Ƚ�
			if ((int)normImage.at<float>(j, i) > 500 + 80)
			{
				circle(srcImg, Point(i, j), 5, Scalar(10, 10, 255), 2, 8, 0);
				circle(scaledImage, Point(i, j), 5, Scalar(0, 10, 255), 2, 8, 0);
			}
		}
	}

	imshow("�ǵ���", srcImg);
	
	
	waitKey(0);
}
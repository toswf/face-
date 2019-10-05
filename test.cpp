//author:niuhao
//
#ifdef _WIN32
#include <winsock2.h>
#include <time.h>
#else
#include <sys/time.h>
#endif

#include <iostream>
#include<string.h>
#include <array>
#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include "NumCpp.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <dlib/image_io.h>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <dlib/server.h>
#include <cstdlib>
#include <dlib/compress_stream.h>
#include <dlib/base64.h>



using namespace dlib;
using namespace std;

static int FACE_POINTS[68 - 17] = { 17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67 };

static int MOUTH_POINTS[13] = { 48,49,50,51,52,53,54,55,56,57,58,59,60 };
static int RIGHT_BROW_POINTS[22 - 17] = { 17,18,19,20,21 };
static int LEFT_BROW_POINTS[27 - 22] = { 22,23,24,25,26 };
static int RIGHT_EYE_POINTS[42 - 36] = { 36,37,38,39,40,41 };
static int LEFT_EYE_POINTS[48 - 42] = { 42,43,44,45,46,47 };
static int NOSE_POINTS[35 - 27] = { 27,28,29,30,31,32,33,34 };
static int JAW_POINTS[17] = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };
static int ALIGN_POINTS[(sizeof(LEFT_BROW_POINTS) + sizeof(RIGHT_EYE_POINTS) + sizeof(LEFT_EYE_POINTS) + sizeof(RIGHT_BROW_POINTS) + sizeof(NOSE_POINTS) + sizeof(MOUTH_POINTS)) / sizeof(int)] = { 0 };
static int OVERLAY_POINTS[2][22] = { 0 };
static int OVERLAY_LEN[2] = { 22,21 };
static float COLOUR_CORRECT_BLUR_FRAC = 0.6;

frontal_face_detector detector = get_frontal_face_detector();
shape_predictor predictor;
static const std::string PRODICTOR_PATH = "";
static const float SCALE_FACTOR = 1;
static const int FEATHER_AMOUNT = 5;


cv::Mat get_face_mask(cv::Mat& matrix, nc::NdArray<double>& landmarks)
{

	cv::Mat im = cv::Mat::zeros(matrix.rows, matrix.cols, CV_64FC1);
	for (size_t i = 0; i < 2; i++)
	{
		size_t len = OVERLAY_LEN[i];
		std::vector<cv::Point> tmp;
		for (size_t j = 0; j < len; j++)
		{
			int index = OVERLAY_POINTS[i][j];
			auto x = landmarks(index, 0);
			auto y = landmarks(index, 1);
			cv::Point pt;
			pt.x = x;
			pt.y = y;
			tmp.push_back(pt);
		}
		std::vector<int> hull;
		cv::convexHull(cv::Mat(tmp), hull);
		std::vector<cv::Point> output;
		for (size_t i = 0; i < hull.size(); i++)
		{
			cv::Point pt;
			pt.x = tmp[hull[i]].x;
			pt.y = tmp[hull[i]].y;
			output.push_back(pt);
		}
		cv::fillConvexPoly(im, output, cv::Scalar(255, 255, 255));
	}

	cv::Mat all = cv::Mat::zeros(matrix.rows, matrix.cols, CV_64FC1);
	all = im + im + im;
	cv::Mat out = cv::Mat::zeros(matrix.rows, matrix.cols, CV_64FC1);
	cv::Size size(FEATHER_AMOUNT, FEATHER_AMOUNT);

	cv::GaussianBlur(all, out, size, 0);
	out = (out > 0) * 1.0;
	cv::GaussianBlur(out, out, size, 0);
	return out;

}

nc::NdArray<double> transformation_from_points(nc::NdArray<double> points1, nc::NdArray<double> points2)
{
	auto p1 = points1.astype<double>();
	auto p2 = points2.astype<double>();
	//cout << p1 << endl;
	auto c1 = nc::mean(p1, nc::Axis::ROW);
	auto c2 = nc::mean(p2, nc::Axis::ROW);
	for (size_t i = 0; i < p1.numRows(); i++)
	{
		p1.at(i, 0) -= c1.at(0, 0);
		p1.at(i, 1) -= c1.at(0, 1);
	}
	for (size_t i = 0; i < p2.numRows(); i++)
	{
		p2.at(i, 0) -= c2.at(0, 0);
		p2.at(i, 1) -= c2.at(0, 1);
	}
	//cout << p1 << endl;
	auto s1 = nc::stdev(p1);
	auto s2 = nc::stdev(p2);
	p1 /= s1.at(0, 0);
	p2 /= s2.at(0, 0);
	//cout << p1 << endl;
	nc::NdArray<double> outU;
	nc::NdArray<double> outS;
	nc::NdArray<double> outVt;
	auto m = p1.transpose().dot(p2);
	nc::linalg::svd(m, outU, outS, outVt);
	auto R = (outU.dot(outVt)).transpose();
	auto per = s2.at(0, 0) / s1.at(0, 0);
	auto pp1 = R * per;
	//cout << pp1 << endl;
	//cout << c1 << endl;
	//cout << c2 << endl;
	auto p = pp1.dot(c1.transpose());
	auto pp = p.astype<double>();
	auto pp2 = c2.transpose() - pp;
	auto hs = nc::hstack({ pp1 , pp2 });
	nc::NdArray<double> v = { 0.0,0.0,1.0 };
	auto ret = nc::vstack({ hs, v });
	return ret;

}


void read_im_and_landmarks(std::string fname, cv::Mat& output, nc::NdArray<double>& matrix)
{
	array2d<bgr_pixel> im;
	load_image(im, fname);
	if (im.nc() > 600)
	{
		double per = 600.0f / im.nc();
		resize_image(per, im);
	}
	std::vector<dlib::rectangle> rects = detector(im, 1);;
	
	if (rects.size() > 1)
	{
		throw std::exception("too mang faces");
	}
	if (rects.size() <= 0)
	{
		throw std::exception("no faces");
	}
	//std::cout << "find face!" << std::endl;

	auto parts = predictor(im, rects[0]);
	int index = 0;
	nc::NdArray<double> list(parts.num_parts(), 2);
	for (int i = 0; i < parts.num_parts(); ++i)
	{
		point p = parts.part(i);
		list[index++] = p.x();
		list[index++] = p.y();

	}
	matrix = list.copy();
	cv::Mat out = toMat(im);
	out.copyTo(output, out);


}

cv::Mat wrap_im(cv::Mat& im, nc::NdArray<double>& M, nc::Shape& dsshape)
{
	cv::Mat output_im = cv::Mat::zeros(dsshape.rows, dsshape.cols, CV_64FC1);
	auto m = M(nc::Slice(0, 2), nc::Slice(0, 3));
	cv::Mat mm = cv::Mat(m.numRows(), m.numCols(), CV_64FC1, m.data());
	cv::Size size(dsshape.cols, dsshape.rows);
	cv::warpAffine(im, output_im, mm, size, cv::WARP_INVERSE_MAP, cv::BORDER_TRANSPARENT);
	return output_im;
}

//what?

cv::Mat correct_colours(cv::Mat& im1, cv::Mat& im2, nc::NdArray<double>& landmarks1)
{
	size_t len = sizeof(LEFT_EYE_POINTS) / sizeof(int);
	nc::NdArray<double> tmp1(len, 2);
	for (size_t j = 0; j < len; j++)
	{
		int index = LEFT_EYE_POINTS[j];
		auto x = landmarks1(index, 0);
		auto y = landmarks1(index, 1);
		tmp1.put(j, 0, x);
		tmp1.put(j, 1, y);
	}
	len = sizeof(RIGHT_EYE_POINTS) / sizeof(int);
	nc::NdArray<double> tmp2(len, 2);
	for (size_t j = 0; j < len; j++)
	{
		int index = RIGHT_EYE_POINTS[j];
		auto x = landmarks1(index, 0);
		auto y = landmarks1(index, 1);
		tmp2.put(j, 0, x);
		tmp2.put(j, 1, y);
	}
	auto mean1 = nc::mean(tmp1, nc::Axis::ROW);
	auto mean2 = nc::mean(tmp2, nc::Axis::ROW);
	auto blur_amount = COLOUR_CORRECT_BLUR_FRAC * nc::norm(mean1 - mean2).at(0, 0);
	int aa = (int)blur_amount;
	if (aa % 2 == 0)
	{
		aa += 1;
	}
	cv::Mat im1_blur;
	cv::Mat im2_blur;
	cv::Size size(aa, aa);
	cv::GaussianBlur(im1, im1_blur, size, 0);
	cv::GaussianBlur(im2, im2_blur, size, 0);
	im2_blur += 128 * (im2_blur <= 1.0);
	im2.convertTo(im2, CV_64FC1);
	im1_blur.convertTo(im1_blur, CV_64FC1);
	im2_blur.convertTo(im2_blur, CV_64FC1);

	cv::Mat out;
	cv::multiply(im2, im1_blur / im2_blur, out);
	return out;
}

cv::Mat convertTo3Channels(const cv::Mat& binImg)
{
	cv::Mat three_channel = cv::Mat::zeros(binImg.rows, binImg.cols, CV_32F);
	std::vector<cv::Mat> channels;
	for (int i = 0; i < 3; i++)
	{
		channels.push_back(binImg);
	}
	cv::merge(channels, three_channel);
	return three_channel;
}
//————————————————
//版权声明：本文为CSDN博主「wx7788250」的原创文章，遵循 CC 4.0 BY - SA 版权协议，转载请附上原文出处链接及本声明。
//原文链接：https ://blog.csdn.net/wx7788250/article/details/70261615

void process(string file1, string file2, string output)
{

	cv::Mat output1;
	nc::NdArray<double> landmark1;
	read_im_and_landmarks(file1, output1, landmark1);
	cv::Mat output2;
	nc::NdArray<double> landmark2;
	read_im_and_landmarks(file2, output2, landmark2);
	size_t len = sizeof(ALIGN_POINTS) / sizeof(int);
	nc::NdArray<double> tmp1(len, 2);
	//cout << landmark1 << endl;

	for (size_t j = 0; j < len; j++)
	{
		int index = ALIGN_POINTS[j];
		auto x = landmark1(index, 0);
		auto y = landmark1(index, 1);
		tmp1.put(j, 0, x);
		tmp1.put(j, 1, y);
	}
	nc::NdArray<double> tmp2(len, 2);

	for (size_t j = 0; j < len; j++)
	{
		int index = ALIGN_POINTS[j];
		auto x = landmark2(index, 0);
		auto y = landmark2(index, 1);
		tmp2.put(j, 0, x);
		tmp2.put(j, 1, y);
	}
	//cout << tmp1 << endl;
	auto t1 = cv::Mat(tmp1.numRows(), tmp1.numCols(), CV_64FC1, tmp1.data());
	auto t2 = cv::Mat(tmp2.numRows(), tmp2.numCols(), CV_64FC1, tmp2.data());
	nc::NdArray<double> M = transformation_from_points(tmp1, tmp2);
	//cout << M << endl;
	auto mask = get_face_mask(output2, landmark2); ///CV_64FC1
	auto mask2 = get_face_mask(output1, landmark1);///CV_64FC1
	nc::Shape shape(output1.rows, output1.cols);
	auto warped_mask = wrap_im(mask, M, shape);//CV_64FC1
	cv::Mat combined_mask;
	cv::max(mask2, warped_mask, combined_mask);

	auto warped_im2 = wrap_im(output2, M, shape);//CV_64FC1
	auto warped_corrected_im2 = correct_colours(output1, warped_im2, landmark1);
	warped_corrected_im2.convertTo(warped_corrected_im2, CV_8UC3);
	cv::Mat out1, out2;
	combined_mask.convertTo(combined_mask, CV_32F);
	//cout << combined_mask/255 << endl;
	cv::Mat comb = 1.0 - combined_mask / 255;
	cv::Mat comb3 = convertTo3Channels(comb);
	output1.convertTo(output1, CV_32F);
	cv::multiply(output1, comb3, out1);
	out1.convertTo(out1, CV_8UC3);
	//cv::imshow("out1", out1);

	auto comb4 = convertTo3Channels(combined_mask / 255);
	warped_corrected_im2.convertTo(warped_corrected_im2, CV_32F);
	cv::multiply(warped_corrected_im2, comb4, out2);
	out2.convertTo(out2, CV_8UC3);
	//cv::imshow("out2", out2);
	cv::Mat output_im = out1 + out2;
	//cv::imshow("output", output_im);
	cv::imwrite(output, output_im);
}

//替换全部字符串
void replaceAll(std::string& strSource, const std::string& strOld, const std::string& strNew)
{
	int nPos = 0;
	while ((nPos = strSource.find(strOld, nPos)) != strSource.npos)
	{
		strSource.replace(nPos, strOld.length(), strNew);
		nPos += strNew.length();
	}
}



string GetCurrentTimeMsec()
{
	stringstream ss;
#ifdef _WIN32
	struct timeval tv;
	time_t clock;
	struct tm tm;
	SYSTEMTIME wtm;
	GetLocalTime(&wtm);
	tm.tm_year = wtm.wYear - 1900;
	tm.tm_mon = wtm.wMonth - 1;
	tm.tm_mday = wtm.wDay;
	tm.tm_hour = wtm.wHour;
	tm.tm_min = wtm.wMinute;
	tm.tm_sec = wtm.wSecond;
	tm.tm_isdst = -1;
	clock = mktime(&tm);
	tv.tv_sec = clock;
	tv.tv_usec = wtm.wMilliseconds * 1000;
	auto ms =  ((unsigned long long)tv.tv_sec * 1000 + (unsigned long long)tv.tv_usec / 1000);
	ss << ms << endl;
#else
	struct timeval tv;
	gettimeofday(&tv, NULL);
	auto ms =  ((unsigned long long)tv.tv_sec * 1000 + (unsigned long long)tv.tv_usec / 1000);
	ss << ms << endl;
#endif
	return ss.str();
}

//-------------------- -
//作者：先亮
//来源：CSDN
//原文：https ://blog.csdn.net/sunxianliang1/article/details/52150365 
//版权声明：本文为博主原创文章，转载请附上博文链接！

std::string save(string& file, int index)
{
	
	string tick = GetCurrentTimeMsec();
	ostringstream in;
	in << "images/" << tick << "_" << index << ".jpg" << endl;
	replaceAll(file, " ", "+");
	replaceAll(file, "data:image/jpeg;base64,", "");
	base64 base64_coder;
	ostringstream sout;
	istringstream sin;
	sin.str(file);
	base64_coder.decode(sin, sout);
	ofstream f;
	f.open(in.str(), ios::out);
	f.write(sout.str().c_str(), strlen(sout.str().c_str()));
	f.close();
	return in.str();
}

class web_server : public server_http
{
	const std::string on_request(
		const incoming_things& incoming,
		outgoing_things& outgoing
	)
	{
		ostringstream sout;
		auto file1 = incoming.queries["f1"];
		auto file2 = incoming.queries["f2"];
		auto file3 = incoming.queries["f3"];
		process(file1, file2, file3);
		sout << "success" << endl;
		return sout.str();
	}

};



int main(int argc, char* argv[])
{
	//if (argc < 3)
	{
		//printf("%s", "the args is too less");
		//return -1;
	}

	//string file1(argv[1]);
	//string file2(argv[2]);
	//string file3(argv[3]);
	//string file1 = "C:\\Project\\faceC++\\face\\face\\x64\\Release\\5.jpg";
	//string file2 = "C:\\Project\\faceC++\\face\\face\\x64\\Release\\1.jpg";
	//string file3 = "C:\\Project\\faceC++\\face\\face\\x64\\Release\\output.jpg";
	int s = sizeof(int);
	memcpy(ALIGN_POINTS, LEFT_BROW_POINTS, sizeof(LEFT_BROW_POINTS));
	memcpy(&ALIGN_POINTS[5], RIGHT_EYE_POINTS, sizeof(RIGHT_EYE_POINTS));
	memcpy(&ALIGN_POINTS[11], LEFT_EYE_POINTS, sizeof(LEFT_EYE_POINTS));
	memcpy(&ALIGN_POINTS[17], RIGHT_BROW_POINTS, sizeof(RIGHT_BROW_POINTS));
	memcpy(&ALIGN_POINTS[22], NOSE_POINTS, sizeof(NOSE_POINTS));
	memcpy(&ALIGN_POINTS[30], MOUTH_POINTS, sizeof(MOUTH_POINTS));

	memcpy(OVERLAY_POINTS[0], LEFT_EYE_POINTS, sizeof(LEFT_EYE_POINTS));
	memcpy(&OVERLAY_POINTS[0][6], RIGHT_EYE_POINTS, sizeof(RIGHT_EYE_POINTS));
	memcpy(&OVERLAY_POINTS[0][12], LEFT_BROW_POINTS, sizeof(LEFT_BROW_POINTS));
	memcpy(&OVERLAY_POINTS[0][17], RIGHT_BROW_POINTS, sizeof(RIGHT_BROW_POINTS));

	memcpy(&OVERLAY_POINTS[1][0], NOSE_POINTS, sizeof(NOSE_POINTS));
	memcpy(&OVERLAY_POINTS[1][8], MOUTH_POINTS, sizeof(MOUTH_POINTS));

	try {
		// Load face detection and pose estimation models.
		//deserialize("C:\\Project\\faceC++\\face\\face\\x64\\Release\\shape_predictor_68_face_landmarks.dat") >> predictor;
		deserialize("shape_predictor_68_face_landmarks.dat") >> predictor;
		// create an instance of our web server
		web_server our_web_server;
		// make it listen on port 5000
		our_web_server.set_listening_port(8089);
		// Tell the server to begin accepting connections.
		our_web_server.start_async();
		cout << "Press enter to end this program" << endl;
		cin.get();
		//process(file1, file2, file3);
		//cv::waitKey();

	}
	catch (error e)
	{
		printf("%s", e.what());
	}
	catch (std::exception e)
	{
		printf("%s", e.what());
	}

}

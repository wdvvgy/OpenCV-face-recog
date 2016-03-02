#ifndef _CAMERA_
#define _CAMERA_

#include "opencv.hpp"
#include <thread>

using namespace cv;
typedef int64 i64;

class Camera {

private:
	static const String const TITLE;
	static const String const CASCADE_FILE;
	static const double TICK_FREQUENCY;
	VideoCapture cap;
	CascadeClassifier path_xml;
	vector<Rect> faces;
	Rect trackingface;
	Rect face_ROI;
	Mat face_Template;
	Mat result;
	Mat original;
	Mat original_ROI;
	bool templateMatch_Run;
	bool found_Face;
	i64 templateMatch_Start;
	i64 templateMatch_Cur;
	double scale;
	int width = 320;
	Point pos;
	double templateMatch_MaxDuration = 3;

	Rect doubleRectSize(const Rect &inputRect, const Rect &frameSize) const;
	Rect biggestFace(vector<Rect> &faces) const;
	Point centerOfRect(const Rect &rect) const;
	Mat getFaceTemplate(const Mat &frame, Rect face);
	void detectFaceAllSizes(const Mat &frame);
	void detectFaceAroundRoi(const Mat &frame);
	void detectFacesTemplateMatching(const Mat &frame);

public:
	Camera();
	void Init();
	void FaceDetect();
	void setResizedWidth(const int width);
	int  resizedWidth() const;
	Rect face() const;
	Point facePosition() const;
	void setTemplateMatchingMaxDuration(const double s);
	double templateMatchingMaxDuration() const;
};

#endif
#ifndef FACE_RECOG_
#define FACE_RECOG_

#include "opencv2\core\core.hpp"
#include "opencv2\contrib\contrib.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\objdetect\objdetect.hpp"
#include "opencv2\opencv.hpp"
using namespace cv;

class FaceRecog{

private:
	vector<Mat> images;
	vector<int> labels;
	CascadeClassifier path_xml;
	Ptr<FaceRecognizer> model;

public:
	FaceRecog();
	void read_csv(const string&, const char);
	void FaceTrain();
	void FaceRecognition();
	void FaceimgSave();
};

#endif
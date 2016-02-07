#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>
#include "FaceRecognition.h"

using namespace std;

/*
Constructor
*/
FaceRecog::FaceRecog(){

	if (!path_xml.load("C:\\Users\\user\\Downloads\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml")){
		cout << " Error loading file" << endl;
		return;
	}

	model = createFisherFaceRecognizer();
}

/*
There is format file (csv.ext) about face image labeled.
read_csv Function conducts split with the label units.
and Put it on a separate vector images(path) and labels(number).
*/

void FaceRecog::read_csv(const string& filename, const char separator = ';') {
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, error_message);
	}
	string line, path, classlabel;
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classlabel);
		if (!path.empty() && !classlabel.empty()) {
			images.push_back(imread(path, 0));
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}

/*
Machines save images and labels.
*/

void FaceRecog::FaceTrain(){
	cout << "Face train start" << endl;
	model->train(images, labels);
	cout << "Face train end" << endl;
}

/*
Recognizing current user using learned data for machines.
there are 2 Mat instance. original is original frame, gray is converted image to gray color(invoked equalize hist).
we are using gray frame because original frame is only graphic interface to show we.
internally, every recognition action is conducted gray frame.
*/

void FaceRecog::FaceRecognition(){

	cout << "start recognizing..." << endl;

	int img_width = images[0].cols;
	int img_height = images[0].rows;

	string title = "My Face Recognition";

	VideoCapture cap(0);
	if (!cap.isOpened())
	{
		cout << "exit" << endl;
		return;
	}

	namedWindow(title, 1);
	long count = 0;

	while (true)
	{
		vector<Rect> faces;
		Mat frame;
		Mat graySacleFrame;
		Mat original;

		cap >> frame;

		count = count + 1;//count frames;

		if (!frame.empty()){
			original = frame.clone();
			cvtColor(original, graySacleFrame, CV_BGR2GRAY);
			equalizeHist(graySacleFrame, graySacleFrame);

			path_xml.detectMultiScale(graySacleFrame, faces, 1.1, 3, 0, cv::Size(90, 90));

			cout << faces.size() << " faces detected" << endl;
			string frameset = std::to_string(count);
			string faceset = std::to_string(faces.size());

			string Pname = "";

			for (int i = 0; i < faces.size(); i++)
			{
				Rect face_i = faces[i];
				Mat face = graySacleFrame(face_i);
				Mat face_resized;
				resize(face, face_resized, Size(img_width, img_height), 1.0, 1.0, INTER_CUBIC);

				int label = -1; double confidence = 0;
				model->predict(face_resized, label, confidence);

				cout << label << endl;

				rectangle(original, face_i, CV_RGB(0, 255, 0), 1);

				string text = "Detected";

				/*
				My label is 35.. in csv.ext file
				if you see your name in original frame, change to label number in csv file
				*/
				if (label == 35){
					Pname = "Jeongki";
				}
				else{
					Pname = "unknown";
				}

				int pos_x = std::max(face_i.tl().x - 10, 0);
				int pos_y = std::max(face_i.tl().y - 10, 0);
				putText(original, text, Point(pos_x, pos_y), FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 0), 1.0);
			}

			putText(original, "Frames: " + frameset, Point(30, 60), CV_FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 255), 1.0);
			putText(original, "Person: " + Pname, Point(30, 90), CV_FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 255), 1.0);

			imshow("My Face Recognition", original);

		}
		if (waitKey(30) >= 0) break;
	}

}

/*
write file 1 pic to camera
img file is written jpg format
*/
void FaceRecog::FaceimgSave(){
	VideoCapture cap(0);
	CascadeClassifier face_cascade;

	if (!cap.isOpened()){
		cout << "cam is can't open" << endl;
		return;
	}

	int count = 0;

	while (true){
		Mat original_frame;
		Mat gray_frame;

		cap >> original_frame;

		cvtColor(original_frame, gray_frame, CV_BGR2GRAY);
		equalizeHist(gray_frame, gray_frame);

		vector<Rect> faces;

		path_xml.detectMultiScale(gray_frame, faces, 1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_SCALE_IMAGE, Size(30, 30));

		for (int i = 0; i < faces.size(); i++){
			Point lu(faces[i].x, faces[i].y);
			Point rd(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
			Rect roi(lu, rd);
			Mat img_roi = original_frame(roi);

			imwrite("D:\\Facejpgs\\" + to_string(i) + ".jpg",img_roi);
		}

		imshow("Face Dection", original_frame);
		if (waitKey(30) >= 0) 
			break;
	}
}
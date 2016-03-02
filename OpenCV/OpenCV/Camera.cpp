#include "Camera.h"

using namespace std;

const double Camera::TICK_FREQUENCY = getTickFrequency();
const String const Camera::TITLE = "Face Recognize";
const String const Camera::CASCADE_FILE = "C:\\Users\\user\\Downloads\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt2.xml";

Camera::Camera(){
	cout << "Starting show cam.." << endl;
	Init();
}

void Camera::Init(){
	
	cap = VideoCapture(0);
	
	if (!cap.isOpened()){
		cout << "cam is can't open" << endl;
		return;
	}

	if (!path_xml.load(CASCADE_FILE)){
		cout << " Error loading file" << endl;
		return;
	}

	namedWindow(TITLE, WINDOW_AUTOSIZE);

}

Rect Camera::face() const
{
	Rect faceRect = trackingface;
	faceRect.x = (int)(faceRect.x / scale);
	faceRect.y = (int)(faceRect.y / scale);
	faceRect.width = (int)(faceRect.width / scale);
	faceRect.height = (int)(faceRect.height / scale);
	return faceRect;
}

void Camera::FaceDetect(){
	while (true){
		Mat frame;
		cap >> frame;
		scale = (double)std::min(width, frame.cols) / frame.cols;

		Size resizedFrameSize = Size((int)(scale*frame.cols), (int)(scale*frame.rows));
		Mat resizedFrame;
		resize(frame, resizedFrame, resizedFrameSize);

		if (!found_Face)
			detectFaceAllSizes(resizedFrame);
		else{
			detectFaceAroundRoi(resizedFrame);
			if (templateMatch_Run)
				detectFacesTemplateMatching(resizedFrame);
		}
		rectangle(frame, face(), Scalar(255, 0, 0));

		imshow(TITLE, frame);
		if (waitKey(25) == 27) break;
	}
}

Rect Camera::doubleRectSize(const Rect &inputRect, const Rect &frameSize) const
{
	Rect outputRect;
	// Double rect size
	outputRect.width = inputRect.width * 2;
	outputRect.height = inputRect.height * 2;

	// Center rect around original center
	outputRect.x = inputRect.x - inputRect.width / 2;
	outputRect.y = inputRect.y - inputRect.height / 2;

	// Handle edge cases
	if (outputRect.x < frameSize.x) {
		outputRect.width += outputRect.x;
		outputRect.x = frameSize.x;
	}
	if (outputRect.y < frameSize.y) {
		outputRect.height += outputRect.y;
		outputRect.y = frameSize.y;
	}

	if (outputRect.x + outputRect.width > frameSize.width) {
		outputRect.width = frameSize.width - outputRect.x;
	}
	if (outputRect.y + outputRect.height > frameSize.height) {
		outputRect.height = frameSize.height - outputRect.y;
	}

	return outputRect;
}

Point Camera::centerOfRect(const Rect &rect) const
{
	return Point(rect.x + rect.width / 2, rect.y + rect.height / 2);
}

Rect Camera::biggestFace(vector<Rect> &faces) const
{
	assert(!faces.empty());

	Rect *biggest = &faces[0];
	for (auto &face : faces) {
		if (face.area() < biggest->area())
			biggest = &face;
	}
	return *biggest;
}

/*
* Face template is small patch in the middle of detected face.
*/
Mat Camera::getFaceTemplate(const Mat &frame, Rect face)
{
	face.x += face.width / 4;
	face.y += face.height / 4;
	face.width /= 2;
	face.height /= 2;

	Mat faceTemplate = frame(face).clone();
	return faceTemplate;
}

void Camera::detectFaceAllSizes(const Mat &frame)
{
	// Minimum face size is 1/5th of screen height
	// Maximum face size is 2/3rds of screen height
	path_xml.detectMultiScale(frame, faces, 1.1, 3, 0,
		Size(frame.rows / 5, frame.rows / 5),
		Size(frame.rows * 2 / 3, frame.rows * 2 / 3));

	if (faces.empty()) return;

	found_Face = true;

	// Locate biggest face
	trackingface = biggestFace(faces);

	// Copy face template
	face_Template = getFaceTemplate(frame, trackingface);

	// Calculate roi
	face_ROI = doubleRectSize(trackingface, Rect(0, 0, frame.cols, frame.rows));

	// Update face position
	pos = centerOfRect(trackingface);
}

void Camera::detectFaceAroundRoi(const Mat &frame)
{
	// Detect faces sized +/-20% off biggest face in previous search
	path_xml.detectMultiScale(frame(face_ROI), faces, 1.1, 3, 0,
		Size(trackingface.width * 8 / 10, trackingface.height * 8 / 10),
		Size(trackingface.width * 12 / 10, trackingface.width * 12 / 10));

	if (faces.empty())
	{
		// Activate template matching if not already started and start timer
		templateMatch_Run = true;
		if (templateMatch_Start == 0)
			templateMatch_Start = getTickCount();
		return;
	}

	// Turn off template matching if running and reset timer
	templateMatch_Run = false;
	templateMatch_Cur = templateMatch_Start = 0;

	// Get detected face
	trackingface = biggestFace(faces);

	// Add roi offset to face
	trackingface.x += face_ROI.x;
	trackingface.y += face_ROI.y;

	// Get face template
	face_Template = getFaceTemplate(frame, trackingface);

	// Calculate roi
	face_ROI = doubleRectSize(trackingface, Rect(0, 0, frame.cols, frame.rows));

	// Update face position
	pos = centerOfRect(trackingface);
}

void Camera::detectFacesTemplateMatching(const Mat &frame)
{
	// Calculate duration of template matching
	templateMatch_Cur = getTickCount();
	double duration = (double)(templateMatch_Cur - templateMatch_Start) / TICK_FREQUENCY;

	// If template matching lasts for more than 2 seconds face is possibly lost
	// so disable it and redetect using cascades
	if (duration > templateMatch_MaxDuration) {
		found_Face = false;
		templateMatch_Run = false;
		templateMatch_Start = templateMatch_Cur = 0;
	}

	// Template matching with last known face 
	//cv::matchTemplate(frame(m_faceRoi), m_faceTemplate, m_matchingResult, CV_TM_CCOEFF);
	matchTemplate(frame(face_ROI), face_Template, result, CV_TM_SQDIFF_NORMED);
	normalize(result, result, 0, 1, NORM_MINMAX, -1, cv::Mat());
	double min, max;
	Point minLoc, maxLoc;
	minMaxLoc(result, &min, &max, &minLoc, &maxLoc);

	// Add roi offset to face position
	minLoc.x += face_ROI.x;
	minLoc.y += face_ROI.y;

	// Get detected face
	//m_trackedFace = cv::Rect(maxLoc.x, maxLoc.y, m_trackedFace.width, m_trackedFace.height);
	trackingface = Rect(minLoc.x, minLoc.y, face_Template.cols, face_Template.rows);
	trackingface = doubleRectSize(trackingface, Rect(0, 0, frame.cols, frame.rows));

	// Get new face template
	face_Template = getFaceTemplate(frame, trackingface);

	// Calculate face roi
	face_ROI = doubleRectSize(trackingface, Rect(0, 0, frame.cols, frame.rows));

	// Update face position
	pos = centerOfRect(trackingface);
}
#include "Controller.h"


Controller::Controller(){
	MyCam = new Camera();
}

void Controller::DoDetect(){
	MyCam->FaceDetect();
}

Controller::~Controller(){
	delete MyCam;
}
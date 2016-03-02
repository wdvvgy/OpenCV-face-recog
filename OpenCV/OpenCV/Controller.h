#ifndef _CONTROLLER_
#define _CONTROLLER_

#include "Camera.h"
#include "Face_Works.h"

class Controller{
private:
	Camera *MyCam;
public:
	Controller();
	~Controller();
	void DoDetect();
	
};

#endif
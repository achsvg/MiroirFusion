#ifndef MIROIR_WINDOW_LISTENER
#define MIROIR_WINDOW_LISTENER

#include "miroir_manager.h"

class MiroirWindowListener
{
public:
	MiroirWindowListener(){ MiroirManager::addWindowListener(this); };
	virtual void onWindowResized(int width, int height) = 0;
};

#endif
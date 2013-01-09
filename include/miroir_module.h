#ifndef MIROIR_MODULE_H
#define MIROIR_MODULE_H

#include "data_packet.h"

/** \brief A module gets an input, applies operations to it and produces an output. 
  *	Modules can be linked to each other by connecting their input and output provided the types match.
  * \author Anthony Chansavang <anthony.chansavang@gmail.com>
  */
class MiroirModule
{
public:

	MiroirModule()
	{
	}

	virtual ~MiroirModule()
	{
	}

	virtual void processData() = 0;
};

#endif
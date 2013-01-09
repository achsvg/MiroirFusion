#ifndef MIROIR_MODULE_MANAGER_H
#define MIROIR_MODULE_MANAGER_H

#include "miroir_module.h"

/** \brief Manager for the modules. 
  * \author Anthony Chansavang <anthony.chansavang@gmail.com>
  */
class MiroirModuleManager
{
private:
	static MiroirModuleManager& singleton;

	MiroirModuleManager();

public:
	virtual ~MiroirModuleManager();

	static MiroirModuleManager& getSingleton()
	{
		return singleton;
	}

	/** \brief Link two modules by their output and input.
      * \param[in] from output module.
	  * \param[in] to input module.
      */
	void plug( const MiroirModule& from, const MiroirModule& to )
	{
		// do some type checking
	}

	/** \brief feed data to the first module.
	  * \param[in] data
      */
	template < typename T > void 
	feed( const DataPacket< T >& data )
	{

	}

	/** \brief Manually feed data to a module.
      * \param[in] module
	  * \param[in] data
      */
	//void feed( const MiroirModule& module, const DataPacket< T >& data )
	//{
	//	// TODO : do type checking
	//	//module.
	//}

	/** \brief Iterate one step.
      */
	void step();

	/** \brief Start the machine.
      */
	void start();

	/** \brief Stop the machine.
      */
	void stop();
};

#endif
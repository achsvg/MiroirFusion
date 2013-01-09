#ifndef DATA_PACKET_H
#define DATA_PACKET_H

#include <boost/shared_ptr.hpp>

/** \brief Data encapsulation for the modules.
  * \author Anthony Chansavang <anthony.chansavang@gmail.com>
  */
template< typename T > class DataPacket
{
private:
	boost::shared_ptr<T> data;
public:

	DataPacket()
	{
	}

	DataPacket( T* d )
	{
		data = boost::shared_ptr<T>(d);
	}

	DataPacket( const boost::shared_ptr<T>& d ) : data(d)
	{
	}

	virtual ~DataPacket()
	{
	}

	boost::shared_ptr<T> getData()
	{
		return data;
	}

	DataPacket<T>& operator=( const DataPacket<T>& d )
	{
		data = d.data;
		return *this;
	}
};

#endif
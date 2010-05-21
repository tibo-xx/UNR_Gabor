/**
* @class voConnection
*
* @brief This is the client class for interfacing with the voServer
*
* @author CMT
*
* $Header $
*/
#ifndef VOCONNECTION_H_
#define VOCONNECTION_H_

//#include <include.h>
#include <stdio.h>
#include <string>
#include <iostream>

// Network
#include <cstdio>
#include <unistd.h>
#include <cstdlib>
#include <errno.h>
#include <cctype>
#include <cstring>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>

/// The type of connections with the vo Server available.
enum Connect_Type {
	BINARY_PUBLISH,
	BINARY_SUBSCRIBE,
	ASCII_PUBLISH,
	ASCII_SUBSCRIBE
};

/// The connection structure for use with the voConnection object.
typedef struct {
	std::string host;
	std::string port;
	std::string connect_name;
	Connect_Type type;
} Connection_Info;

class voConnection {
	public:
		voConnection(Connection_Info, int);		// Constructor
		~voConnection();						// Destructor
		bool voConnect();
		bool voDisconnect();
		bool poll();
		bool bpoll();
		bool publish(std::string);
		bool publish(char*, int, int);
		bool interpreterPublish(std::string out);
		size_t getData(char*, int);
		std::string getString();
	private:
		// Private Functions
		void bail(const char *on_what);
		bool establishConnection();
		bool getTransmitStream();
		bool getReceiveStream();
		bool sendReadRequest();
		bool sendReadRequest_Binary();
		bool sendWriteRequest();
		bool sendWriteRequest_Binary();

		// Private Variables
		struct sockaddr_in _adr_srvr;
		int _s;
		int _s2;
		FILE *_rx;
		FILE *_tx;
		Connection_Info _connInfo;
		char* _buffer;
		int _maxInputBufferSize;

};

#endif /* VOCONNECTION_H_ */

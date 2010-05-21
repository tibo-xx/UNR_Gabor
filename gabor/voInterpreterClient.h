/*
 * voInterpreterClient.h
 *
 *  Created on: Mar 20, 2009
 *      Author: thibec
 */

#ifndef VOINTERPRETERCLIENT_H_
#define VOINTERPRETERCLIENT_H_

//#include <touchpad_include.h>
#include <voConnection.h>

class voInterpreterClient {

public:
	voInterpreterClient(Connection_Info, int);
	~voInterpreterClient();
	bool voConnect();
	bool voDisconnect();
	bool poll();
	bool bpoll();
	bool publish(std::string);
	bool publish(char*, int, int);
	size_t getData(char*, int);
	std::string getString();
private:
	// Private Functions
	void bail(const char *on_what);
	bool establishConnection();
	bool getTransmitStream();
	bool getReceiveStream();
	// Private Variables
	struct sockaddr_in _adr_srvr;
	int _s;
	int _s2;
	FILE *_rx;
	FILE *_tx;
	char* _buffer;
	int _maxInputBufferSize;
	Connection_Info _connInfo;
};

#endif /* VOINTERPRETERCLIENT_H_ */

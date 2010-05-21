/*
 * voInterpreterClient.cpp
 *
 *  Created on: Mar 20, 2009
 *      Author: thibec
 */

#include "voInterpreterClient.h"

using namespace std;

// ----------------------------------------------------------------------------
//							Public Functions
// ----------------------------------------------------------------------------
voInterpreterClient::voInterpreterClient(Connection_Info connIn, int maxInputBufferSize) {
	_connInfo = connIn;

	if(maxInputBufferSize > 0) {
		_maxInputBufferSize = maxInputBufferSize;
		_buffer = new char[maxInputBufferSize];
	}

};

voInterpreterClient::~voInterpreterClient() {};

bool voInterpreterClient::voConnect() {
	bool retval = false;

	establishConnection();

	_s2 = dup(_s);
	if(getReceiveStream() && getTransmitStream()) {
		retval = true;
	}
	return retval;
}

bool voInterpreterClient::voDisconnect() {
	bool retval = true;
	fflush(_tx);
	fclose(_tx);
	shutdown(fileno(_rx), SHUT_RDWR);
	fclose(_rx);
	return retval;
}

bool voInterpreterClient::poll() {
	bool retval = false;
	fd_set readfds;
	struct timeval tv;
	tv.tv_sec = 0;
	tv.tv_usec = 0;

	FD_ZERO(&readfds);
	FD_SET(_s, &readfds);

	select(_s+1, &readfds, NULL,NULL,&tv);

	if(FD_ISSET(_s, &readfds)) {
		retval = true;
	}
	return retval;
}

bool voInterpreterClient::bpoll() {
	bool retval = true;
	fd_set readfds;

	FD_ZERO(&readfds);
	FD_SET(_s, &readfds);
	select(_s+1, &readfds, NULL,NULL,NULL);	// Wait indefinitely for input

	return retval;
}

size_t voInterpreterClient::getData(char *buffer, int Size) {
	return fread(&buffer[0],1,Size,_rx);
}

string voInterpreterClient::getString() {

	string retVal;
	ssize_t tsize;

	if(fgets (&_buffer[0], _maxInputBufferSize, _rx)) {
		// Get rid of the newLine.
		if (_buffer[strlen(_buffer)-1] == '\n')
			_buffer[strlen(_buffer)-1] = '\0';

		retVal = string(_buffer);
		memset(&_buffer[0],0,sizeof _buffer);
	}

	return retVal;

}

bool voInterpreterClient::publish(string out) {
	bool retval = true;
	fprintf(_tx, "%s\n",out.c_str());
	fflush(_tx);
	return retval;
}

bool voInterpreterClient::publish(char *buffer, int dataSize, int msgSize) {
	bool retval = true;
	fwrite(&buffer[0],dataSize,msgSize,_tx);
	fflush(_tx);
	return retval;
}

// ----------------------------------------------------------------------------
//							Private Functions
// ----------------------------------------------------------------------------
void voInterpreterClient::bail(const char *on_what) {
    fputs(strerror(errno),stderr);
    fputs(": ",stderr);
    fputs(on_what,stderr);
    fputc('\n',stderr);
    exit(1);
}

bool voInterpreterClient::establishConnection() {
	int 	len_inet;               	/* length  */
	int 	s;							/* Socket */
	int 	z;
	struct hostent *hp = NULL;
	string	hostAddr;
	const char *host;

	memset(&_adr_srvr,0,sizeof _adr_srvr);

	_adr_srvr.sin_family = AF_INET;
	_adr_srvr.sin_port = htons(atoi(_connInfo.port.c_str()));
	host = _connInfo.host.c_str();

	if ((isdigit(host[0]))) {
		_adr_srvr.sin_addr.s_addr = inet_addr(host);
		if ( _adr_srvr.sin_addr.s_addr == INADDR_NONE )
				bail("bad address (Numeric).");
	} else {
		hp = gethostbyname(_connInfo.host.c_str());
		if ( !hp )
			bail("bad address.");
		if ( hp->h_addrtype != AF_INET )
			bail("bad address (Hostname).");
		_adr_srvr.sin_addr = *(struct in_addr *)hp->h_addr_list[0];
	}

	len_inet = sizeof _adr_srvr;

	//Create a TCP/IP socket to use :
	_s = socket(PF_INET,SOCK_STREAM,0);
	if ( _s == -1 )
		bail("socket()");

	// Connect to the server:
	z = connect(_s, (const sockaddr*)&_adr_srvr, len_inet);
	if ( z == -1 )
		bail("connect(2)");

	#if DEBUG_NETWORK
		 cout << "\n---- Established Network Connection ----"  << endl;
	#endif

}

bool voInterpreterClient::getTransmitStream() {
	bool retval = true;
	_tx = fdopen(_s2,"w");

	if (!_tx) {
		close(_s2);  /* Failed */
		retval = false;
	} else {
		setbuf(_tx, NULL);
	}
	return retval;
}

bool voInterpreterClient::getReceiveStream() {
	bool retval = true;
	_rx = fdopen(_s,"r");

	if ( !_rx ) {
		close(_s);  /* Failed */
		retval = false;
	} else {
		setbuf(_rx, NULL);
	}
	return retval;
}



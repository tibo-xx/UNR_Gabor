/*
 * voConnection.cpp
 *
 *  Created on: Jan 31, 2009
 *      Author: thibec
 */
#include "voConnection.h"

using namespace std;

// ----------------------------------------------------------------------------
//							Public Functions
// ----------------------------------------------------------------------------
voConnection::voConnection(Connection_Info connInfo, int maxInputBufferSize) {
	_connInfo = connInfo;

	if(maxInputBufferSize > 0) {
		_maxInputBufferSize = maxInputBufferSize;
		_buffer = new char[maxInputBufferSize];
	}

};

voConnection::~voConnection() {};

bool voConnection::voConnect() {
	bool retval = true;

	establishConnection();

	_s2 = dup(_s);
	if(getReceiveStream() && getTransmitStream()) {
		if(_connInfo.type == BINARY_PUBLISH) {
			sendWriteRequest_Binary();
		} else if(_connInfo.type == BINARY_SUBSCRIBE) {
			sendReadRequest_Binary();
		} else if(_connInfo.type == ASCII_PUBLISH) {
			sendWriteRequest();
		} else {
			sendReadRequest();
		}
	} else {
		retval = false;
	}
	return retval;
}

bool voConnection::voDisconnect() {
	bool retval = true;
	fflush(_tx);
	fclose(_tx);
	shutdown(fileno(_rx), SHUT_RDWR);
	fclose(_rx);
	return retval;
}

bool voConnection::poll() {
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

bool voConnection::bpoll() {
	bool retval = true;
	fd_set readfds;

	FD_ZERO(&readfds);
	FD_SET(_s, &readfds);
	select(_s+1, &readfds, NULL,NULL,NULL);	// Wait indefinitely for input

	return retval;
}

size_t voConnection::getData(char *buffer, int Size) {
	return fread(&buffer[0],1,Size,_rx);
}

string voConnection::getString() {
	string retVal;
	ssize_t tsize;

	tsize = read(_s, &_buffer[0], _maxInputBufferSize);

	if(tsize > 0) {
		retVal = string(_buffer);
		cout << "tsize: " << tsize << endl;
		memset(&_buffer[0],0,tsize);
	}
	return retVal;
}

bool voConnection::publish(string out) {
	bool retval = true;

	// This is set for int but out.length is a long.
	// Some testing needs to be completed to see if this
	// will still work with a long
	fprintf(_tx, "%d %s\n",out.length()+1 ,out.c_str());
	fflush(_tx);
	return retval;
}

bool voConnection::publish(char *buffer, int dataSize, int msgSize) {
	bool retval = true;
	fprintf(_tx,"%d ", msgSize*dataSize);
	fwrite(&buffer[0],dataSize,msgSize,_tx);
	fflush(_tx);
	return retval;
}

bool voConnection::interpreterPublish(string out) {
	bool retval = true;
	fprintf(_tx, "%s\n",out.c_str());
	fflush(_tx);
	return retval;
}

// ----------------------------------------------------------------------------
//							Private Functions
// ----------------------------------------------------------------------------
void voConnection::bail(const char *on_what) {
    fputs(strerror(errno),stderr);
    fputs(": ",stderr);
    fputs(on_what,stderr);
    fputc('\n',stderr);
    exit(1);
}

bool voConnection::establishConnection() {
	int 	len_inet;               	/* length  */
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

	return true;

}

bool voConnection::getTransmitStream() {
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

bool voConnection::getReceiveStream() {
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

bool voConnection::sendReadRequest() {
	fprintf(_tx, "request\nread\nascii\n%s\n", _connInfo.connect_name.c_str());
	fflush(_tx);
	return true;
}

bool voConnection::sendReadRequest_Binary() {
	fprintf(_tx, "request\nread\nbinary\n%s\n", _connInfo.connect_name.c_str());
	fflush(_tx);
	return true;
}

bool voConnection::sendWriteRequest() {
	fprintf(_tx, "request\nwrite\nascii\n%s\n", _connInfo.connect_name.c_str());
	fflush(_tx);
	return true;
}

bool voConnection::sendWriteRequest_Binary() {
	fprintf(_tx, "request\nwrite\nbinary\n%s\n", _connInfo.connect_name.c_str());
	fflush(_tx);
	return true;
}

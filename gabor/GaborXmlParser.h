/*
* File name: GaborXmlParser.h
*/

#ifndef __GABOR_XML_PARSER_COMM_H__
#define __GABOR_XML_PARSER_COMM_H__

#include <vector>
#include <string>
#include <map>
#include "tinyxml.h"

#define DEFAULT_NAME    "gabor_1"
#define DEFAULT_THETA   0.0f
#define DEFAULT_SIGMA   9.0f
#define DEFAULT_LAMBDA  25.0f
#define DEFAULT_PSI     0.0f
#define DEFAULT_HOST	"localhost"
#define DEFAULT_PORT	20001
#define DEFAULT_JOB		"defaultjob"
#define DEFAULT_TYPE	"defaultype"
#define DEFAULT_DIV_X	2
#define DEFAULT_DIV_Y	2

// All TAGs are used in XML parsing/writing

// Filter config
#define TAG_ROOT			"ProgramConfiguration"
#define TAG_GABOR			"GaborFilter"
#define TAG_NAME			"Name"
#define TAG_THETA			"Theta"
#define TAG_SIGMA			"Sigma"
#define TAG_LAMBDA			"Lambda"
#define TAG_PSI				"Psi"

// Network Config (filter specific)
#define TAG_HOST			"Host"
#define TAG_PORT			"Port"
#define TAG_TYPE			"Type"
#define TAG_JOB				"Job"

// Image division (filter specific)
#define TAG_DIV_X			"Div_X"
#define TAG_DIV_Y			"Div_Y"

// Global config
#define TAG_GLOBAL			"GlobalConfig"
#define TAG_SEND_FREQ			"SendFreqency"
#define TAG_HEARTBEAT_FREQ		"HeartbeatFrequency"
#define TAG_NCSTOOLS			"NCSTools"
#define TAG_SERVER_THROTTLE		"ServerThrottle"
#define TAG_PORT_NCSTOOLS		"PortNCSTools"
#define TAG_HOST_NCSTOOLS		"HostNCSTools"
#define TAG_DIFF_IMAGES			"DiffImages"

using namespace std;

/**
* Gabor configuration for single filter
*/
struct GaborConfig
{
	// Filter parameters
	string name;
	float theta;
	float sigma;
	float lambda;
	float psi;
	// Network config
	string host;
	int port;
	string type;
	string job;
	// Image division
	int div_x;
	int div_y;
};

/**
* Class for complete configuration
* Includes:
*	- list of GaborConfigs
*	- global configuration
*/
class CompleteConfig
{

public:

	/**
	* Default constructor
	*/
	CompleteConfig()
	{
		send_freq = 100.0;	
		b_ncstools = false;
		b_server_throttle = false;
		host_ncstools = string("localhost");
		port_ncstools = 20001;
		heartbeat_freq = 10.0;
		b_valid = false;
		b_diff_images = true;
	}

	/**
	* Default destructor
	*/
	~CompleteConfig() {}

	/**
	* Update the b_valid boolean to reflect whether the configuration is valid.
	* @ param error Updated to reflect the status of any errors on a failed validation.
	*/
	void validate(string &error)
	{
		b_valid = false;
		// Make sure all names are unique and valid
		map<string, int> str_map;
		for(int i=0; i<filters.size(); ++i)
		{
			if( 0 != str_map[ filters[i].name ] )
			{
				// Name was already added to map	
				error = string( "duplicate name " );
				error.append( filters[i].name );
				return;
			}
			if( string( "" ) == filters[i].name )
			{
				// Invalid name
				error = string( "invalid name value in GaborFilter" );
				return;
			}
			str_map[ filters[i].name ] = 1;
		}
		// Make sure values are valid for all filter parameters
		for(int i=0; i<filters.size(); ++i)
		{
			if( filters[i].theta < 0.0 ||
				filters[i].sigma < 0.0 ||
				filters[i].lambda < 0.0 ||
				filters[i].psi < 0.0 )
			{
				error = string( "invalid configuration value in GaborFilter" );
				return;
			}
		}

		// TODO: validate network configuration for each filter
		// TODO: validate global configuration

		b_valid = true;
	}

	/**
	* Returns validity of the current configuration.
	* Assumes that validate() has been called since last configuratino change.
	*/
	bool isValid()
	{ return b_valid; }

	/**
	* Reset configuration to defaults.
	*/
	void clear()
	{
		filters.clear();
		send_freq = 100.0;	
		b_ncstools = false;
		b_server_throttle = false;
		host_ncstools = string("localhost");
		port_ncstools = 20001;
		heartbeat_freq = 10.0;
		b_valid = false;
	}

	/**
	* Returns a GaborConfig struct with default values.
	*/
	GaborConfig getDefaultGaborConfig()
	{
		GaborConfig conf;
		// Filter parameters
		conf.name = DEFAULT_NAME;
		conf.theta = DEFAULT_THETA;
		conf.sigma = DEFAULT_SIGMA;  
		conf.lambda = DEFAULT_LAMBDA;
		conf.psi = DEFAULT_PSI;		
		// Network config
		conf.host = DEFAULT_HOST;
		conf.port = DEFAULT_PORT;
		conf.job = DEFAULT_JOB;
		conf.type = DEFAULT_TYPE;
		conf.div_x = DEFAULT_DIV_X;
		conf.div_y = DEFAULT_DIV_Y;
		return conf;	
	}

	float send_freq; // Frequency (ms) of send
	vector<GaborConfig> filters; // List of filter objects
	float heartbeat_freq; // Frequency of heartbeat
	bool b_ncstools; // NCS Tools flag
	bool b_server_throttle; // Server throttle flag
	int port_ncstools; // NCS Tools port (network communication)
	string host_ncstools; // NCS Tools host (network communication)
	bool b_diff_images; // Difference images flag

private:

	bool b_valid; // Validity status of configuration

};

/**
* Encapsulates all XML parsing, file opening, saving for configurations.
*/
class GaborXmlParser
{

public:

	/**
	* Default constructor.
	*/
	GaborXmlParser(){}

	/**
	* Default destructor.
	*/
	~GaborXmlParser(){}

	/**
	* Save configuration to XML file.
	* @param filename Name of file to save.
	* @param config Configuration object to save to file.
	*/
	bool save( string filename, CompleteConfig config );

	/**
	* Load configuration from file. Returns complete configuration object.
	* @param filename Name of XML file to load.
	* @param error String to be updated with error on failed load.
	*/
	CompleteConfig loadConfig( string filename, string& error );

private:

};

#endif




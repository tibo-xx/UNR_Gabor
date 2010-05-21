/*
* File name: GaborXmlParser.cpp
*/

#include "GaborXmlParser.h"

// Loads XML file into a DOM tree and converts DOM tree into CompleteConfig object.
// Returns CompleteConfig object.
CompleteConfig GaborXmlParser::loadConfig(string filename, string& error)
{
	TiXmlDocument doc;
	CompleteConfig config;

	// Load XML file into a DOM tree
	if( !doc.LoadFile( filename ) )
	{
		error = string( "could not load xml file" );
		return config;
	}
	
	TiXmlHandle doc_handle( &doc );

	// Check if the correct root element exists
	if( !doc_handle.FirstChild( TAG_ROOT ).ToElement() )
	{
		error = string( "root element " );
		error.append( TAG_ROOT );
		error.append( " not found" );
		return config;
	}

	// Get global configuration
	TiXmlHandle global_handle( doc_handle.FirstChild( TAG_ROOT ).FirstChild( TAG_GLOBAL ).ToElement() );
	TiXmlElement* send_freq_element = global_handle.FirstChild(TAG_SEND_FREQ).ToElement();
	TiXmlElement* heartbeat_freq_element = global_handle.FirstChild(TAG_HEARTBEAT_FREQ).ToElement();
	TiXmlElement* ncstools_element = global_handle.FirstChild(TAG_NCSTOOLS).ToElement();
	TiXmlElement* server_throttle_element = global_handle.FirstChild(TAG_SERVER_THROTTLE).ToElement();
	TiXmlElement* port_ncstools_element = global_handle.FirstChild(TAG_PORT_NCSTOOLS).ToElement();
	TiXmlElement* host_ncstools_element = global_handle.FirstChild(TAG_HOST_NCSTOOLS).ToElement();
	TiXmlElement* diff_images_element = global_handle.FirstChild(TAG_DIFF_IMAGES).ToElement();
	// Make sure all global elements exist
	if( !( send_freq_element && heartbeat_freq_element && ncstools_element && server_throttle_element && port_ncstools_element && host_ncstools_element && diff_images_element ) )
	{
		// At least one global configuration parameter does not exist
		error = string( "missing global configuration element" );
		return config;
	}
	// Fill in CompleteConfig global parameters
	config.send_freq = atof( string(send_freq_element->GetText()).c_str() );
	config.heartbeat_freq = atof( string(heartbeat_freq_element->GetText()).c_str() );
	config.b_ncstools = atoi( string(ncstools_element->GetText()).c_str() );
	config.b_server_throttle = atoi( string(server_throttle_element->GetText()).c_str() );
	config.b_diff_images = atoi( string(diff_images_element->GetText()).c_str() );
	config.port_ncstools = atoi( string(port_ncstools_element->GetText()).c_str() );
	config.host_ncstools = string( host_ncstools_element->GetText() );

	// Loop through all gabor filter configurations (each an element)
	TiXmlElement* gabor_element = doc_handle.FirstChild( TAG_ROOT ).FirstChild( TAG_GABOR ).ToElement();
	for( ; gabor_element; gabor_element=gabor_element->NextSiblingElement(TAG_GABOR))
	{
		TiXmlHandle gabor_handle( gabor_element );
		// Get element for each filter parameter
		TiXmlElement* name_element = gabor_handle.FirstChild(TAG_NAME).ToElement();
		TiXmlElement* theta_element = gabor_handle.FirstChild(TAG_THETA).ToElement();
		TiXmlElement* sigma_element = gabor_handle.FirstChild(TAG_SIGMA).ToElement();
		TiXmlElement* lambda_element = gabor_handle.FirstChild(TAG_LAMBDA).ToElement();
		TiXmlElement* psi_element = gabor_handle.FirstChild(TAG_PSI).ToElement();
		// Get elements for network config of this filter
		TiXmlElement* host_element = gabor_handle.FirstChild(TAG_HOST).ToElement();
		TiXmlElement* port_element = gabor_handle.FirstChild(TAG_PORT).ToElement();
		TiXmlElement* type_element = gabor_handle.FirstChild(TAG_TYPE).ToElement();
		TiXmlElement* job_element = gabor_handle.FirstChild(TAG_JOB).ToElement();
		// Get elements for the image division
		TiXmlElement* div_x_element = gabor_handle.FirstChild(TAG_DIV_X).ToElement();
		TiXmlElement* div_y_element = gabor_handle.FirstChild(TAG_DIV_Y).ToElement();

		// Make sure all elements exist
		if( !( name_element && theta_element && sigma_element && lambda_element && psi_element && host_element && port_element && type_element && job_element && div_x_element && div_y_element) )
		{
			// At least one parameter does not exist in this gabor config
			error = string( "missing configuration element in " );
			error.append( TAG_GABOR );
			return config;
		}

		// Create GaborConfig struct with respective values
		GaborConfig gaborConfig;

		// Filter parameters
		gaborConfig.name = string(name_element->GetText());
		gaborConfig.theta = atof( string(theta_element->GetText()).c_str() );
		gaborConfig.sigma = atof( string(sigma_element->GetText()).c_str() );
		gaborConfig.lambda = atof( string(lambda_element->GetText()).c_str() );
		gaborConfig.psi = atof( string(psi_element->GetText()).c_str() );
		// Network gaborConfiguration
		gaborConfig.host = string(host_element->GetText());
		gaborConfig.type = string(type_element->GetText());
		gaborConfig.job = string(job_element->GetText());
		gaborConfig.port = atoi( port_element->GetText() );
		// Image division
		gaborConfig.div_x = atoi( div_x_element->GetText() );
		gaborConfig.div_y = atoi( div_y_element->GetText() );

		// Add this GaborConfig to the list
		config.filters.push_back(gaborConfig);
	}

	// Validate
	config.validate(error);

	return config;
}

// Save CompleteConfig object to XML file.
bool GaborXmlParser::save( string filename, CompleteConfig config )
{
	TiXmlDocument new_doc;
	TiXmlDeclaration *decl = new TiXmlDeclaration( "1.0", "", "" );
	char tmp[100];

	// Root element (all gabor configs will be a child of this element)
	TiXmlElement *root_element = new TiXmlElement( TAG_ROOT );

	// Global config element
	TiXmlElement *global_element = new TiXmlElement( TAG_GLOBAL );
	// Create elements for each global config parameter
	TiXmlElement *send_freq_element = new TiXmlElement( TAG_SEND_FREQ );
	TiXmlElement *heartbeat_freq_element = new TiXmlElement( TAG_HEARTBEAT_FREQ );
	TiXmlElement *ncstools_element = new TiXmlElement( TAG_NCSTOOLS );
	TiXmlElement *server_throttle_element = new TiXmlElement( TAG_SERVER_THROTTLE );
	TiXmlElement *port_ncstools_element = new TiXmlElement( TAG_PORT_NCSTOOLS );
	TiXmlElement *host_ncstools_element = new TiXmlElement( TAG_HOST_NCSTOOLS );
	TiXmlElement *diff_images_element = new TiXmlElement( TAG_DIFF_IMAGES );
	// Create text nodes for each global config parameter
	sprintf(tmp, "%f", config.send_freq);
	TiXmlText *send_freq_text = new TiXmlText( tmp );
	sprintf(tmp, "%f", config.heartbeat_freq);
	TiXmlText *heartbeat_freq_text = new TiXmlText( tmp );
	sprintf(tmp, "%d", config.b_ncstools);
	TiXmlText *ncstools_text = new TiXmlText( tmp );
	sprintf(tmp, "%d", config.b_server_throttle);
	TiXmlText *server_throttle_text = new TiXmlText( tmp );
	sprintf(tmp, "%d", config.b_diff_images);
	TiXmlText *diff_images_text = new TiXmlText( tmp );
	sprintf(tmp, "%d", config.port_ncstools);
	TiXmlText *port_ncstools_text = new TiXmlText( tmp );
	TiXmlText *host_ncstools_text = new TiXmlText( config.host_ncstools.c_str() );
	// Link each global text node with respective element
	send_freq_element->LinkEndChild( send_freq_text );
	heartbeat_freq_element->LinkEndChild( heartbeat_freq_text );
	ncstools_element->LinkEndChild( ncstools_text );
	server_throttle_element->LinkEndChild( server_throttle_text );
	diff_images_element->LinkEndChild( diff_images_text );
	port_ncstools_element->LinkEndChild( port_ncstools_text );
	host_ncstools_element->LinkEndChild( host_ncstools_text );
	// Link each global element to global_element
	global_element->LinkEndChild( send_freq_element );
	global_element->LinkEndChild( heartbeat_freq_element );
	global_element->LinkEndChild( ncstools_element );
	global_element->LinkEndChild( server_throttle_element );
	global_element->LinkEndChild( port_ncstools_element );
	global_element->LinkEndChild( host_ncstools_element );
	global_element->LinkEndChild( diff_images_element );
	// Link the global_element to the root_element
	root_element->LinkEndChild( global_element );

	// Build tree branches for each gabor filter configuration
	for(int i=0; i<config.filters.size(); ++i)
	{
		TiXmlElement *gabor_config_element = new TiXmlElement( TAG_GABOR );
		// Create element node for each parameter
		TiXmlElement *name_element = new TiXmlElement( TAG_NAME );
		TiXmlElement *theta_element = new TiXmlElement( TAG_THETA );
		TiXmlElement *sigma_element = new TiXmlElement( TAG_SIGMA );
		TiXmlElement *lambda_element = new TiXmlElement( TAG_LAMBDA );
		TiXmlElement *psi_element = new TiXmlElement( TAG_PSI );
		// Create element nodes for each network config parameter
		TiXmlElement *host_element = new TiXmlElement( TAG_HOST );
		TiXmlElement *port_element = new TiXmlElement( TAG_PORT );
		TiXmlElement *type_element = new TiXmlElement( TAG_TYPE );
		TiXmlElement *job_element = new TiXmlElement( TAG_JOB );
		TiXmlElement *div_x_element = new TiXmlElement( TAG_DIV_X );
		TiXmlElement *div_y_element = new TiXmlElement( TAG_DIV_Y );
		// Create text node for each parameter
		TiXmlText *name_text = new TiXmlText(config.filters[i].name.c_str());
		sprintf(tmp, "%f", config.filters[i].theta);
		TiXmlText *theta_text = new TiXmlText( tmp );
		sprintf(tmp, "%f", config.filters[i].sigma);
		TiXmlText *sigma_text = new TiXmlText( tmp );
		sprintf(tmp, "%f", config.filters[i].lambda);
		TiXmlText *lambda_text = new TiXmlText( tmp );
		sprintf(tmp, "%f", config.filters[i].psi);
		TiXmlText *psi_text = new TiXmlText( tmp );
		// Create text node for each network config parameter
		TiXmlText *host_text = new TiXmlText(config.filters[i].host.c_str());
		TiXmlText *type_text = new TiXmlText(config.filters[i].type.c_str());
		TiXmlText *job_text = new TiXmlText(config.filters[i].job.c_str());
		sprintf( tmp, "%d", config.filters[i].port );
		TiXmlText *port_text = new TiXmlText( tmp );
		sprintf( tmp, "%d", config.filters[i].div_x );
		TiXmlText *div_x_text = new TiXmlText( tmp );
		sprintf( tmp, "%d", config.filters[i].div_y );
		TiXmlText *div_y_text = new TiXmlText( tmp );
		// Link up each parameter element node with respective text node
		name_element->LinkEndChild( name_text );
		theta_element->LinkEndChild( theta_text );
		sigma_element->LinkEndChild( sigma_text );
		lambda_element->LinkEndChild( lambda_text );
		psi_element->LinkEndChild( psi_text );
		// Link up each network parameter with respective text
		host_element->LinkEndChild( host_text );
		port_element->LinkEndChild( port_text );
		type_element->LinkEndChild( type_text );
		job_element->LinkEndChild( job_text );
		div_x_element->LinkEndChild( div_x_text );
		div_y_element->LinkEndChild( div_y_text );
		// Link current GaborConfig element with filter parameter elements
		gabor_config_element->LinkEndChild( name_element );
		gabor_config_element->LinkEndChild( theta_element );
		gabor_config_element->LinkEndChild( sigma_element );
		gabor_config_element->LinkEndChild( lambda_element );
		gabor_config_element->LinkEndChild( psi_element );
		// Link current GaborConfig element with network elements
		gabor_config_element->LinkEndChild( host_element );
		gabor_config_element->LinkEndChild( port_element );
		gabor_config_element->LinkEndChild( type_element );
		gabor_config_element->LinkEndChild( job_element );
		gabor_config_element->LinkEndChild( div_x_element );
		gabor_config_element->LinkEndChild( div_y_element );

		// Link complete GaborConfig node to root element node
		root_element->LinkEndChild( gabor_config_element );
	}
	new_doc.LinkEndChild( decl );
	new_doc.LinkEndChild( root_element );
	return new_doc.SaveFile( filename.c_str() );
}


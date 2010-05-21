/*
 *  MainWindowImp.cpp
 */

#include <QtGui>
#include <QHBoxLayout>
#include <QDebug>
#include "MainWindowImp.h"
#include "GaborNameDialogImp.h"
#include "tinyxml.h"

#include <stdio.h>
#include <string.h>

using namespace std;

MainWindowImp::MainWindowImp(QString video_source, QWidget *parent) : QMainWindow( parent )
{
	setupUi(this);

	_gl_context = new GLContext(video_source, this);
	setupConnections();
	
	QHBoxLayout *preview_layout = new QHBoxLayout();
	preview_layout->addWidget( _gl_context );
	frame_preview->setLayout( preview_layout );

	// Set initial parameter values
	_slider_theta_val = DEFAULT_THETA;
	_dsb_theta_val = DEFAULT_THETA;
	_slider_sigma_val = DEFAULT_SIGMA;
	_dsb_sigma_val = DEFAULT_SIGMA;
	_slider_lambda_val = DEFAULT_LAMBDA;
	_dsb_lambda_val = DEFAULT_LAMBDA;
	_slider_psi_val = DEFAULT_PSI;
	_dsb_psi_val = DEFAULT_PSI;

	newSetup();
}

MainWindowImp::~MainWindowImp()
{
}

void MainWindowImp::closeEvent(QCloseEvent *event)
{
	_gl_context->cleanup();
	event->accept();
}

void MainWindowImp::setupConnections()
{
	// Menu items
	connect( actionQuit, SIGNAL( activated() ), this, SLOT( close() ) );
	connect( actionSave, SIGNAL( activated() ), this, SLOT( save() ) );
	connect( actionOpen, SIGNAL( activated() ), this, SLOT( open() ) );
	connect( actionNew, SIGNAL( activated() ), this, SLOT( newSetup() ) );

	// Filter Parameters (double spin boxes)
	connect(dsb_theta, SIGNAL(valueChanged(double)), this, SLOT(dsbThetaChanged()));
	connect(dsb_sigma, SIGNAL(valueChanged(double)), this, SLOT(dsbSigmaChanged()));
	connect(dsb_lambda, SIGNAL(valueChanged(double)), this, SLOT(dsbLambdaChanged()));
	connect(dsb_psi, SIGNAL(valueChanged(double)), this, SLOT(dsbPsiChanged()));

	// Filter Parameters (sliders)
	connect(slider_theta, SIGNAL(valueChanged(int)), this, SLOT(sliderThetaChanged()));
	connect(slider_sigma, SIGNAL(valueChanged(int)), this, SLOT(sliderSigmaChanged()));
	connect(slider_lambda, SIGNAL(valueChanged(int)), this, SLOT(sliderLambdaChanged()));
	connect(slider_psi, SIGNAL(valueChanged(int)), this, SLOT(sliderPsiChanged()));

	// Filter opacity
	connect(slider_filter_opacity, SIGNAL( valueChanged(int)), this, SLOT(updateFilterOpacity(int)));

	// FPS display
	connect(_gl_context, SIGNAL(newFPS(float)), this, SLOT(updateFPSDisplay(float)));

	// Check boxes
	connect(cb_difference_images, SIGNAL(stateChanged(int)), this, SLOT(diffImagesChanged(int)));

	// Gabor configuration list
	connect(btn_add, SIGNAL(clicked()), this, SLOT(createNewConfiguration()));
	connect(btn_rename, SIGNAL(clicked()), this, SLOT(renameConfiguration()));
	connect(btn_remove, SIGNAL(clicked()), this, SLOT(removeConfiguration()));
	connect(btn_duplicate, SIGNAL(clicked()), this, SLOT(duplicateConfiguration()));
	connect(btn_move_up, SIGNAL(clicked()), this, SLOT(moveConfigurationUp()));
	connect(btn_move_down, SIGNAL(clicked()), this, SLOT(moveConfigurationDown()));
	connect(lw_configurations, SIGNAL(currentRowChanged(int)), this, SLOT(updateConfigurationSelection(int)));

	// Image division
	connect( sb_div_x, SIGNAL( valueChanged(int) ), this, SLOT( divXChanged() ) );
	connect( sb_div_y, SIGNAL( valueChanged(int) ), this, SLOT( divYChanged() ) );
	
	// Global config
	connect( dsb_send_freq, SIGNAL( valueChanged(double) ), this, SLOT( sendFreqChanged() ) );
	connect( dsb_heartbeat_freq, SIGNAL( valueChanged(double) ), this, SLOT( heartbeatFreqChanged() ) );
	connect( gcb_ncstools, SIGNAL( toggled(bool) ), this, SLOT( NCSToolsChanged() ) );
	connect( gcb_server_throttle, SIGNAL( toggled(bool) ), this, SLOT( serverThrottleChanged() ) );
	connect( sb_port_ncstools, SIGNAL( valueChanged(int) ), this, SLOT( portNCSToolsChanged() ) );
	connect( le_host_ncstools, SIGNAL( editingFinished() ), this, SLOT( hostNCSToolsChanged() ) );
	
	// Communication (filter specific)
	connect( le_host, SIGNAL( editingFinished() ), this, SLOT( hostChanged() ) );
	connect( le_type, SIGNAL( editingFinished() ), this, SLOT( typeChanged() ) );
	connect( le_job, SIGNAL( editingFinished() ), this, SLOT( jobChanged() ) );
	connect( sb_port, SIGNAL( valueChanged(int) ), this, SLOT( portChanged() ) );
}

void MainWindowImp::diffImagesChanged(int b)
{
	_config.b_diff_images = b;	

	// Updat the gl context to diff/not diff images accordingly
	_gl_context->setDifferenceImages(b);
}

void MainWindowImp::divXChanged()
{
	_config.filters[_curr_config_index].div_x = sb_div_x->value();
}
void MainWindowImp::divYChanged()
{
	_config.filters[_curr_config_index].div_y = sb_div_y->value();
}
void MainWindowImp::sendFreqChanged()
{
	_config.send_freq = dsb_send_freq->value();
}
void MainWindowImp::heartbeatFreqChanged()
{
	_config.heartbeat_freq = dsb_heartbeat_freq->value();
}
void MainWindowImp::NCSToolsChanged()
{
	_config.b_ncstools = gcb_ncstools->isChecked();
}
void MainWindowImp::serverThrottleChanged()
{
	_config.b_server_throttle = gcb_server_throttle->isChecked();
}
void MainWindowImp::portNCSToolsChanged()
{
	_config.port_ncstools = sb_port_ncstools->value();
}
void MainWindowImp::hostNCSToolsChanged()
{
	_config.host_ncstools = le_host_ncstools->text().toStdString();
}
void MainWindowImp::hostChanged()
{
	_config.filters[_curr_config_index].host = le_host->text().toStdString();
}
void MainWindowImp::typeChanged()
{
	_config.filters[_curr_config_index].type = le_type->text().toStdString();
}
void MainWindowImp::jobChanged()
{
	_config.filters[_curr_config_index].job = le_job->text().toStdString();
}
void MainWindowImp::portChanged()
{
	_config.filters[_curr_config_index].port = sb_port->value();
}

void MainWindowImp::updateFPSDisplay(float fps)
{
	lbl_fps_numbers->setText( QString::number(fps) );
}

void MainWindowImp::updateFilterOpacity(int new_opacity)
{
	float f_opacity = new_opacity / 100.0f;
	_gl_context->setFilterOpacity( f_opacity );
}

void MainWindowImp::sliderThetaChanged()
{
	_slider_theta_val = (float)slider_theta->value();
	//_slider_theta_val = (float)slider_theta->value() / 100.0f;
	// Check if the received signal was from the spinbox's change
	if( _dsb_theta_val != _slider_theta_val )
	{
		dsb_theta->setValue( _slider_theta_val );
		_config.filters[_curr_config_index].theta = _slider_theta_val * M_PI / 180.0;
		filterParameterChanged();
	}
}
void MainWindowImp::dsbThetaChanged()
{
	_dsb_theta_val = dsb_theta->value();
	// Check if the received signal was from the slider's change
	if( _dsb_theta_val != _slider_theta_val )
	{
		slider_theta->setValue( (int)(_dsb_theta_val) );
		_config.filters[_curr_config_index].theta = _dsb_theta_val * M_PI / 180.0;
		filterParameterChanged();
	}
}
void MainWindowImp::sliderSigmaChanged()
{
	_slider_sigma_val = (float)slider_sigma->value();
	// Check if the received signal was from the spinbox's change
	if( _dsb_sigma_val != _slider_sigma_val )
	{
		dsb_sigma->setValue( _slider_sigma_val );
		_config.filters[_curr_config_index].sigma = _slider_sigma_val;
		filterParameterChanged();
	}
}
void MainWindowImp::dsbSigmaChanged()
{
	_dsb_sigma_val = dsb_sigma->value();
	// Check if the received signal was from the slider's change
	if( _dsb_sigma_val != _slider_sigma_val )
	{
		slider_sigma->setValue( (int)(_dsb_sigma_val) );
		_config.filters[_curr_config_index].sigma = _dsb_sigma_val;
		filterParameterChanged();
	}
}
void MainWindowImp::sliderLambdaChanged()
{
	_slider_lambda_val = (float)slider_lambda->value();
	// Check if the received signal was from the spinbox's change
	if( _dsb_lambda_val != _slider_lambda_val )
	{
		dsb_lambda->setValue( _slider_lambda_val );
		_config.filters[_curr_config_index].lambda = _slider_lambda_val;
		filterParameterChanged();
	}
}
void MainWindowImp::dsbLambdaChanged()
{
	_dsb_lambda_val = dsb_lambda->value();
	// Check if the received signal was from the slider's change
	if( _dsb_lambda_val != _slider_lambda_val )
	{
		slider_lambda->setValue( (int)(_dsb_lambda_val) );
		_config.filters[_curr_config_index].lambda = _dsb_lambda_val;
		filterParameterChanged();
	}
}
void MainWindowImp::sliderPsiChanged()
{
	_slider_psi_val = (float)slider_psi->value();
	// Check if the received signal was from the spinbox's change
	if( _dsb_psi_val != _slider_psi_val )
	{
		dsb_psi->setValue( _slider_psi_val );
		_config.filters[_curr_config_index].psi = _slider_psi_val * M_PI / 180.0;
		filterParameterChanged();
	}
}
void MainWindowImp::dsbPsiChanged()
{
	_dsb_psi_val = dsb_psi->value();
	// Check if the received signal was from the slider's change
	if( _dsb_psi_val != _slider_psi_val )
	{
		slider_psi->setValue( (int)(_dsb_psi_val) );
		_config.filters[_curr_config_index].psi = _dsb_psi_val * M_PI / 180.0;
		filterParameterChanged();
	}
}

void MainWindowImp::filterParameterChanged()
{
	float theta = dsb_theta->value() * M_PI / 180.0;
	float sigma = dsb_sigma->value();
	float lambda = dsb_lambda->value();
	float psi = dsb_psi->value() * M_PI / 180.0;

	_gl_context->setNewFilterParameters(theta, sigma, lambda, psi);
}

void MainWindowImp::updateConfigurationSelection(int row)
{
	int config_ct = _config.filters.size();
	if( row < 0 || row > config_ct - 1 )
	{
		return;
	}
	_curr_config_index = row;

	// Setting the "dsp" (double spin box) will create
	// a new gabor and update sliders because of connections
	// TODO!!! Fix: A new filter is created 4 times (once for each
	// parameter value because of connections)... Doesn't hurt,
	// but can be optimised.
	dsb_theta->setValue( _config.filters[row].theta * 180.0 / M_PI );
	dsb_sigma->setValue( _config.filters[row].sigma );
	dsb_lambda->setValue( _config.filters[row].lambda );
	dsb_psi->setValue( _config.filters[row].psi * 180.0 / M_PI );
	
	sb_div_x->setValue( _config.filters[row].div_x );
	sb_div_y->setValue( _config.filters[row].div_y );
	sb_port->setValue( _config.filters[row].port );
	le_host->setText( QString(_config.filters[row].host.c_str() ) );
	le_type->setText( QString(_config.filters[row].type.c_str() ) );
	le_job->setText( QString(_config.filters[row].job.c_str() ) );

	// Enable/disable buttons as needed
	updateButtonStates();
}

void MainWindowImp::updateButtonStates()
{
	int config_ct = _config.filters.size();
	int row = lw_configurations->currentRow();

	if( config_ct == 1 )
	{
		btn_remove->setEnabled(false);
		btn_move_up->setEnabled(false);
		btn_move_down->setEnabled(false);
	}
	else if( row == 0 )
	{
		btn_remove->setEnabled(true);
		btn_move_up->setEnabled(false);
		btn_move_down->setEnabled(true);
	}
	else if( row == config_ct - 1 )
	{
		btn_remove->setEnabled(true);
		btn_move_up->setEnabled(true);
		btn_move_down->setEnabled(false);
	}
	else
	{
		btn_remove->setEnabled(true);
		btn_move_up->setEnabled(true);
		btn_move_down->setEnabled(true);
	}
}

void MainWindowImp::createNewConfiguration()
{
	// Create and show gabor name dialog
	int curr_row = lw_configurations->currentRow();
	GaborNameDialogImp *name_dialog = new GaborNameDialogImp(this);	
	QString next_name("gabor_");
	next_name.append( QString::number(_config_name_ct+1) );
	name_dialog->le_name->setText(next_name);
	int result = name_dialog->exec();
	if( result == QDialog::Accepted )
	{
		// Get the name of the new gabor
		QString name( name_dialog->le_name->text() );
		// Check if the name already exists
		for(unsigned int i=0; i<_config.filters.size(); ++i)
		{
			if( _config.filters[i].name == name.toStdString() )
			{
				QMessageBox msg_box(this);
				msg_box.setWindowTitle("Invalid Name");
				msg_box.setText( "Name is already in use." );
				msg_box.setStandardButtons( QMessageBox::Ok );
				msg_box.setIcon( QMessageBox::Warning );
				msg_box.exec();
				return;
			}
		}
		GaborConfig new_config = _config.getDefaultGaborConfig();
		new_config.name = name.toStdString();

		// Add the new config to the list and the list widget
		vector<GaborConfig>::iterator iter = _config.filters.begin();
		iter += _curr_config_index+1;
		_config.filters.insert( iter, new_config );
		lw_configurations->insertItem( _curr_config_index+1, name );
		_curr_config_index++;
		_config_name_ct++;
		// Select the new config in the list widget
		lw_configurations->setCurrentRow( _curr_config_index );
		// Enable/disable buttons as needed
		updateButtonStates();
	}
}

void MainWindowImp::renameConfiguration()
{
	// Create and show gabor name dialog
	int curr_row = lw_configurations->currentRow();
	GaborNameDialogImp *name_dialog = new GaborNameDialogImp(this);	
	name_dialog->setWindowTitle("Rename");
	name_dialog->le_name->setText( lw_configurations->currentItem()->text() );
	int result = name_dialog->exec();
	if( result == QDialog::Accepted )
	{
		// Get the name of the new gabor
		QString name( name_dialog->le_name->text() );
		// Check if the name already exists
		for(int i=0; i<_config.filters.size(); ++i)
		{
			if( _config.filters[i].name == name.toStdString() )
			{
				if( i == curr_row )
				{
					// Don't do anything if the "Ok" button was
					// pressed and the name was not changed
					return;
				}
				QMessageBox msg_box(this);
				msg_box.setWindowTitle("Invalid Name");
				msg_box.setText( "Name is already in use." );
				msg_box.setStandardButtons( QMessageBox::Ok );
				msg_box.setIcon( QMessageBox::Warning );
				msg_box.exec();
				return;
			}
		}
		// Change name in list and list widget
		lw_configurations->currentItem()->setText(name);
		_config.filters[curr_row].name = name.toStdString();
	}
}

void MainWindowImp::removeConfiguration()
{
	// Make sure there is a selection and there is more than one configuration
	int curr_row = lw_configurations->currentRow();
	int config_ct = lw_configurations->count();
	if( config_ct < 2 || curr_row < 0 ||
		_config.filters.size() < 2 )
	{
		return;
	}	

	vector<GaborConfig>::iterator iter = _config.filters.begin();
	iter += curr_row;
	_config.filters.erase( iter );
//	_config.filters.removeAt( curr_row );
	// *Note managed by Qt: needs to be deleted manually
	delete lw_configurations->takeItem(curr_row);
	// Update the selection
	if( curr_row == config_ct-1 )
	{
		// Removed item was at end... Select new end.	
		lw_configurations->setCurrentRow( curr_row-1 );
	}
	else
	{
		// Removed item was not at end... Keep the same index.
		updateConfigurationSelection(curr_row);
	}

	// Enable/disable buttons as needed
	updateButtonStates();
}

void MainWindowImp::duplicateConfiguration()
{
	int curr_row = lw_configurations->currentRow();
	// Create and show gabor name dialog
	GaborNameDialogImp *name_dialog = new GaborNameDialogImp(this);	
	name_dialog->setWindowTitle("Duplicate (Copy)");
	name_dialog->le_name->setText( lw_configurations->currentItem()->text() );
	int result = name_dialog->exec();
	if( result == QDialog::Accepted )
	{
		// Get the name of the new gabor
		QString name( name_dialog->le_name->text() );
		// Check if the name already exists
		for(int i=0; i<_config.filters.size(); ++i)
		{
			if( _config.filters[i].name == name.toStdString() )
			{
				QMessageBox msg_box(this);
				msg_box.setWindowTitle("Invalid Name");
				msg_box.setText( "Name is already in use." );
				msg_box.setStandardButtons( QMessageBox::Ok );
				msg_box.setIcon( QMessageBox::Warning );
				msg_box.exec();
				return;
			}
		}

		GaborConfig new_config;
		new_config = _config.filters[curr_row];
		new_config.name = name.toStdString();

		// Add the new config to the list and the list widget
		vector<GaborConfig>::iterator iter = _config.filters.begin();
		iter += _curr_config_index+1;
		_config.filters.insert( iter, new_config );
		lw_configurations->insertItem( _curr_config_index+1, name );
		_curr_config_index++;
		_config_name_ct++;
		// Select the new config in the list widget
		lw_configurations->setCurrentRow( _curr_config_index );
		// Enable/disable buttons as needed
		updateButtonStates();
	}
}

void MainWindowImp::moveConfigurationUp()
{
	// Make sure the top item is not selected (something's wrong)
	int curr_row = lw_configurations->currentRow();	
	if( curr_row == 0 )
	{
		btn_move_up->setEnabled(false);
		return;
	}
	
	// Move the selected item up
	GaborConfig tmp = _config.filters[curr_row];
	_config.filters[curr_row] = _config.filters[curr_row-1];
	_config.filters[curr_row-1] = tmp;
	QString curr_text = lw_configurations->currentItem()->text();
	QString up_text = lw_configurations->item(curr_row-1)->text();
	lw_configurations->currentItem()->setText(up_text);
	lw_configurations->item(curr_row-1)->setText(curr_text);
	lw_configurations->setCurrentRow(curr_row-1);
	_curr_config_index = curr_row-1;

	// Enable/disable buttons as needed
	updateButtonStates();
}

void MainWindowImp::moveConfigurationDown()
{
	// Make sure the bottom item is not selected (something's wrong)
	int curr_row = lw_configurations->currentRow();	
	int last_row = lw_configurations->count()-1;
	if( curr_row == last_row )
	{
		btn_move_down->setEnabled(false);
		return;
	}
	
	// Move the selected item down
	GaborConfig tmp = _config.filters[curr_row];
	_config.filters[curr_row] = _config.filters[curr_row+1];
	_config.filters[curr_row+1] = tmp;
	QString curr_text = lw_configurations->currentItem()->text();
	QString down_text = lw_configurations->item(curr_row+1)->text();
	lw_configurations->currentItem()->setText(down_text);
	lw_configurations->item(curr_row+1)->setText(curr_text);
	lw_configurations->setCurrentRow(curr_row+1);
	_curr_config_index = curr_row+1;

	// Enable/disable buttons as needed
	updateButtonStates();
}

void MainWindowImp::updateGlobalConfigWidgets()
{
	dsb_send_freq->setValue( _config.send_freq );
	dsb_heartbeat_freq->setValue( _config.heartbeat_freq );
	gcb_ncstools->setChecked( _config.b_ncstools );
	gcb_server_throttle->setChecked( _config.b_server_throttle );
	le_host_ncstools->setText( QString(_config.host_ncstools.c_str() ) );
	sb_port_ncstools->setValue( _config.port_ncstools );
}

void MainWindowImp::newSetup()
{
	// Reset to default configuration
	_config.clear();
	// Update the global configuration widgets
	updateGlobalConfigWidgets();
	lw_configurations->clear();
	// Add single default configuration
	_config.filters.push_back( _config.getDefaultGaborConfig() );
	lw_configurations->addItem( QString( _config.filters[0].name.c_str() ) );
	// Select default configuration
	lw_configurations->setCurrentRow( 0 );
	// Update spin boxes / sliders to reflect default values
	updateConfigurationSelection( 0 );	
	// Reset count for automatic configuration naming
	_config_name_ct = 1;
}

void MainWindowImp::save()
{
	QString filename = QFileDialog::getSaveFileName(this, "Save Configuration", "", "XML Files (*.xml)");
	if( filename == QString("") )
		return;

	if( !_gabor_xml_parser.save( filename.toStdString(), _config ) )
	{
   		QMessageBox msg_box(this);
   		msg_box.setWindowTitle("Error Saving XML File");
   		msg_box.setText( "Unknown error." );
   		msg_box.setStandardButtons( QMessageBox::Ok );
   		msg_box.setIcon( QMessageBox::Warning );
		msg_box.exec();
		return;
	}
}

void MainWindowImp::open()
{
	QString filename = QFileDialog::getOpenFileName(this, "Open Configuration File", "", "XML Files (*.xml)");
	if( filename == QString("") )
		return;

	// Load the file and convert to CompleteConfig class
	CompleteConfig tmp_config;
	string error;
	tmp_config = _gabor_xml_parser.loadConfig( filename.toStdString(), error );
	if( !tmp_config.isValid() )
	{
   		QMessageBox msg_box(this);
   		msg_box.setWindowTitle("Error Loading XML File");
   		msg_box.setText( QString( error.c_str() ) );
   		msg_box.setStandardButtons( QMessageBox::Ok );
   		msg_box.setIcon( QMessageBox::Warning );
		msg_box.exec();
		return;
	}

	// Save the configuration
	_config = tmp_config;
	
	// Update the list widget to hold the new names
	lw_configurations->clear();
	for(int i=0; i<_config.filters.size(); i++)
	{
		lw_configurations->addItem( QString(_config.filters[i].name.c_str()) );
	}

	// Update the global config widgets
	updateGlobalConfigWidgets();

	// Make sure the first item is selected and the filter parameters are updated
	lw_configurations->setCurrentRow(0);
	updateConfigurationSelection(0);
}



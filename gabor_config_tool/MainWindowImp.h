/*
 * File name: MainWindowImp.h
 */

#ifndef __MAIN_WINDOW_H__
#define __MAIN_WINDOW_H__

#include <QList>
#include <QString>
#include "ui_MainWindow.h"
#include "GLContext.h"
#include "GaborXmlParser.h"

/**
* Main widget to contain all other widgets and functionality. 
*/
class MainWindowImp : public QMainWindow, public Ui::MainWindow
{
	Q_OBJECT
public:
	MainWindowImp(QString video_source, QWidget* parent = 0);
	~MainWindowImp();
	void closeEvent(QCloseEvent *event);
private slots:
	void diffImagesChanged(int);
	void filterParameterChanged();
	void updateFilterOpacity(int);
	void updateFPSDisplay(float);
	void sliderThetaChanged();
	void dsbThetaChanged();
	void sliderSigmaChanged();
	void dsbSigmaChanged();
	void sliderLambdaChanged();
	void dsbLambdaChanged();
	void sliderPsiChanged();
	void dsbPsiChanged();
	void createNewConfiguration();
	void renameConfiguration();
	void removeConfiguration();
	void updateConfigurationSelection(int);
	void updateButtonStates();
	void duplicateConfiguration();
	void moveConfigurationUp();
	void moveConfigurationDown();
	void newSetup();
	void save();
	void open();

	void divXChanged();
	void divYChanged();
	void sendFreqChanged();
	void heartbeatFreqChanged();
	void NCSToolsChanged();
	void serverThrottleChanged();
	void portNCSToolsChanged();
	void hostNCSToolsChanged();
	void hostChanged();
	void typeChanged();
	void jobChanged();
	void portChanged();

private:
	void setupConnections();
	void updateGlobalConfigWidgets();
	GLContext *_gl_context;	
	float _slider_theta_val;
	float _dsb_theta_val;
	float _slider_sigma_val;
	float _dsb_sigma_val;
	float _slider_lambda_val;
	float _dsb_lambda_val;
	float _slider_psi_val;
	float _dsb_psi_val;
	CompleteConfig _config;
	int _curr_config_index;
	int _config_name_ct;
	GaborXmlParser _gabor_xml_parser;
};

#endif

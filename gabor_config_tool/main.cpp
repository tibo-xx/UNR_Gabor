/*
 *  main.cpp
 */

#include <QApplication>
#include <QString>
#include <QFile>
#include <QDebug>

#include "MainWindowImp.h"

using namespace std;

void printUsage()
{
	qDebug() << "Usage: gabor <video source>";
	qDebug() << "   Ex: gabor /dev/video0";
}

int main(int argc, char *argv[])
{
	if( argc != 2 )
	{
		printUsage();
		return 0;
	}	

	QString str = QString( argv[1] );
	QFile file( str );
	if( !file.exists() )
	{
		qDebug() << "Video source " << str << " does not exist.";
		return 0;
	}

	// Main window
	QApplication app(argc, argv);
	MainWindowImp main_window( str );
	main_window.show();	
	return app.exec();
}


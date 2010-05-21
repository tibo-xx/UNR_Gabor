/********************************************************************************
** Form generated from reading ui file 'VideoSourceDialog.ui'
**
** Created: Tue Sep 1 01:15:41 2009
**      by: Qt User Interface Compiler version 4.5.0
**
** WARNING! All changes made in this file will be lost when recompiling ui file!
********************************************************************************/

#ifndef UI_VIDEOSOURCEDIALOG_H
#define UI_VIDEOSOURCEDIALOG_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QDialog>
#include <QtGui/QGridLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QListWidget>
#include <QtGui/QPushButton>
#include <QtGui/QSpacerItem>

QT_BEGIN_NAMESPACE

class Ui_VideoSourceDialog
{
public:
    QGridLayout *gridLayout;
    QLabel *label;
    QListWidget *lw_video_sources;
    QSpacerItem *horizontalSpacer;
    QPushButton *btn_ok;
    QSpacerItem *horizontalSpacer_2;

    void setupUi(QDialog *VideoSourceDialog)
    {
        if (VideoSourceDialog->objectName().isEmpty())
            VideoSourceDialog->setObjectName(QString::fromUtf8("VideoSourceDialog"));
        VideoSourceDialog->resize(270, 272);
        gridLayout = new QGridLayout(VideoSourceDialog);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        label = new QLabel(VideoSourceDialog);
        label->setObjectName(QString::fromUtf8("label"));
        label->setAlignment(Qt::AlignCenter);

        gridLayout->addWidget(label, 0, 0, 1, 3);

        lw_video_sources = new QListWidget(VideoSourceDialog);
        lw_video_sources->setObjectName(QString::fromUtf8("lw_video_sources"));

        gridLayout->addWidget(lw_video_sources, 1, 0, 1, 3);

        horizontalSpacer = new QSpacerItem(75, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout->addItem(horizontalSpacer, 2, 0, 1, 1);

        btn_ok = new QPushButton(VideoSourceDialog);
        btn_ok->setObjectName(QString::fromUtf8("btn_ok"));

        gridLayout->addWidget(btn_ok, 2, 1, 1, 1);

        horizontalSpacer_2 = new QSpacerItem(74, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout->addItem(horizontalSpacer_2, 2, 2, 1, 1);


        retranslateUi(VideoSourceDialog);

        QMetaObject::connectSlotsByName(VideoSourceDialog);
    } // setupUi

    void retranslateUi(QDialog *VideoSourceDialog)
    {
        VideoSourceDialog->setWindowTitle(QApplication::translate("VideoSourceDialog", "Video Source", 0, QApplication::UnicodeUTF8));
        label->setText(QApplication::translate("VideoSourceDialog", "Please select a video source (camera):", 0, QApplication::UnicodeUTF8));
        btn_ok->setText(QApplication::translate("VideoSourceDialog", "Ok", 0, QApplication::UnicodeUTF8));
        Q_UNUSED(VideoSourceDialog);
    } // retranslateUi

};

namespace Ui {
    class VideoSourceDialog: public Ui_VideoSourceDialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_VIDEOSOURCEDIALOG_H

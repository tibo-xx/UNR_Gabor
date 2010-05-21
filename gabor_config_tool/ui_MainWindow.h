/********************************************************************************
** Form generated from reading ui file 'MainWindow.ui'
**
** Created: Thu May 20 16:35:33 2010
**      by: Qt User Interface Compiler version 4.5.0
**
** WARNING! All changes made in this file will be lost when recompiling ui file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QCheckBox>
#include <QtGui/QDoubleSpinBox>
#include <QtGui/QFrame>
#include <QtGui/QGridLayout>
#include <QtGui/QGroupBox>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QLineEdit>
#include <QtGui/QListWidget>
#include <QtGui/QMainWindow>
#include <QtGui/QMenu>
#include <QtGui/QMenuBar>
#include <QtGui/QPushButton>
#include <QtGui/QSlider>
#include <QtGui/QSpacerItem>
#include <QtGui/QSpinBox>
#include <QtGui/QStatusBar>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QAction *actionQuit;
    QAction *actionSave;
    QAction *actionNew;
    QAction *actionSave_As;
    QAction *actionOpen;
    QWidget *centralwidget;
    QGridLayout *gridLayout_9;
    QGroupBox *groupBox_2;
    QHBoxLayout *horizontalLayout_8;
    QFrame *frame_5;
    QGridLayout *gridLayout;
    QListWidget *lw_configurations;
    QPushButton *btn_add;
    QSpacerItem *horizontalSpacer_4;
    QPushButton *btn_rename;
    QSpacerItem *horizontalSpacer_5;
    QPushButton *btn_move_up;
    QPushButton *btn_remove;
    QPushButton *btn_duplicate;
    QPushButton *btn_move_down;
    QGroupBox *groupBox;
    QHBoxLayout *horizontalLayout_7;
    QFrame *frame_4;
    QGridLayout *gridLayout_7;
    QLabel *lbl_theta;
    QDoubleSpinBox *dsb_theta;
    QSlider *slider_theta;
    QLabel *lbl_sigma;
    QDoubleSpinBox *dsb_sigma;
    QSlider *slider_sigma;
    QLabel *lbl_lambda;
    QDoubleSpinBox *dsb_lambda;
    QSlider *slider_lambda;
    QLabel *lbl_psi;
    QDoubleSpinBox *dsb_psi;
    QSlider *slider_psi;
    QVBoxLayout *verticalLayout;
    QHBoxLayout *horizontalLayout_3;
    QLabel *lbl_fps_text;
    QLabel *lbl_fps_numbers;
    QFrame *frame_preview;
    QHBoxLayout *horizontalLayout_2;
    QSpacerItem *horizontalSpacer_2;
    QGridLayout *gridLayout_2;
    QSlider *slider_filter_opacity;
    QLabel *lbl_filter_opacity;
    QSpacerItem *horizontalSpacer_6;
    QCheckBox *cb_difference_images;
    QSpacerItem *horizontalSpacer_8;
    QSpacerItem *horizontalSpacer_3;
    QGroupBox *groupBox_4;
    QHBoxLayout *horizontalLayout_6;
    QFrame *frame_3;
    QGridLayout *gridLayout_6;
    QSpinBox *sb_div_y;
    QSpinBox *sb_div_x;
    QSpacerItem *horizontalSpacer_7;
    QLabel *label_5;
    QLabel *label_6;
    QGroupBox *groupBox_5;
    QHBoxLayout *horizontalLayout_4;
    QFrame *frame;
    QGridLayout *gridLayout_3;
    QLabel *label_18;
    QDoubleSpinBox *dsb_send_freq;
    QSpacerItem *horizontalSpacer;
    QGroupBox *gcb_ncstools;
    QGridLayout *gridLayout_5;
    QGridLayout *gridLayout_4;
    QLineEdit *le_host_ncstools;
    QSpinBox *sb_port_ncstools;
    QLabel *label_8;
    QLabel *label_7;
    QGroupBox *gcb_server_throttle;
    QHBoxLayout *horizontalLayout;
    QLabel *label_17;
    QDoubleSpinBox *dsb_heartbeat_freq;
    QGroupBox *groupBox_3;
    QHBoxLayout *horizontalLayout_5;
    QFrame *frame_2;
    QGridLayout *gridLayout_8;
    QLineEdit *le_host;
    QSpinBox *sb_port;
    QLineEdit *le_type;
    QLineEdit *le_job;
    QLabel *label;
    QLabel *label_2;
    QLabel *label_3;
    QLabel *label_4;
    QMenuBar *menubar;
    QMenu *menuFile;
    QStatusBar *statusbar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
        MainWindow->resize(1374, 673);
        QSizePolicy sizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(MainWindow->sizePolicy().hasHeightForWidth());
        MainWindow->setSizePolicy(sizePolicy);
        MainWindow->setMinimumSize(QSize(999, 622));
        MainWindow->setMaximumSize(QSize(20000, 20000));
        actionQuit = new QAction(MainWindow);
        actionQuit->setObjectName(QString::fromUtf8("actionQuit"));
        actionSave = new QAction(MainWindow);
        actionSave->setObjectName(QString::fromUtf8("actionSave"));
        actionNew = new QAction(MainWindow);
        actionNew->setObjectName(QString::fromUtf8("actionNew"));
        actionSave_As = new QAction(MainWindow);
        actionSave_As->setObjectName(QString::fromUtf8("actionSave_As"));
        actionOpen = new QAction(MainWindow);
        actionOpen->setObjectName(QString::fromUtf8("actionOpen"));
        centralwidget = new QWidget(MainWindow);
        centralwidget->setObjectName(QString::fromUtf8("centralwidget"));
        gridLayout_9 = new QGridLayout(centralwidget);
        gridLayout_9->setObjectName(QString::fromUtf8("gridLayout_9"));
        groupBox_2 = new QGroupBox(centralwidget);
        groupBox_2->setObjectName(QString::fromUtf8("groupBox_2"));
        horizontalLayout_8 = new QHBoxLayout(groupBox_2);
        horizontalLayout_8->setObjectName(QString::fromUtf8("horizontalLayout_8"));
        frame_5 = new QFrame(groupBox_2);
        frame_5->setObjectName(QString::fromUtf8("frame_5"));
        frame_5->setFrameShape(QFrame::StyledPanel);
        frame_5->setFrameShadow(QFrame::Raised);
        gridLayout = new QGridLayout(frame_5);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        lw_configurations = new QListWidget(frame_5);
        lw_configurations->setObjectName(QString::fromUtf8("lw_configurations"));

        gridLayout->addWidget(lw_configurations, 0, 0, 1, 5);

        btn_add = new QPushButton(frame_5);
        btn_add->setObjectName(QString::fromUtf8("btn_add"));
        QIcon icon;
        icon.addPixmap(QPixmap(QString::fromUtf8(":/icons/icons/Add-64.png")), QIcon::Normal, QIcon::Off);
        btn_add->setIcon(icon);

        gridLayout->addWidget(btn_add, 1, 0, 1, 1);

        horizontalSpacer_4 = new QSpacerItem(13, 25, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout->addItem(horizontalSpacer_4, 1, 1, 1, 1);

        btn_rename = new QPushButton(frame_5);
        btn_rename->setObjectName(QString::fromUtf8("btn_rename"));
        QIcon icon1;
        icon1.addPixmap(QPixmap(QString::fromUtf8(":/icons/icons/Pencil-64.png")), QIcon::Normal, QIcon::Off);
        btn_rename->setIcon(icon1);

        gridLayout->addWidget(btn_rename, 1, 2, 1, 1);

        horizontalSpacer_5 = new QSpacerItem(13, 25, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout->addItem(horizontalSpacer_5, 1, 3, 1, 1);

        btn_move_up = new QPushButton(frame_5);
        btn_move_up->setObjectName(QString::fromUtf8("btn_move_up"));
        btn_move_up->setEnabled(false);
        QIcon icon2;
        icon2.addPixmap(QPixmap(QString::fromUtf8(":/icons/icons/Up-64.png")), QIcon::Normal, QIcon::Off);
        btn_move_up->setIcon(icon2);

        gridLayout->addWidget(btn_move_up, 1, 4, 1, 1);

        btn_remove = new QPushButton(frame_5);
        btn_remove->setObjectName(QString::fromUtf8("btn_remove"));
        btn_remove->setEnabled(false);
        QIcon icon3;
        icon3.addPixmap(QPixmap(QString::fromUtf8(":/icons/icons/Delete-64.png")), QIcon::Normal, QIcon::Off);
        btn_remove->setIcon(icon3);

        gridLayout->addWidget(btn_remove, 2, 0, 1, 1);

        btn_duplicate = new QPushButton(frame_5);
        btn_duplicate->setObjectName(QString::fromUtf8("btn_duplicate"));
        QIcon icon4;
        icon4.addPixmap(QPixmap(QString::fromUtf8(":/icons/icons/Copy-64.png")), QIcon::Normal, QIcon::Off);
        btn_duplicate->setIcon(icon4);

        gridLayout->addWidget(btn_duplicate, 2, 2, 1, 1);

        btn_move_down = new QPushButton(frame_5);
        btn_move_down->setObjectName(QString::fromUtf8("btn_move_down"));
        btn_move_down->setEnabled(false);
        QIcon icon5;
        icon5.addPixmap(QPixmap(QString::fromUtf8(":/icons/icons/Down-64.png")), QIcon::Normal, QIcon::Off);
        btn_move_down->setIcon(icon5);

        gridLayout->addWidget(btn_move_down, 2, 4, 1, 1);


        horizontalLayout_8->addWidget(frame_5);


        gridLayout_9->addWidget(groupBox_2, 0, 0, 2, 1);

        groupBox = new QGroupBox(centralwidget);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        horizontalLayout_7 = new QHBoxLayout(groupBox);
        horizontalLayout_7->setObjectName(QString::fromUtf8("horizontalLayout_7"));
        frame_4 = new QFrame(groupBox);
        frame_4->setObjectName(QString::fromUtf8("frame_4"));
        frame_4->setFrameShape(QFrame::StyledPanel);
        frame_4->setFrameShadow(QFrame::Raised);
        gridLayout_7 = new QGridLayout(frame_4);
        gridLayout_7->setObjectName(QString::fromUtf8("gridLayout_7"));
        lbl_theta = new QLabel(frame_4);
        lbl_theta->setObjectName(QString::fromUtf8("lbl_theta"));
        lbl_theta->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout_7->addWidget(lbl_theta, 0, 0, 1, 1);

        dsb_theta = new QDoubleSpinBox(frame_4);
        dsb_theta->setObjectName(QString::fromUtf8("dsb_theta"));
        dsb_theta->setMaximum(359);
        dsb_theta->setSingleStep(0.5);

        gridLayout_7->addWidget(dsb_theta, 0, 1, 1, 1);

        slider_theta = new QSlider(frame_4);
        slider_theta->setObjectName(QString::fromUtf8("slider_theta"));
        slider_theta->setMinimumSize(QSize(150, 0));
        slider_theta->setMaximum(359);
        slider_theta->setOrientation(Qt::Horizontal);

        gridLayout_7->addWidget(slider_theta, 0, 2, 1, 1);

        lbl_sigma = new QLabel(frame_4);
        lbl_sigma->setObjectName(QString::fromUtf8("lbl_sigma"));
        lbl_sigma->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout_7->addWidget(lbl_sigma, 1, 0, 1, 1);

        dsb_sigma = new QDoubleSpinBox(frame_4);
        dsb_sigma->setObjectName(QString::fromUtf8("dsb_sigma"));
        dsb_sigma->setMinimum(1);
        dsb_sigma->setMaximum(75);
        dsb_sigma->setSingleStep(0.5);
        dsb_sigma->setValue(9);

        gridLayout_7->addWidget(dsb_sigma, 1, 1, 1, 1);

        slider_sigma = new QSlider(frame_4);
        slider_sigma->setObjectName(QString::fromUtf8("slider_sigma"));
        slider_sigma->setMinimumSize(QSize(150, 0));
        slider_sigma->setMinimum(1);
        slider_sigma->setMaximum(75);
        slider_sigma->setValue(9);
        slider_sigma->setOrientation(Qt::Horizontal);

        gridLayout_7->addWidget(slider_sigma, 1, 2, 1, 1);

        lbl_lambda = new QLabel(frame_4);
        lbl_lambda->setObjectName(QString::fromUtf8("lbl_lambda"));
        lbl_lambda->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout_7->addWidget(lbl_lambda, 2, 0, 1, 1);

        dsb_lambda = new QDoubleSpinBox(frame_4);
        dsb_lambda->setObjectName(QString::fromUtf8("dsb_lambda"));
        dsb_lambda->setMinimum(1);
        dsb_lambda->setMaximum(75);
        dsb_lambda->setValue(25);

        gridLayout_7->addWidget(dsb_lambda, 2, 1, 1, 1);

        slider_lambda = new QSlider(frame_4);
        slider_lambda->setObjectName(QString::fromUtf8("slider_lambda"));
        slider_lambda->setMinimumSize(QSize(150, 0));
        slider_lambda->setMinimum(1);
        slider_lambda->setMaximum(75);
        slider_lambda->setValue(25);
        slider_lambda->setOrientation(Qt::Horizontal);

        gridLayout_7->addWidget(slider_lambda, 2, 2, 1, 1);

        lbl_psi = new QLabel(frame_4);
        lbl_psi->setObjectName(QString::fromUtf8("lbl_psi"));
        lbl_psi->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout_7->addWidget(lbl_psi, 3, 0, 1, 1);

        dsb_psi = new QDoubleSpinBox(frame_4);
        dsb_psi->setObjectName(QString::fromUtf8("dsb_psi"));
        dsb_psi->setMaximum(359);
        dsb_psi->setSingleStep(0.5);

        gridLayout_7->addWidget(dsb_psi, 3, 1, 1, 1);

        slider_psi = new QSlider(frame_4);
        slider_psi->setObjectName(QString::fromUtf8("slider_psi"));
        slider_psi->setMinimumSize(QSize(150, 0));
        slider_psi->setMaximum(359);
        slider_psi->setOrientation(Qt::Horizontal);

        gridLayout_7->addWidget(slider_psi, 3, 2, 1, 1);


        horizontalLayout_7->addWidget(frame_4);


        gridLayout_9->addWidget(groupBox, 0, 1, 1, 1);

        verticalLayout = new QVBoxLayout();
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        lbl_fps_text = new QLabel(centralwidget);
        lbl_fps_text->setObjectName(QString::fromUtf8("lbl_fps_text"));
        lbl_fps_text->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        horizontalLayout_3->addWidget(lbl_fps_text);

        lbl_fps_numbers = new QLabel(centralwidget);
        lbl_fps_numbers->setObjectName(QString::fromUtf8("lbl_fps_numbers"));
        lbl_fps_numbers->setAlignment(Qt::AlignLeading|Qt::AlignLeft|Qt::AlignVCenter);

        horizontalLayout_3->addWidget(lbl_fps_numbers);


        verticalLayout->addLayout(horizontalLayout_3);

        frame_preview = new QFrame(centralwidget);
        frame_preview->setObjectName(QString::fromUtf8("frame_preview"));
        sizePolicy.setHeightForWidth(frame_preview->sizePolicy().hasHeightForWidth());
        frame_preview->setSizePolicy(sizePolicy);
        frame_preview->setMinimumSize(QSize(640, 480));
        frame_preview->setFrameShape(QFrame::StyledPanel);
        frame_preview->setFrameShadow(QFrame::Raised);

        verticalLayout->addWidget(frame_preview);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        horizontalSpacer_2 = new QSpacerItem(148, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_2->addItem(horizontalSpacer_2);

        gridLayout_2 = new QGridLayout();
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        slider_filter_opacity = new QSlider(centralwidget);
        slider_filter_opacity->setObjectName(QString::fromUtf8("slider_filter_opacity"));
        slider_filter_opacity->setEnabled(true);
        slider_filter_opacity->setMaximum(100);
        slider_filter_opacity->setValue(100);
        slider_filter_opacity->setOrientation(Qt::Horizontal);

        gridLayout_2->addWidget(slider_filter_opacity, 0, 0, 1, 3);

        lbl_filter_opacity = new QLabel(centralwidget);
        lbl_filter_opacity->setObjectName(QString::fromUtf8("lbl_filter_opacity"));
        lbl_filter_opacity->setAlignment(Qt::AlignCenter);

        gridLayout_2->addWidget(lbl_filter_opacity, 1, 0, 1, 3);

        horizontalSpacer_6 = new QSpacerItem(28, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout_2->addItem(horizontalSpacer_6, 2, 0, 1, 1);

        cb_difference_images = new QCheckBox(centralwidget);
        cb_difference_images->setObjectName(QString::fromUtf8("cb_difference_images"));
        cb_difference_images->setChecked(true);

        gridLayout_2->addWidget(cb_difference_images, 2, 1, 1, 1);

        horizontalSpacer_8 = new QSpacerItem(28, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout_2->addItem(horizontalSpacer_8, 2, 2, 1, 1);


        horizontalLayout_2->addLayout(gridLayout_2);

        horizontalSpacer_3 = new QSpacerItem(148, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_2->addItem(horizontalSpacer_3);


        verticalLayout->addLayout(horizontalLayout_2);


        gridLayout_9->addLayout(verticalLayout, 0, 2, 3, 1);

        groupBox_4 = new QGroupBox(centralwidget);
        groupBox_4->setObjectName(QString::fromUtf8("groupBox_4"));
        horizontalLayout_6 = new QHBoxLayout(groupBox_4);
        horizontalLayout_6->setObjectName(QString::fromUtf8("horizontalLayout_6"));
        frame_3 = new QFrame(groupBox_4);
        frame_3->setObjectName(QString::fromUtf8("frame_3"));
        frame_3->setFrameShape(QFrame::StyledPanel);
        frame_3->setFrameShadow(QFrame::Raised);
        gridLayout_6 = new QGridLayout(frame_3);
        gridLayout_6->setObjectName(QString::fromUtf8("gridLayout_6"));
        sb_div_y = new QSpinBox(frame_3);
        sb_div_y->setObjectName(QString::fromUtf8("sb_div_y"));
        sb_div_y->setMinimum(1);
        sb_div_y->setMaximum(128);
        sb_div_y->setValue(2);

        gridLayout_6->addWidget(sb_div_y, 1, 1, 1, 1);

        sb_div_x = new QSpinBox(frame_3);
        sb_div_x->setObjectName(QString::fromUtf8("sb_div_x"));
        sb_div_x->setMinimum(1);
        sb_div_x->setMaximum(128);
        sb_div_x->setValue(2);

        gridLayout_6->addWidget(sb_div_x, 0, 1, 1, 1);

        horizontalSpacer_7 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout_6->addItem(horizontalSpacer_7, 0, 3, 1, 1);

        label_5 = new QLabel(frame_3);
        label_5->setObjectName(QString::fromUtf8("label_5"));

        gridLayout_6->addWidget(label_5, 0, 0, 1, 1);

        label_6 = new QLabel(frame_3);
        label_6->setObjectName(QString::fromUtf8("label_6"));

        gridLayout_6->addWidget(label_6, 1, 0, 1, 1);

        sb_div_y->raise();
        sb_div_x->raise();
        label_5->raise();
        label_6->raise();

        horizontalLayout_6->addWidget(frame_3);


        gridLayout_9->addWidget(groupBox_4, 1, 1, 1, 1);

        groupBox_5 = new QGroupBox(centralwidget);
        groupBox_5->setObjectName(QString::fromUtf8("groupBox_5"));
        horizontalLayout_4 = new QHBoxLayout(groupBox_5);
        horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));
        frame = new QFrame(groupBox_5);
        frame->setObjectName(QString::fromUtf8("frame"));
        frame->setFrameShape(QFrame::StyledPanel);
        frame->setFrameShadow(QFrame::Raised);
        gridLayout_3 = new QGridLayout(frame);
        gridLayout_3->setObjectName(QString::fromUtf8("gridLayout_3"));
        label_18 = new QLabel(frame);
        label_18->setObjectName(QString::fromUtf8("label_18"));

        gridLayout_3->addWidget(label_18, 0, 0, 1, 1);

        dsb_send_freq = new QDoubleSpinBox(frame);
        dsb_send_freq->setObjectName(QString::fromUtf8("dsb_send_freq"));
        dsb_send_freq->setDecimals(1);
        dsb_send_freq->setMinimum(1);
        dsb_send_freq->setMaximum(1e+09);
        dsb_send_freq->setValue(100);

        gridLayout_3->addWidget(dsb_send_freq, 0, 1, 1, 1);

        horizontalSpacer = new QSpacerItem(68, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout_3->addItem(horizontalSpacer, 0, 2, 1, 1);

        gcb_ncstools = new QGroupBox(frame);
        gcb_ncstools->setObjectName(QString::fromUtf8("gcb_ncstools"));
        gcb_ncstools->setFlat(false);
        gcb_ncstools->setCheckable(true);
        gcb_ncstools->setChecked(false);
        gridLayout_5 = new QGridLayout(gcb_ncstools);
        gridLayout_5->setObjectName(QString::fromUtf8("gridLayout_5"));
        gridLayout_4 = new QGridLayout();
        gridLayout_4->setObjectName(QString::fromUtf8("gridLayout_4"));
        le_host_ncstools = new QLineEdit(gcb_ncstools);
        le_host_ncstools->setObjectName(QString::fromUtf8("le_host_ncstools"));

        gridLayout_4->addWidget(le_host_ncstools, 0, 1, 1, 1);

        sb_port_ncstools = new QSpinBox(gcb_ncstools);
        sb_port_ncstools->setObjectName(QString::fromUtf8("sb_port_ncstools"));
        sb_port_ncstools->setMinimum(1);
        sb_port_ncstools->setMaximum(999999999);
        sb_port_ncstools->setValue(20001);

        gridLayout_4->addWidget(sb_port_ncstools, 1, 1, 1, 1);

        label_8 = new QLabel(gcb_ncstools);
        label_8->setObjectName(QString::fromUtf8("label_8"));

        gridLayout_4->addWidget(label_8, 0, 0, 1, 1);

        label_7 = new QLabel(gcb_ncstools);
        label_7->setObjectName(QString::fromUtf8("label_7"));

        gridLayout_4->addWidget(label_7, 1, 0, 1, 1);


        gridLayout_5->addLayout(gridLayout_4, 0, 0, 1, 1);

        gcb_server_throttle = new QGroupBox(gcb_ncstools);
        gcb_server_throttle->setObjectName(QString::fromUtf8("gcb_server_throttle"));
        gcb_server_throttle->setCheckable(true);
        gcb_server_throttle->setChecked(false);
        horizontalLayout = new QHBoxLayout(gcb_server_throttle);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        label_17 = new QLabel(gcb_server_throttle);
        label_17->setObjectName(QString::fromUtf8("label_17"));

        horizontalLayout->addWidget(label_17);

        dsb_heartbeat_freq = new QDoubleSpinBox(gcb_server_throttle);
        dsb_heartbeat_freq->setObjectName(QString::fromUtf8("dsb_heartbeat_freq"));
        dsb_heartbeat_freq->setDecimals(1);
        dsb_heartbeat_freq->setMinimum(1);
        dsb_heartbeat_freq->setMaximum(1e+09);
        dsb_heartbeat_freq->setValue(10);

        horizontalLayout->addWidget(dsb_heartbeat_freq);


        gridLayout_5->addWidget(gcb_server_throttle, 1, 0, 1, 2);


        gridLayout_3->addWidget(gcb_ncstools, 1, 0, 1, 3);


        horizontalLayout_4->addWidget(frame);


        gridLayout_9->addWidget(groupBox_5, 2, 0, 1, 1);

        groupBox_3 = new QGroupBox(centralwidget);
        groupBox_3->setObjectName(QString::fromUtf8("groupBox_3"));
        horizontalLayout_5 = new QHBoxLayout(groupBox_3);
        horizontalLayout_5->setObjectName(QString::fromUtf8("horizontalLayout_5"));
        frame_2 = new QFrame(groupBox_3);
        frame_2->setObjectName(QString::fromUtf8("frame_2"));
        frame_2->setFrameShape(QFrame::StyledPanel);
        frame_2->setFrameShadow(QFrame::Raised);
        gridLayout_8 = new QGridLayout(frame_2);
        gridLayout_8->setObjectName(QString::fromUtf8("gridLayout_8"));
        le_host = new QLineEdit(frame_2);
        le_host->setObjectName(QString::fromUtf8("le_host"));

        gridLayout_8->addWidget(le_host, 0, 1, 1, 1);

        sb_port = new QSpinBox(frame_2);
        sb_port->setObjectName(QString::fromUtf8("sb_port"));
        sb_port->setMinimum(1);
        sb_port->setMaximum(999999999);
        sb_port->setValue(20001);

        gridLayout_8->addWidget(sb_port, 1, 1, 1, 1);

        le_type = new QLineEdit(frame_2);
        le_type->setObjectName(QString::fromUtf8("le_type"));

        gridLayout_8->addWidget(le_type, 2, 1, 1, 1);

        le_job = new QLineEdit(frame_2);
        le_job->setObjectName(QString::fromUtf8("le_job"));

        gridLayout_8->addWidget(le_job, 3, 1, 1, 1);

        label = new QLabel(frame_2);
        label->setObjectName(QString::fromUtf8("label"));

        gridLayout_8->addWidget(label, 0, 0, 1, 1);

        label_2 = new QLabel(frame_2);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        gridLayout_8->addWidget(label_2, 1, 0, 1, 1);

        label_3 = new QLabel(frame_2);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        gridLayout_8->addWidget(label_3, 2, 0, 1, 1);

        label_4 = new QLabel(frame_2);
        label_4->setObjectName(QString::fromUtf8("label_4"));

        gridLayout_8->addWidget(label_4, 3, 0, 1, 1);


        horizontalLayout_5->addWidget(frame_2);


        gridLayout_9->addWidget(groupBox_3, 2, 1, 1, 1);

        MainWindow->setCentralWidget(centralwidget);
        menubar = new QMenuBar(MainWindow);
        menubar->setObjectName(QString::fromUtf8("menubar"));
        menubar->setGeometry(QRect(0, 0, 1374, 25));
        menuFile = new QMenu(menubar);
        menuFile->setObjectName(QString::fromUtf8("menuFile"));
        MainWindow->setMenuBar(menubar);
        statusbar = new QStatusBar(MainWindow);
        statusbar->setObjectName(QString::fromUtf8("statusbar"));
        MainWindow->setStatusBar(statusbar);

        menubar->addAction(menuFile->menuAction());
        menuFile->addAction(actionNew);
        menuFile->addAction(actionSave);
        menuFile->addAction(actionOpen);
        menuFile->addSeparator();
        menuFile->addAction(actionQuit);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "Gabor Configuration Tool", 0, QApplication::UnicodeUTF8));
        actionQuit->setText(QApplication::translate("MainWindow", "Quit", 0, QApplication::UnicodeUTF8));
        actionQuit->setShortcut(QApplication::translate("MainWindow", "Ctrl+Q", 0, QApplication::UnicodeUTF8));
        actionSave->setText(QApplication::translate("MainWindow", "Save", 0, QApplication::UnicodeUTF8));
        actionSave->setShortcut(QApplication::translate("MainWindow", "Ctrl+S", 0, QApplication::UnicodeUTF8));
        actionNew->setText(QApplication::translate("MainWindow", "New", 0, QApplication::UnicodeUTF8));
        actionNew->setShortcut(QApplication::translate("MainWindow", "Ctrl+N", 0, QApplication::UnicodeUTF8));
        actionSave_As->setText(QApplication::translate("MainWindow", "Save As", 0, QApplication::UnicodeUTF8));
        actionSave_As->setShortcut(QApplication::translate("MainWindow", "Ctrl+Shift+S", 0, QApplication::UnicodeUTF8));
        actionOpen->setText(QApplication::translate("MainWindow", "Open", 0, QApplication::UnicodeUTF8));
        actionOpen->setShortcut(QApplication::translate("MainWindow", "Ctrl+O", 0, QApplication::UnicodeUTF8));
        groupBox_2->setTitle(QApplication::translate("MainWindow", "Create Filters", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        btn_add->setToolTip(QApplication::translate("MainWindow", "Create a new configuration.", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        btn_add->setText(QString());
#ifndef QT_NO_TOOLTIP
        btn_rename->setToolTip(QApplication::translate("MainWindow", "Rename the selected configuration.", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        btn_rename->setText(QString());
#ifndef QT_NO_TOOLTIP
        btn_move_up->setToolTip(QApplication::translate("MainWindow", "Move the selected configuration up.", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        btn_move_up->setText(QString());
#ifndef QT_NO_TOOLTIP
        btn_remove->setToolTip(QApplication::translate("MainWindow", "Remove the selected configuration.", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        btn_remove->setText(QString());
#ifndef QT_NO_TOOLTIP
        btn_duplicate->setToolTip(QApplication::translate("MainWindow", "Duplicated the selected configuration.", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        btn_duplicate->setText(QString());
#ifndef QT_NO_TOOLTIP
        btn_move_down->setToolTip(QApplication::translate("MainWindow", "Move the selected configuration down.", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        btn_move_down->setText(QString());
        groupBox->setTitle(QApplication::translate("MainWindow", "Filter Configuration", 0, QApplication::UnicodeUTF8));
        lbl_theta->setText(QApplication::translate("MainWindow", "Theta (deg)", 0, QApplication::UnicodeUTF8));
        lbl_sigma->setText(QApplication::translate("MainWindow", "Sigma (px)", 0, QApplication::UnicodeUTF8));
        lbl_lambda->setText(QApplication::translate("MainWindow", "Lambda (px)", 0, QApplication::UnicodeUTF8));
        lbl_psi->setText(QApplication::translate("MainWindow", "Psi (deg)", 0, QApplication::UnicodeUTF8));
        lbl_fps_text->setText(QApplication::translate("MainWindow", "FPS:", 0, QApplication::UnicodeUTF8));
        lbl_fps_numbers->setText(QApplication::translate("MainWindow", "0.0", 0, QApplication::UnicodeUTF8));
        lbl_filter_opacity->setText(QApplication::translate("MainWindow", "Filter Opacity", 0, QApplication::UnicodeUTF8));
        cb_difference_images->setText(QApplication::translate("MainWindow", "Difference Images", 0, QApplication::UnicodeUTF8));
        groupBox_4->setTitle(QApplication::translate("MainWindow", "Image Division", 0, QApplication::UnicodeUTF8));
        label_5->setText(QApplication::translate("MainWindow", "X", 0, QApplication::UnicodeUTF8));
        label_6->setText(QApplication::translate("MainWindow", "Y", 0, QApplication::UnicodeUTF8));
        groupBox_5->setTitle(QApplication::translate("MainWindow", "Global Configuration", 0, QApplication::UnicodeUTF8));
        label_18->setText(QApplication::translate("MainWindow", "Send Freq (ms)", 0, QApplication::UnicodeUTF8));
        gcb_ncstools->setTitle(QApplication::translate("MainWindow", "Connect to NCSTools", 0, QApplication::UnicodeUTF8));
        le_host_ncstools->setText(QApplication::translate("MainWindow", "localhost", 0, QApplication::UnicodeUTF8));
        label_8->setText(QApplication::translate("MainWindow", "Host", 0, QApplication::UnicodeUTF8));
        label_7->setText(QApplication::translate("MainWindow", "Port", 0, QApplication::UnicodeUTF8));
        gcb_server_throttle->setTitle(QApplication::translate("MainWindow", "Server Throttle", 0, QApplication::UnicodeUTF8));
        label_17->setText(QApplication::translate("MainWindow", "Heartbeat Freq (ms)", 0, QApplication::UnicodeUTF8));
        groupBox_3->setTitle(QApplication::translate("MainWindow", "NCS Communication (filter specific)", 0, QApplication::UnicodeUTF8));
        le_host->setText(QApplication::translate("MainWindow", "localhost", 0, QApplication::UnicodeUTF8));
        le_type->setText(QApplication::translate("MainWindow", "default", 0, QApplication::UnicodeUTF8));
        le_job->setText(QApplication::translate("MainWindow", "default", 0, QApplication::UnicodeUTF8));
        label->setText(QApplication::translate("MainWindow", "Host", 0, QApplication::UnicodeUTF8));
        label_2->setText(QApplication::translate("MainWindow", "Port", 0, QApplication::UnicodeUTF8));
        label_3->setText(QApplication::translate("MainWindow", "Type", 0, QApplication::UnicodeUTF8));
        label_4->setText(QApplication::translate("MainWindow", "Job", 0, QApplication::UnicodeUTF8));
        menuFile->setTitle(QApplication::translate("MainWindow", "File", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H

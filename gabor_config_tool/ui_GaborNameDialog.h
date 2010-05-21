/********************************************************************************
** Form generated from reading ui file 'GaborNameDialog.ui'
**
** Created: Thu May 20 16:35:33 2010
**      by: Qt User Interface Compiler version 4.5.0
**
** WARNING! All changes made in this file will be lost when recompiling ui file!
********************************************************************************/

#ifndef UI_GABORNAMEDIALOG_H
#define UI_GABORNAMEDIALOG_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QDialog>
#include <QtGui/QDialogButtonBox>
#include <QtGui/QGridLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QLineEdit>

QT_BEGIN_NAMESPACE

class Ui_GaborNameDialog
{
public:
    QGridLayout *gridLayout;
    QLabel *lbl_message;
    QLabel *lbl_name;
    QLineEdit *le_name;
    QDialogButtonBox *buttonBox;

    void setupUi(QDialog *GaborNameDialog)
    {
        if (GaborNameDialog->objectName().isEmpty())
            GaborNameDialog->setObjectName(QString::fromUtf8("GaborNameDialog"));
        GaborNameDialog->resize(364, 108);
        QSizePolicy sizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(GaborNameDialog->sizePolicy().hasHeightForWidth());
        GaborNameDialog->setSizePolicy(sizePolicy);
        GaborNameDialog->setMaximumSize(QSize(364, 108));
        QIcon icon;
        icon.addPixmap(QPixmap(QString::fromUtf8(":/icons/icons/Pencil-64.png")), QIcon::Normal, QIcon::Off);
        GaborNameDialog->setWindowIcon(icon);
        GaborNameDialog->setSizeGripEnabled(false);
        GaborNameDialog->setModal(true);
        gridLayout = new QGridLayout(GaborNameDialog);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        lbl_message = new QLabel(GaborNameDialog);
        lbl_message->setObjectName(QString::fromUtf8("lbl_message"));

        gridLayout->addWidget(lbl_message, 0, 0, 1, 2);

        lbl_name = new QLabel(GaborNameDialog);
        lbl_name->setObjectName(QString::fromUtf8("lbl_name"));
        lbl_name->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout->addWidget(lbl_name, 1, 0, 1, 1);

        le_name = new QLineEdit(GaborNameDialog);
        le_name->setObjectName(QString::fromUtf8("le_name"));

        gridLayout->addWidget(le_name, 1, 1, 1, 1);

        buttonBox = new QDialogButtonBox(GaborNameDialog);
        buttonBox->setObjectName(QString::fromUtf8("buttonBox"));
        buttonBox->setOrientation(Qt::Horizontal);
        buttonBox->setStandardButtons(QDialogButtonBox::Cancel|QDialogButtonBox::Ok);

        gridLayout->addWidget(buttonBox, 2, 0, 1, 2);


        retranslateUi(GaborNameDialog);
        QObject::connect(buttonBox, SIGNAL(accepted()), GaborNameDialog, SLOT(accept()));
        QObject::connect(buttonBox, SIGNAL(rejected()), GaborNameDialog, SLOT(reject()));

        QMetaObject::connectSlotsByName(GaborNameDialog);
    } // setupUi

    void retranslateUi(QDialog *GaborNameDialog)
    {
        GaborNameDialog->setWindowTitle(QApplication::translate("GaborNameDialog", "New", 0, QApplication::UnicodeUTF8));
        lbl_message->setText(QApplication::translate("GaborNameDialog", "Please enter a new name for the gabor configuration.", 0, QApplication::UnicodeUTF8));
        lbl_name->setText(QApplication::translate("GaborNameDialog", "Name:", 0, QApplication::UnicodeUTF8));
        le_name->setText(QApplication::translate("GaborNameDialog", "untitled", 0, QApplication::UnicodeUTF8));
        Q_UNUSED(GaborNameDialog);
    } // retranslateUi

};

namespace Ui {
    class GaborNameDialog: public Ui_GaborNameDialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_GABORNAMEDIALOG_H

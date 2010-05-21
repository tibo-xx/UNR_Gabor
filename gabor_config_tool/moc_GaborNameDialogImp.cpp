/****************************************************************************
** Meta object code from reading C++ file 'GaborNameDialogImp.h'
**
** Created: Thu May 20 16:35:42 2010
**      by: The Qt Meta Object Compiler version 61 (Qt 4.5.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "GaborNameDialogImp.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'GaborNameDialogImp.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 61
#error "This file was generated using the moc from 4.5.0. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_GaborNameDialogImp[] = {

 // content:
       2,       // revision
       0,       // classname
       0,    0, // classinfo
       0,    0, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors

       0        // eod
};

static const char qt_meta_stringdata_GaborNameDialogImp[] = {
    "GaborNameDialogImp\0"
};

const QMetaObject GaborNameDialogImp::staticMetaObject = {
    { &QDialog::staticMetaObject, qt_meta_stringdata_GaborNameDialogImp,
      qt_meta_data_GaborNameDialogImp, 0 }
};

const QMetaObject *GaborNameDialogImp::metaObject() const
{
    return &staticMetaObject;
}

void *GaborNameDialogImp::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_GaborNameDialogImp))
        return static_cast<void*>(const_cast< GaborNameDialogImp*>(this));
    if (!strcmp(_clname, "Ui::GaborNameDialog"))
        return static_cast< Ui::GaborNameDialog*>(const_cast< GaborNameDialogImp*>(this));
    return QDialog::qt_metacast(_clname);
}

int GaborNameDialogImp::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QDialog::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    return _id;
}
QT_END_MOC_NAMESPACE

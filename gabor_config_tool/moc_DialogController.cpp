/****************************************************************************
** Meta object code from reading C++ file 'DialogController.h'
**
** Created: Tue Sep 1 12:07:07 2009
**      by: The Qt Meta Object Compiler version 61 (Qt 4.5.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "DialogController.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'DialogController.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 61
#error "This file was generated using the moc from 4.5.0. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_DialogController[] = {

 // content:
       2,       // revision
       0,       // classname
       0,    0, // classinfo
       2,   12, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors

 // slots: signature, parameters, type, tag, flags
      18,   17,   17,   17, 0x0a,
      42,   17,   17,   17, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_DialogController[] = {
    "DialogController\0\0showVideoSourceDialog()\0"
    "openMainWindow()\0"
};

const QMetaObject DialogController::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_DialogController,
      qt_meta_data_DialogController, 0 }
};

const QMetaObject *DialogController::metaObject() const
{
    return &staticMetaObject;
}

void *DialogController::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_DialogController))
        return static_cast<void*>(const_cast< DialogController*>(this));
    return QWidget::qt_metacast(_clname);
}

int DialogController::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: showVideoSourceDialog(); break;
        case 1: openMainWindow(); break;
        default: ;
        }
        _id -= 2;
    }
    return _id;
}
QT_END_MOC_NAMESPACE

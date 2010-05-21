/****************************************************************************
** Meta object code from reading C++ file 'WorkerThread.h'
**
** Created: Thu May 20 16:35:45 2010
**      by: The Qt Meta Object Compiler version 61 (Qt 4.5.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "WorkerThread.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'WorkerThread.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 61
#error "This file was generated using the moc from 4.5.0. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_WorkerThread[] = {

 // content:
       2,       // revision
       0,       // classname
       0,    0, // classinfo
       4,   12, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors

 // signals: signature, parameters, type, tag, flags
      14,   13,   13,   13, 0x05,
      31,   13,   13,   13, 0x05,

 // slots: signature, parameters, type, tag, flags
      71,   48,   13,   13, 0x0a,
     121,  119,   13,   13, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_WorkerThread[] = {
    "WorkerThread\0\0filterComplete()\0"
    "newFilterImage()\0theta,sigma,lambda,psi\0"
    "setNewFilterParameters(float,float,float,float)\0"
    "b\0setDifferenceImages(bool)\0"
};

const QMetaObject WorkerThread::staticMetaObject = {
    { &QThread::staticMetaObject, qt_meta_stringdata_WorkerThread,
      qt_meta_data_WorkerThread, 0 }
};

const QMetaObject *WorkerThread::metaObject() const
{
    return &staticMetaObject;
}

void *WorkerThread::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_WorkerThread))
        return static_cast<void*>(const_cast< WorkerThread*>(this));
    return QThread::qt_metacast(_clname);
}

int WorkerThread::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QThread::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: filterComplete(); break;
        case 1: newFilterImage(); break;
        case 2: setNewFilterParameters((*reinterpret_cast< float(*)>(_a[1])),(*reinterpret_cast< float(*)>(_a[2])),(*reinterpret_cast< float(*)>(_a[3])),(*reinterpret_cast< float(*)>(_a[4]))); break;
        case 3: setDifferenceImages((*reinterpret_cast< bool(*)>(_a[1]))); break;
        default: ;
        }
        _id -= 4;
    }
    return _id;
}

// SIGNAL 0
void WorkerThread::filterComplete()
{
    QMetaObject::activate(this, &staticMetaObject, 0, 0);
}

// SIGNAL 1
void WorkerThread::newFilterImage()
{
    QMetaObject::activate(this, &staticMetaObject, 1, 0);
}
QT_END_MOC_NAMESPACE

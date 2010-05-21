/****************************************************************************
** Meta object code from reading C++ file 'GLContext.h'
**
** Created: Thu May 20 16:35:43 2010
**      by: The Qt Meta Object Compiler version 61 (Qt 4.5.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "GLContext.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'GLContext.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 61
#error "This file was generated using the moc from 4.5.0. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_GLContext[] = {

 // content:
       2,       // revision
       0,       // classname
       0,    0, // classinfo
       8,   12, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors

 // signals: signature, parameters, type, tag, flags
      11,   10,   10,   10, 0x05,

 // slots: signature, parameters, type, tag, flags
      25,   10,   10,   10, 0x0a,
      41,   10,   10,   10, 0x0a,
      72,   49,   10,   10, 0x0a,
     120,   10,   10,   10, 0x0a,
     146,  144,   10,   10, 0x0a,
     171,   10,   10,   10, 0x08,
     183,   10,   10,   10, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_GLContext[] = {
    "GLContext\0\0newFPS(float)\0displayImages()\0"
    "start()\0theta,sigma,lambda,psi\0"
    "setNewFilterParameters(float,float,float,float)\0"
    "setFilterOpacity(float)\0b\0"
    "setDifferenceImages(int)\0updateFPS()\0"
    "updateFilterImage()\0"
};

const QMetaObject GLContext::staticMetaObject = {
    { &QGLWidget::staticMetaObject, qt_meta_stringdata_GLContext,
      qt_meta_data_GLContext, 0 }
};

const QMetaObject *GLContext::metaObject() const
{
    return &staticMetaObject;
}

void *GLContext::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_GLContext))
        return static_cast<void*>(const_cast< GLContext*>(this));
    return QGLWidget::qt_metacast(_clname);
}

int GLContext::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QGLWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: newFPS((*reinterpret_cast< float(*)>(_a[1]))); break;
        case 1: displayImages(); break;
        case 2: start(); break;
        case 3: setNewFilterParameters((*reinterpret_cast< float(*)>(_a[1])),(*reinterpret_cast< float(*)>(_a[2])),(*reinterpret_cast< float(*)>(_a[3])),(*reinterpret_cast< float(*)>(_a[4]))); break;
        case 4: setFilterOpacity((*reinterpret_cast< float(*)>(_a[1]))); break;
        case 5: setDifferenceImages((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 6: updateFPS(); break;
        case 7: updateFilterImage(); break;
        default: ;
        }
        _id -= 8;
    }
    return _id;
}

// SIGNAL 0
void GLContext::newFPS(float _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}
QT_END_MOC_NAMESPACE

/****************************************************************************
** Meta object code from reading C++ file 'MainWindowImp.h'
**
** Created: Thu May 20 16:35:44 2010
**      by: The Qt Meta Object Compiler version 61 (Qt 4.5.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "MainWindowImp.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'MainWindowImp.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 61
#error "This file was generated using the moc from 4.5.0. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_MainWindowImp[] = {

 // content:
       2,       // revision
       0,       // classname
       0,    0, // classinfo
      35,   12, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors

 // slots: signature, parameters, type, tag, flags
      15,   14,   14,   14, 0x08,
      38,   14,   14,   14, 0x08,
      63,   14,   14,   14, 0x08,
      88,   14,   14,   14, 0x08,
     112,   14,   14,   14, 0x08,
     133,   14,   14,   14, 0x08,
     151,   14,   14,   14, 0x08,
     172,   14,   14,   14, 0x08,
     190,   14,   14,   14, 0x08,
     212,   14,   14,   14, 0x08,
     231,   14,   14,   14, 0x08,
     250,   14,   14,   14, 0x08,
     266,   14,   14,   14, 0x08,
     291,   14,   14,   14, 0x08,
     313,   14,   14,   14, 0x08,
     335,   14,   14,   14, 0x08,
     369,   14,   14,   14, 0x08,
     390,   14,   14,   14, 0x08,
     415,   14,   14,   14, 0x08,
     437,   14,   14,   14, 0x08,
     461,   14,   14,   14, 0x08,
     472,   14,   14,   14, 0x08,
     479,   14,   14,   14, 0x08,
     486,   14,   14,   14, 0x08,
     500,   14,   14,   14, 0x08,
     514,   14,   14,   14, 0x08,
     532,   14,   14,   14, 0x08,
     555,   14,   14,   14, 0x08,
     573,   14,   14,   14, 0x08,
     597,   14,   14,   14, 0x08,
     619,   14,   14,   14, 0x08,
     641,   14,   14,   14, 0x08,
     655,   14,   14,   14, 0x08,
     669,   14,   14,   14, 0x08,
     682,   14,   14,   14, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_MainWindowImp[] = {
    "MainWindowImp\0\0diffImagesChanged(int)\0"
    "filterParameterChanged()\0"
    "updateFilterOpacity(int)\0"
    "updateFPSDisplay(float)\0sliderThetaChanged()\0"
    "dsbThetaChanged()\0sliderSigmaChanged()\0"
    "dsbSigmaChanged()\0sliderLambdaChanged()\0"
    "dsbLambdaChanged()\0sliderPsiChanged()\0"
    "dsbPsiChanged()\0createNewConfiguration()\0"
    "renameConfiguration()\0removeConfiguration()\0"
    "updateConfigurationSelection(int)\0"
    "updateButtonStates()\0duplicateConfiguration()\0"
    "moveConfigurationUp()\0moveConfigurationDown()\0"
    "newSetup()\0save()\0open()\0divXChanged()\0"
    "divYChanged()\0sendFreqChanged()\0"
    "heartbeatFreqChanged()\0NCSToolsChanged()\0"
    "serverThrottleChanged()\0portNCSToolsChanged()\0"
    "hostNCSToolsChanged()\0hostChanged()\0"
    "typeChanged()\0jobChanged()\0portChanged()\0"
};

const QMetaObject MainWindowImp::staticMetaObject = {
    { &QMainWindow::staticMetaObject, qt_meta_stringdata_MainWindowImp,
      qt_meta_data_MainWindowImp, 0 }
};

const QMetaObject *MainWindowImp::metaObject() const
{
    return &staticMetaObject;
}

void *MainWindowImp::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_MainWindowImp))
        return static_cast<void*>(const_cast< MainWindowImp*>(this));
    if (!strcmp(_clname, "Ui::MainWindow"))
        return static_cast< Ui::MainWindow*>(const_cast< MainWindowImp*>(this));
    return QMainWindow::qt_metacast(_clname);
}

int MainWindowImp::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: diffImagesChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 1: filterParameterChanged(); break;
        case 2: updateFilterOpacity((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 3: updateFPSDisplay((*reinterpret_cast< float(*)>(_a[1]))); break;
        case 4: sliderThetaChanged(); break;
        case 5: dsbThetaChanged(); break;
        case 6: sliderSigmaChanged(); break;
        case 7: dsbSigmaChanged(); break;
        case 8: sliderLambdaChanged(); break;
        case 9: dsbLambdaChanged(); break;
        case 10: sliderPsiChanged(); break;
        case 11: dsbPsiChanged(); break;
        case 12: createNewConfiguration(); break;
        case 13: renameConfiguration(); break;
        case 14: removeConfiguration(); break;
        case 15: updateConfigurationSelection((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 16: updateButtonStates(); break;
        case 17: duplicateConfiguration(); break;
        case 18: moveConfigurationUp(); break;
        case 19: moveConfigurationDown(); break;
        case 20: newSetup(); break;
        case 21: save(); break;
        case 22: open(); break;
        case 23: divXChanged(); break;
        case 24: divYChanged(); break;
        case 25: sendFreqChanged(); break;
        case 26: heartbeatFreqChanged(); break;
        case 27: NCSToolsChanged(); break;
        case 28: serverThrottleChanged(); break;
        case 29: portNCSToolsChanged(); break;
        case 30: hostNCSToolsChanged(); break;
        case 31: hostChanged(); break;
        case 32: typeChanged(); break;
        case 33: jobChanged(); break;
        case 34: portChanged(); break;
        default: ;
        }
        _id -= 35;
    }
    return _id;
}
QT_END_MOC_NAMESPACE

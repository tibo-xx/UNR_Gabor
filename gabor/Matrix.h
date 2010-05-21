#ifndef __MATRIX_H__
#define __MATRIX_H__

#define pi 3.1415926
#define FALSE 0
#define TRUE  1
#define SWAP(a, b) tempr=(a); (a)=(b); (b) = tempr;


/** Function for fourier transform.
 */
void fourn( float data[], unsigned long nn[], int ndim, int isign );

/** Definition of Matrix Class.
 */
class Matrix 
{
private:
    float *p;
    long Row;
    long Col;
    bool bComplex;

public:
    // Useful member functions 
    Matrix();
    virtual ~Matrix();
    long GetRows(void);
    long GetCols(void);
    Matrix(long n);
    Matrix(long row ,long col); 
    Matrix(unsigned char *image, long imgHeight, long imgWidth);    
    Matrix(const Matrix &m);    
    bool IsVector(void);    
    Matrix SubMatrix(long xmin, long xmax, long ymin, long ymax);
    float Mean(void);
    float * operator[](long heightPos);    
    Matrix &operator = (Matrix &m);
    Matrix &operator + (Matrix &m);    
    Matrix &operator - (float alpha);    
    Matrix &operator * (float alpha);
    Matrix &operator / (float alpha);    
    Matrix &operator ^ (float alpha);        
    void Print(void);
    void PrintToFile(char *);
    void WriteToPgm(char *strFileName);
    void WriteToRgb32(unsigned char *destBuf, int nWidth, int nHeight);
    void WriteToRgb32(unsigned char *destBuf, int nWidth, int nHeight, char color);
    // Sebastian added
    unsigned char *getCharArray();
    
    friend void Meshgrid(Matrix &fx, long Xmin, long Xmax, Matrix &fy, long Ymin, long Ymax);
    friend Matrix log(Matrix &m, float alpha);
    friend Matrix exp(Matrix &m);
    friend Matrix fabs(Matrix &m);
    friend Matrix anglediff(Matrix &m);
    friend void cart2pol( Matrix &fx, Matrix &fy, Matrix &theta, Matrix &rho );
    friend Matrix times(Matrix &m1, Matrix &m2);
    friend Matrix fftshift(Matrix &m);
    friend Matrix fft2(Matrix &m, int isign);
    friend Matrix Complex(Matrix &m);
    friend Matrix Real(Matrix &m);
    friend Matrix WhitenFrame(Matrix &m);
        
    // Other member functions
    Matrix(float *arrAddress ,long col);
    Matrix(float *arrAddress ,long arrWidth ,long arrHeight);
    Matrix SubMatrix(long offset);
    float Arg(void);
    Matrix T(void);
    Matrix operator * ( Matrix &m1 );
    void Normalize();
};

#endif


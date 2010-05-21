// Matrix.cpp
//

#include <cstdlib>
#include <iostream>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "CUFFT.h"

#include "Matrix.h"

using namespace std;

// Matrix

Matrix::Matrix()
{
    this->p = new float[1];
    this->Col = 1;
    this->Row = 1;
    this->bComplex = FALSE;
}

Matrix::~Matrix()
{
    delete []p;
}


// Matrix memeber function

long Matrix::GetRows(void)
{
    return this->Row;
}

long Matrix::GetCols(void)
{
    return this->Col;
}

Matrix::Matrix(long n)
{
    this->bComplex = FALSE;
    this->Row = this->Col = n;
    this->p = new float[n * n];
    for(long i = 0 ; i < n ; i++)
        for(long j = 0 ;j < n; j++)
            if(i == j) *(p + n * i + j) = 1;
            else *(p + n * i + j) = 0;
}

Matrix::Matrix(long row ,long col)
{
    this->bComplex = FALSE;
    this->Row = row;
    this->Col = col;
    this->p    = new float[row * col];
    
    memset(this->p, 0, (row * col) * sizeof(float));
}

Matrix::Matrix(unsigned char *image, long imgHeight, long imgWidth)
{    
    // The data is stored in image by columns, so need to convert it to matrix by rows
    this->bComplex = FALSE;
    this->Row = imgHeight;
    this->Col = imgWidth;
    this->p = new float[(this->Row) * (this->Col)];
    for(long i = 0 ; i < this->Row ; i++)
        for(long j = 0 ; j < this->Col ; j ++)
            *(p + this->Col * i + j) = (float)(*(image + imgHeight * j + i));
                    
    return;
}

Matrix::Matrix(const Matrix &m)
{
    //cout<<"Enter copy constructor"<<endl;
    this->bComplex = m.bComplex;
    this->Col = m.Col;
    this->Row = m.Row;
    p = new float[this->Col * this->Row];

    memcpy(this->p, m.p, sizeof(float) * m.Row * m.Col);
}

bool Matrix::IsVector(void)
{
    return !(this->Col == 1 && this->Row ==1);
}

Matrix Matrix::SubMatrix(long xmin, long xmax, long ymin, long ymax)
{
    //cout<<xmin<<xmax<<ymin<<ymax<<this->Row<<this->Col<<endl;
    assert((xmin < this->Col) && (ymin < this->Row) && (xmax < this->Col) && (ymax < this->Row));
    Matrix ret(abs(ymax - ymin) + 1, abs(xmax - xmin) + 1);
    
    for(long  i = 0 ; i < ret.Row ; i++)
        for(long j = 0 ; j < ret.Col ; j ++)
        {
            assert((i + ymin < this->Row) && (j + xmin < this->Col));
            *(ret.p + ret.Col * i + j) = *(this->p + this->Col * (i + ymin) + (j+xmin));
        }
    
    return ret;    
}

float Matrix::Mean(void)
{
    float dSum = 0;
    for(long i = 0 ; i < this->Row ; i++)
    {
        for(long j = 0 ; j < this->Col ; j++)
        {
            dSum += fabs(*(this->p + this->Col * i + j));
        }
    }
    
    return (dSum / (this->Col * this->Row));
}

float * Matrix::operator[](long heightPos)
{
    assert(heightPos >= 0 && heightPos < this->Row);
    return this->p + this->Col * heightPos;
    return NULL;
}

Matrix &Matrix::operator=(Matrix &m)
{
    if(&m == this)return *this;
    this->Row = m.Row;
    this->Col = m.Col;
    this->bComplex = m.bComplex;
    delete []this->p;
    p = new float[this->Row * this->Col];
    
    memcpy(this->p, m.p, sizeof(float) * m.Row * m.Col);
    
    return *this;
}

Matrix &Matrix::operator +(Matrix &m)
{
    assert( (m.Col == this->Col) && (m.Row == this->Row) ); 
    for(int i = 0 ; i < this->Row ; i++)
        for(int j = 0 ; j < this->Col ; j++)
            *(this->p + this->Row * j + i) += *(m.p + m.Row * j + i);   
    return (*this);
}


Matrix &Matrix::operator -(float alpha)
{
    for(int i = 0 ; i < this->Row ; i++)
        for(int j = 0 ; j < this->Col ; j++)
            *(this->p + this->Row * j + i) -= alpha;

    return (*this);
}

Matrix &Matrix::operator *(float alpha)
{
    for(int i = 0 ; i < this->Row ; i++)
        for(int j = 0 ; j < this->Col ; j++)
            *(this->p + this->Row * j + i) = *(this->p + this->Row * j + i) * alpha;

    return (*this);
}

Matrix &Matrix::operator /(float alpha)
{
    for(int i = 0 ; i < this->Row ; i++)
        for(int j = 0 ; j < this->Col ; j++)
            *(this->p + this->Row * j + i) = *(this->p + this->Row * j + i) / alpha;

    return (*this);
}

Matrix &Matrix::operator ^ (float alpha)
{
    for(int i = 0 ; i < this->Row ; i++)
        for(int j = 0 ; j < this->Col ; j++)
            *(this->p + this->Row * j + i) = pow(*(this->p + this->Row * j + i), alpha);

    return (*this);
}

void Matrix::Print(void)
{   
    static int countFilter = 1;
    cout<<"\n\n  Filter: "<<countFilter<<endl;
    countFilter++;
 
//   cout << "Size: " << Row << " x " << Col << endl;
    if (this->bComplex == TRUE)
    	printf("It is a complex matrix.\n");
        
    for(long i = 0 ; i < this->Row ; i ++)
    {
        for(long j = 0 ; j < this->Col ; j++)
        {
            printf("%f  " ,*(this->p + this->Col * i + j));
        }
        printf("\n");
    }
}


void Matrix::PrintToFile(char * fileName)
{
    FILE * outFile = fopen(fileName, "w");

    for(long i = 0 ; i < this->Row ; i ++)
    {
        for(long j = 0 ; j < this->Col ; j++)
        {
            fprintf(outFile, "%f  " ,*(this->p + this->Col * i + j));
        }
        fprintf(outFile, "\n");
    }
    fclose(outFile);
}

void Matrix::WriteToPgm(char *strFileName)
{
   //FILE *out, *out2;
   FILE *out;
   //char fname2[64];

   //strcpy(fname2, strFileName);
   //strcat(fname2, ".real");
   out = fopen( strFileName, "wb" );
   //out2 = fopen( fname2, "wb" );
   assert( out );
   fprintf( out, "P2\n" ); 
   fprintf( out, "%ld %ld\n",this->Col, this->Row ); 
   fprintf( out, "%d\n",255 ); 
   int value;
   //float value2;
   
   for (int i = 0; i < this->Row; i++ )
       for (int j = 0; j < this->Col; j++ )       
       {
           //value2 = (*(this->p + i * this->Col + j));
           value = (int)(*(this->p + i * this->Col + j));
	   value = (int)value;
           value = (unsigned int)round(value - 1);
           if (value < 0)
              value = 0;
           	   
           if ((i == this->Row - 1) && ( j == this->Col - 1)){
	       fprintf(out, "%d", value);
//	       fprintf(out2, "%0.4f", value2);
	   }else if (j == this->Col - 1){
	       fprintf(out, "%d\n", value);
//	       fprintf(out2, "%0.4f\n", value2);
	   }else{
               fprintf(out, "%d ", value);
//	       fprintf(out2, "%0.4f ", value2);
	   }
       }   
           
   fclose(out);  
   //fclose(out2);  
}

void Matrix::WriteToRgb32(unsigned char *destBuf, int nWidth, int nHeight)
{
   assert((this->Row <= nHeight) && (this->Col <= nWidth));
   int value;
   unsigned char *pb32 = NULL;

   for (int i = 0; i < this->Row; i++ )
       for (int j = 0; j < this->Col; j++ )
       {
           value = (int)(*(this->p + i * this->Col + j));
           value = (unsigned int)round(value - 1);
           if (value < 0)
               value = 0;
           pb32 = destBuf + i * nWidth * 4 + 4 * j;
           *pb32 = (unsigned char) value;
           *(pb32 + 1) = *pb32;
           *(pb32 + 2) = *pb32;
           *(pb32 + 3) = 255;
       }
}

void Matrix::WriteToRgb32(unsigned char *destBuf, int nWidth, int nHeight, char color)
{
   assert((this->Row <= nHeight) && (this->Col <= nWidth));
   int value;
   unsigned char *pb32 = NULL;

   for (int i = 0; i < this->Row; i++ )
       for (int j = 0; j < this->Col; j++ )
       {
           value = (int)(*(this->p + i * this->Col + j));
           value = (unsigned int)round(value - 1);
           if (value < 0) value = 0;
           pb32 = destBuf + i * nWidth * 4 + 4 * j;
           switch(color){
             case 0: *pb32 = (unsigned char) value;
                     *(pb32 + 1) = *pb32;
                     *(pb32 + 2) = *pb32;
                     *(pb32 + 3) = 255;
                     break;
             case 1: *pb32 = (unsigned char) value;
                     *(pb32 + 1) = (unsigned char) value;
                     *(pb32 + 2) = 0;
                     *(pb32 + 3) = 255;
                     break;
             case 2: *pb32 = 0;
                     *(pb32 + 1) = (unsigned char) value;
                     *(pb32 + 2) = 0;
                     *(pb32 + 3) = 255;
                     break;
             case 3: *pb32 = 0;
                     *(pb32 + 1) = 0; 
                     *(pb32 + 2) = (unsigned char) value;
                     *(pb32 + 3) = 255;
                     break;
           }
       }
}

// Sebastian Added
unsigned char *Matrix::getCharArray()
{
   unsigned char *retval = new unsigned char[Row * Col];

   for(int i = 0; i < (Row * Col); i++)
       retval[i] = (unsigned char) abs((int)p[i]);

   return retval;
}

void Meshgrid(Matrix &fx, long Xmin, long Xmax, Matrix &fy, long Ymin, long Ymax)
{
    long lRow = (Ymax - Ymin) + 1;
    long lCol = (Xmax - Xmin) + 1;
    
    for(long  i = 0 ; i < lRow ; i++)
        for(long j = 0 ; j < lCol ; j ++)
        {
            assert((Xmin+j) <= Xmax);
            assert((Ymin+i) <= Ymax);
            *(fx.p + fx.Col * i + j) = Xmin + j;
            *(fy.p + fy.Col * i + j) = Ymin + i;
        }    
}

Matrix log(Matrix &m, float alpha)
{
    Matrix ret(m);
    
    assert( (ret.Col == m.Col) && (ret.Row == m.Row) );
    
    for(long  i = 0 ; i < ret.Row ; i++)
        for(long j = 0 ; j < ret.Col ; j ++)
            *(ret.p + ret.Col * i + j) = log(*(m.p + m.Col * i + j)) / log(alpha);
    
    return ret;
}

Matrix exp(Matrix &m)
{
    Matrix ret(m);
    
    assert( (ret.Col == m.Col) && (ret.Row == m.Row) );
    
    for(long  i = 0 ; i < ret.Row ; i++)
        for(long j = 0 ; j < ret.Col ; j ++)
            *(ret.p + ret.Col * i + j) = exp(*(m.p + m.Col * i + j));
    
    return ret;    
}

Matrix fabs(Matrix &m)
{
    Matrix ret(m);
    
    assert( (ret.Col == m.Col) && (ret.Row == m.Row) );
    
    for(long  i = 0 ; i < ret.Row ; i++)
        for(long j = 0 ; j < ret.Col ; j ++)
            *(ret.p + ret.Col * i + j) = fabs(*(m.p + m.Col * i + j));
    
    return ret;
}

Matrix anglediff(Matrix &m)
{
    Matrix ret(m);
    
    assert( (ret.Col == m.Col) && (ret.Row == m.Row) );
    
    for(long  i = 0 ; i < ret.Row ; i++)
        for(long j = 0 ; j < ret.Col ; j ++)
        {
            if ( *(m.p + m.Col * i + j) > pi )
                *(ret.p + ret.Col * i + j) = 2 * pi - *(m.p + m.Col * i + j);
        }
    
    return ret;
}

void cart2pol( Matrix &fx, Matrix &fy, Matrix &theta, Matrix &rho )
{
    assert( (fx.Col == fy.Col) && (fx.Row == fy.Row) );
    assert( (fx.Col == theta.Col) && (fx.Row == theta.Row) );
    assert( (fx.Col == rho.Col) && (fx.Row == rho.Row) );
    
    for(long  i = 0 ; i < fx.Row ; i++)
        for(long j = 0 ; j < fx.Col ; j ++)
        {
            *(rho.p + rho.Col * i + j) = sqrt((*(fx.p + fx.Col * i + j)) * (*(fx.p + fx.Col * i + j))  
                                                + (*(fy.p + fy.Col * i + j)) * (*(fy.p + fy.Col * i + j)));
            *(theta.p + theta.Col * i + j) = atan2( *(fy.p + fy.Col * i + j), *(fx.p + fx.Col * i + j) );        
        }
    return;
}

Matrix times(Matrix &m1, Matrix &m2)
{  
    assert( (!m1.bComplex) || (!m2.bComplex));
    assert( ((m1.Col == m2.Col) || (2 * m1.Col == m2.Col) || (m1.Col == 2 * m2.Col)) && (m1.Row == m2.Row) );
    Matrix ret;
    Matrix real;
    Matrix complex;
    if (m1.Col == 2 * m2.Col)
    {
    	real = m2;
    	complex = m1;
    }
    else
    {
    	real = m1;
    	complex = m2;
	}
	
	ret = complex;
	    
    if (ret.bComplex == FALSE)
    {
    	for(long  i = 0 ; i < ret.Row ; i++)
       	    for(long j = 0 ; j < ret.Col ; j ++)
            {
                *(ret.p + ret.Col * i + j) = (*(m1.p + m1.Col * i + j)) * (*(m2.p + m2.Col * i + j));
            }
    }
    else
    {
    	for(long  i = 0 ; i < ret.Row ; i++)
            for(long j = 0 ; j < ret.Col ; j ++)
            {
            	*(ret.p + ret.Col * i + j) = (*(complex.p + complex.Col * i + j)) * (*(real.p + real.Col * i + j / 2));
            }
    }
    return ret;
}

Matrix fftshift(Matrix &m)
{
    Matrix ret(m);
    int nShift = m.Col / 2;
    if (m.bComplex == FALSE)
    {
    	assert(( m.Col == m.Row ) && (m.Col % 2 == 0));
    	    
        for(long  i = 0 ; i < ret.Row ; i++)
       	    for(long j = 0 ; j < ret.Col ; j ++)
            {
            	if ( i < nShift && j < nShift ) 
            	{
                    assert((i + nShift < ret.Row) && (j + nShift < ret.Col));
                    *(ret.p + ret.Col * i + j) = *(m.p + m.Col * (i + nShift) + (j + nShift));
            	}
            	else if ( i < nShift && j >= nShift)
            	{
                    assert((i + nShift < ret.Row) && (j - nShift >= 0));
                    *(ret.p + ret.Col * i + j) = *(m.p + m.Col * (i + nShift) + (j - nShift));
            	}
            	else if ( i >= nShift && j < nShift)
            	{
                    assert((i - nShift >= 0 ) && (j + nShift < ret.Col ));
                    *(ret.p + ret.Col * i + j) = *(m.p + m.Col * (i - nShift) + (j + nShift));
            	}
            	else
            	{
                    assert((i - nShift >=0) && (j - nShift >= 0));
                    *(ret.p + ret.Col * i + j) = *(m.p + m.Col * (i - nShift) + (j - nShift));
            	}
            }                                       
    }
    else
    {
    	ret.bComplex = TRUE;
    	assert(( m.Col == 2 * m.Row ) && (m.Col % 4 == 0));
    	nShift /= 2;
    
    	for(long  i = 0 ; i < ret.Row ; i++)
            for(long j = 0 ; j < ret.Col ; j += 2)
            {
            	if ( i < nShift && j < 2 * nShift ) 
            	{
               	    assert((i + nShift < ret.Row) && (j + 2 * nShift + 1< ret.Col));
                    *(ret.p + ret.Col * i + j) = *(m.p + m.Col * (i + nShift) + (j + 2 * nShift));
                    *(ret.p + ret.Col * i + j + 1) = *(m.p + m.Col * (i + nShift) + (j + 1 + 2 * nShift));
            	}
            	else if ( i < nShift && j >= 2 * nShift)
            	{
                    assert((i + nShift < ret.Row) && (j - 2 * nShift >= 0));
                    *(ret.p + ret.Col * i + j) = *(m.p + m.Col * (i + nShift) + (j - 2 * nShift));
                    *(ret.p + ret.Col * i + j + 1) = *(m.p + m.Col * (i + nShift) + (j + 1 - 2 * nShift));
            	}
            	else if ( i >= nShift && j < 2* nShift)
            	{
                    assert((i - nShift >= 0 ) && (j + 1 + 2 * nShift < ret.Col ));
                    *(ret.p + ret.Col * i + j) = *(m.p + m.Col * (i - nShift) + (j + 2 * nShift));
                    *(ret.p + ret.Col * i + j + 1) = *(m.p + m.Col * (i - nShift) + (j + 1 + 2 * nShift));
            	}
            	else
            	{
                    assert((i - nShift >=0) && (j - 2 * nShift >= 0));
                    *(ret.p + ret.Col * i + j) = *(m.p + m.Col * (i - nShift) + (j - 2 * nShift));
                    *(ret.p + ret.Col * i + j + 1) = *(m.p + m.Col * (i - nShift) + (j + 1 - 2 *nShift));
            	}
            }                                       
    }
        
    return ret;
}

Matrix fft2(Matrix &m, int isign)
{
    assert(m.bComplex == TRUE);
    int ndim = 2;	
    long row = m.GetRows(); 
    long col = m.GetCols();
	
    Matrix ret(m);
    unsigned long nn[3];
		
    nn[1] = row;
    nn[2] = col / 2;
		
#if 1
    float *pdata = new float[row * col + 1];
    memcpy( pdata + 1, m.p, sizeof(float) * row * col);
    fourn( pdata, nn, ndim, isign );
    memcpy( ret.p, pdata + 1, sizeof(float) * row * col);
    delete[] pdata;
#else
	float* pdata = new float[row * col];
	memcpy(pdata, m.p, sizeof(float) * row * col);
	CUFFT::fft(pdata, col / 2, row, isign == 1);
	CUFFT::fft(pdata, col / 2, row, isign != 1);
	memcpy(ret.p, pdata, sizeof(float) * row * col);
	delete [] pdata;
#endif
    ret.bComplex = TRUE;
		
    return ret;	
}

Matrix Complex(Matrix &m)
{   
    if (m.bComplex == TRUE)
    {
    	Matrix ret(m);
    	return ret;
    }
    
    Matrix ret(m.Row, m.Col * 2);
    
    for(long  i = 0 ; i < m.Row ; i++)
        for(long j = 0 ; j < m.Col ; j ++)
        {           
            assert( 2 * j + 1 < ret.Col );
            *(ret.p + ret.Col * i + 2 * j) = *(m.p + m.Col * i + j);
            *(ret.p + ret.Col * i + 2 * j + 1) = 0;
        }
    
    ret.bComplex = TRUE;
    
    return ret;    
}

Matrix Real(Matrix &m)
{	
    assert(m.bComplex == TRUE);
    Matrix ret(m.Row, m.Col / 2);
    
    for(long  i = 0 ; i < ret.Row ; i++)
        for(long j = 0 ; j < ret.Col ; j ++)
        {           
            assert( 2 * j < m.Col );
            *(ret.p + ret.Col * i + j) = *(m.p + m.Col * i + 2 * j);            
        }   
    
    return ret;    	
}

Matrix WhitenFrame(Matrix &m)
{
    int nRow = m.GetRows();
    int nCol = m.GetCols();
    assert(nRow == nCol);
	
    Matrix fx(nRow), fy(nRow);
    Meshgrid(fx, -nRow / 2, nRow / 2 - 1, fy, -nRow / 2, nRow / 2 - 1);
    Matrix theta = fx;
    Matrix rho = fx;
    
    cart2pol( fx,fy,theta,rho );
	
#if 1
    Matrix complex = Complex(m);
    Matrix fftm = fft2(complex, 1);
    Matrix imF = fftshift(fftm);
    Matrix time = times(rho, imF);
    Matrix imW = fftshift(time);
#else
    Matrix complex = Complex(m);
    Matrix fftm = fft2(complex, 1);
    Matrix imF = fftshift(fftm);
    Matrix time = times(rho, imF);
    Matrix imW = fftshift(time);
#endif

    return imW;
}


// Other member
Matrix::Matrix(float *arrAddress ,long arrWidth)
{
    long arrHeight = 1;
    this->p = new float[arrWidth * arrHeight];
    for(long i = 0 ; i < arrHeight ; i++)
        for(long j = 0 ; j < arrWidth; j++)
            *(p + arrWidth * i + j ) = *(arrAddress + arrWidth * i + j);
    this->Col = arrWidth ;
    this->Row = arrHeight;
}
Matrix::Matrix(float *arrAddress,long rows,long cols)
{
    this->p = new float[rows * cols];
    for(long i = 0 ; i < rows ; i++)
        for(long j = 0 ; j < cols ; j ++)
            *(p +cols * i + j) = *(arrAddress + cols * i + j );
    this->Col = cols;
    this->Row = rows;
}

Matrix Matrix::SubMatrix(long offset)
{
    assert(this->Col ==    this->Row && offset <= this->Col && offset >= 0);
    float *t = new float[offset * offset] ;
    for(long i = 0 ; i < offset ; i++)
        for(long j = 0 ; j < offset ; j++)
            *(t + offset * i + j )  = *(p + this->Col * i + j);
    Matrix m(t ,offset ,offset);
    delete []t;
    return m;
}

float Matrix::Arg(void)
{
    assert(this->Row == this->Col);
    float result = 1;
    float k ;
    Matrix m = *this;
    for(long i = 0 ; i < this->Row - 1 ; i++)
    {
        for(long j = i + 1; j < this->Row ; j++)
        {
            k = m[j][i] / m[i][i];
            m[j][i] = 0 ;
            for(long n = i + 1; n < this->Col ; n ++)
            {
                m[j][n] = m[j][n] - k * m[i][n];
            }
        }
    }
    for(long i = 0 ; i < this->Row ; i++)
    {
        for(long j = 0 ; j < this->Col ; j++)
        {
            if(i == j ) result *= m[i][j];
        }
    }
    return result;
}

Matrix Matrix::T(void)
{
    float *t = new float[this->Col * this->Row] ;
    for(long i = 0 ; i< this->Row ;i++)
        for(long j = 0 ; j < this->Col ; j++)
            *(t + this->Row * j + i) = *(this->p + this->Col * i + j);
    Matrix m(t ,this->Col ,this->Row);
    delete []t;
    return m;
}

Matrix Matrix::operator *(Matrix &m1)
{
    if(this->Col == m1.Row)
    {
        Matrix ttt(*this);
        int mr = this->Row;
        int mc = m1.Col;
        Matrix tt(mr ,mc);
        for(long i = 0 ; i < mr ; i++)
        {
            for(long j = 0 ; j < mc; j++)
            {
                for(int ii = 0 ; ii < this->Col; ii++)
                {
                    
                    tt[i][j] += ttt[i][ii] * m1[ii][j];
                }
            }
        }
        return tt;
    }
}

void fourn( float data[], unsigned long nn[], int ndim, int isign )
{
    int idim;
    unsigned long i1, i2, i3, i2rev, i3rev, ip1, ip2,ip3, ifp1, ifp2;
    unsigned long ibit, k1, k2, n, nprev, nrem, ntot;
    float tempi, tempr;
    float theta, wi, wpi, wpr, wr, wtemp;
    
    for (ntot = 1, idim = 1; idim <= ndim; idim ++ )
        ntot *= nn[idim];
    nprev = 1;
    for ( idim = ndim; idim >= 1; idim --)
    {
        n = nn[idim];
        nrem = ntot / (n * nprev);
        ip1 = nprev << 1;
        ip2 = ip1 * n;
        ip3 = ip2 * nrem;
        i2rev = 1;
        for (i2 = 1; i2 <= ip2; i2 += ip1)
        {
            if (i2 < i2rev )
            {   
                for (i1 = i2; i1 <= i2 + ip1 -2; i1 +=2 )
                {
                    for ( i3 = i1; i3 <= ip3; i3 += ip2)
                    {
                        i3rev = i2rev + i3 - i2;
                        SWAP(data[i3], data[i3rev]);
                        SWAP(data[i3 + 1], data[ i3rev + 1 ]);
                    }
                }
            }
            ibit = ip2 >> 1;
            while (ibit >= ip1 && i2rev > ibit)
            {
                i2rev -= ibit;
                ibit >>= 1;
            }
            i2rev += ibit;
        }
        ifp1 = ip1;
        while (ifp1 < ip2)
        {
            ifp2 = ifp1 << 1;
            theta = - isign * 6.28318530717959 / (ifp2 / ip1 );
            wtemp = sin( 0.5 * theta );
            wpr = -2.0 * wtemp * wtemp;
            wpi = sin(theta);
            wr = 1.0;
            wi = 0.0;
            for (i3 = 1; i3 <= ifp1; i3 += ip1)
            {
                for ( i1 = i3; i1 <= i3 + ip1 - 2; i1 += 2)
                {
                    for (i2 = i1; i2 <= ip3; i2 += ifp2)
                    {
                        k1 = i2;
                        k2 = k1 + ifp1;
                        tempr = (float) wr * data[k2] - (float) wi * data[ k2 + 1];
                        tempi = (float) wr * data[k2 + 1] + (float) wi * data[ k2 ];
                        data[k2] = data[k1] - tempr;
                        data[k2 + 1] = data[k1 + 1] - tempi;
                        data[k1] += tempr;
                        data[k1 + 1] += tempi;
                    } 
                }
                wr = (wtemp = wr) * wpr - wi * wpi + wr;
                wi = wi * wpr + wtemp * wpi + wi;
            }
            ifp1 = ifp2;
        }
        nprev *= n;
    }
    
    if (isign == -1)
    {
    	wtemp = 1.0;
    	for (idim = ndim; idim >= 1; idim --)
    	{    		
    		wtemp *= nn[idim];
    	}
    	for (i1 = 1; i1 <= 2 * wtemp; i1 ++)
    	{
    		data[i1] /= wtemp;
    	}
    }
}

void Matrix::Normalize()
{
    float dMin = p[0];
    float dMax = p[0];
    float dSum = 0;
	
    for (uint i = 0; i < Row * Col; i++)
    {
    	if (p[i] < dMin)
    		dMin = p[i];
        if (p[i] > dMax)
        	dMax = p[i];
        dSum += p[i];
    }	
    for (uint i = 0; i < Row * Col; i++)
    {
    	if ((dMax - dMin) == 0)
            p[i] = 0;
    	else
    	    p[i] = (p[i] - dMin) / (dMax - dMin);
    }
}








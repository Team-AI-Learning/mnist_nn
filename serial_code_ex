#include<iostream>
#include"FileReader.hpp"
#include<math.h>
#include<random>

#define DIM0 784
#define DIM1 50
#define DIM2 10
#define ETA 0.01
#define BATCH_SIZE 100

double T[DIM2];
double Z0[DIM0], Z1[DIM1], Z2[DIM2];
double X0[DIM0], X1[DIM1], X2[DIM2];
double W1[DIM1*DIM0], W2[DIM2*DIM1];
double B1[DIM1], B2[DIM2];

double DJDZ0[DIM0], DJDZ1[DIM1], DJDZ2[DIM2];
double DJDX0[DIM0], DJDX1[DIM1], DJDX2[DIM2];
double DJDW1[DIM1*DIM0], DJDW2[DIM2*DIM1];
double DJDB1[DIM1], DJDB2[DIM2];

using namespace std;

void setInput(double v[], double* inputvec, int size_in);
void setAns(double v[], double* ansvec, int size_out);
void setRandomVec(double v[], int size_in);
void setRandomMat(double m[], int size_in, int size_out);
void setZeroVec(double v[] , int size_in);
void setZeroMat(double m[], int size_in, int size_out);
void MatVecMulBias(double u[], double M[], double v[], double b[], int size_in, int size_out);
void Sigmoidize(double u[], double v[], int size_out);
void Softmaxize(double u[], double v[], int size_out);

double getCrossEntropy(double t[], double x[], int size_out);
double getMeanSquare(double t[], double x[], int size_out);

void DiffVecs(double y[], double u[], double v[], int size_out);
void setDJDXpre(double DJDXpre[], double DJDZ[], double W[], int size_in, int size_out);
void setDJDZ(double DJDZ[], double DJDX[], double X[], int size_out);
void setDJDW(double DJDW[], double DJDZ[], double Xpre[], int size_in, int size_out);
void setDJDB(double DJDB[], double DJDZ[], int size_out);
void updateParams(double W[], double B[], double DJDW[], double DJDB[], int size_in, int size_out);

int getMaxVecElemIDX(double v[], int size_out);

int main(void)
{
	ImageReader inputImage("train-images.idx3-ubyte");
	inputImage.read();
	LabelReader inputLabel("train-labels.idx1-ubyte");
	inputLabel.read();
	ImageReader testImage("t10k-images.idx3-ubyte");
	testImage.read();
	LabelReader testLabel("t10k-labels.idx1-ubyte");
	testLabel.read();

	setRandomMat(W1, DIM0, DIM1);
	setRandomMat(W2, DIM1, DIM2);
	setRandomVec(B1, DIM1);
	setRandomVec(B2, DIM2);

	double cost = 0.0;

	for (int i = 0; i < 60000; ++i)
	{
		setInput(X0, inputImage.images[i], DIM0);
		setAns(T, inputLabel.ans[i], DIM2);
		MatVecMulBias(Z1, W1, X0, B1, DIM0, DIM1);
		Sigmoidize(X1, Z1, DIM1);
		MatVecMulBias(Z2, W2, X1, B2, DIM1, DIM2);
		Softmaxize(X2, Z2, DIM2);

		cost += getCrossEntropy(inputLabel.ans[i], X2, DIM2);
//		cost += getMeanSquare(inputLabel.ans[i], X2, DIM2);

		DiffVecs(DJDZ2, X2, T, DIM2);
		setDJDXpre(DJDX1, DJDZ2, W2, DIM1, DIM2);
		setDJDW(DJDW2, DJDZ2, X1, DIM1, DIM2);
		setDJDB(DJDB2, DJDZ2, DIM2);

		setDJDZ(DJDZ1, DJDX1, X1, DIM1);
		//	setDJDXpre(DJDX0, DJDZ1, W1, DIM0, DIM1);
		setDJDW(DJDW1, DJDZ1, X0, DIM0, DIM1);
		setDJDB(DJDB1, DJDZ1, DIM1);

		if ((i + 1) % BATCH_SIZE == 0)
		{
			cout << cost / (double)BATCH_SIZE << endl;
			cost = 0.0;

			updateParams(W2, B2, DJDW2, DJDB2, DIM1, DIM2);
			updateParams(W1, B1, DJDW1, DJDB1, DIM0, DIM1);
			setZeroVec(DJDB2, DIM2);
			setZeroMat(DJDW2, DIM1, DIM2);
			setZeroVec(DJDB1, DIM1);
			setZeroMat(DJDW1, DIM0, DIM1);
		}
	}

	int count = 0;
	int numTest = 1000;

	for (int i = 0; i < numTest; ++i)
	{
		setInput(X0, testImage.images[i], DIM0);
		setAns(T, testLabel.ans[i], DIM2);

		MatVecMulBias(Z1, W1, X0, B1, DIM0, DIM1);
		Sigmoidize(X1, Z1, DIM1);
		MatVecMulBias(Z2, W2, X1, B2, DIM1, DIM2);
		Softmaxize(X2, Z2, DIM2);

		if (T[getMaxVecElemIDX(X2, DIM2)] == 1)
			count++;
	}

	cout << "true predictions: " << count << " (out of " << numTest << ")" << endl;

	return 0;
}

void setInput(double v[], double* inputvec, int size_in)
{
	for (int i = 0; i < size_in; ++i)
		v[i] = inputvec[i];
}

void setAns(double v[], double* ansvec, int size_out)
{
	for (int i = 0; i < size_out; ++i)
		v[i] = ansvec[i];
}

void setRandomVec(double v[], int size_in)
{
	default_random_engine generator;
	double variance = 1.0 / (double)size_in;
	normal_distribution<double> distribution(0.0, variance);

	for (int i = 0; i < size_in; ++i)
		v[i] = distribution(generator);
}

void setRandomMat(double m[], int size_in, int size_out)
{
	default_random_engine generator;
	double variance = 1.0 / (double)size_in;
	normal_distribution<double> distribution(0.0, variance);

	for (int i = 0; i < size_out; ++i)
		for (int j = 0; j < size_in; ++j)
			m[size_in * i + j] = distribution(generator);
}

void setZeroVec(double v[], int size_in)
{
	for (int i = 0; i < size_in; ++i)
		v[i] = 0.0;
}

void setZeroMat(double m[], int size_in, int size_out)
{
	for (int i = 0; i < size_out; ++i)
		for (int j = 0; j < size_in; ++j)
			m[size_in * i + j] = 0.0;
}

void MatVecMulBias(double u[], double M[], double v[], double b[], int size_in, int size_out)
{
	for (int i = 0; i < size_out; ++i)
	{
		u[i] = 0.0;

		for (int j = 0; j < size_in; ++j)
			u[i] += M[size_in * i + j] * v[j];

		u[i] += b[i];
	}
}

void Sigmoidize(double u[], double v[], int size_out)
{
	for (int i = 0; i < size_out; ++i)
		u[i] = 1 / (1 + exp(-v[i]));
}

void Softmaxize(double u[], double v[], int size_out)
{
	double z_max = 0.0;
	for (int i = 0; i < size_out; ++i)
		z_max = v[i] > z_max ? v[i] : z_max;

	double sum = 0.0;
	for (int i = 0; i < size_out; ++i)
		sum += exp(v[i] - z_max);

	for (int i = 0; i < size_out; ++i)
		u[i] = exp(v[i] - z_max) / sum;
}

double getCrossEntropy(double t[], double x[], int size_out)
{
	double ret = 0.0;

	for (int i = 0; i < size_out; i++)
		ret += t[i] * log(x[i]) + (1 - t[i]) * log(1 - x[i]);

	return -ret;
}

double getMeanSquare(double t[], double x[], int size_out)
{
	double ret = 0;

	for (int i = 0; i < size_out; i++)
		ret += 0.5 * (t[i] - x[i]) * (t[i] - x[i]);

	return ret;
}

void DiffVecs(double y[], double u[], double v[], int size_out)
{
	for (int i = 0; i < size_out; i++)
		y[i] = u[i] - v[i];
}

void setDJDXpre(double DJDXpre[], double DJDZ[], double W[], int size_in, int size_out)
{
	for (int j = 0; j < size_in; j++)
	{
		DJDXpre[j] = 0;

		for (int i = 0; i < size_out; i++)
			DJDXpre[j] += DJDZ[i] * W[size_in*i + j];
	}
}

void setDJDZ(double DJDZ[], double DJDX[], double X[], int size_out)
{
	for (int i = 0; i < size_out; i++)
		DJDZ[i] = DJDX[i] * X[i] * (1 - X[i]);
}

void setDJDW(double DJDW[], double DJDZ[], double Xpre[], int size_in, int size_out)
{
	for (int i = 0; i < size_out; i++)
		for (int j = 0; j < size_in; j++)
			DJDW[i*size_in + j] += DJDZ[i] * Xpre[j];
}

void setDJDB(double DJDB[], double DJDZ[], int size_out)
{
	for (int i = 0; i < size_out; i++)
		DJDB[i] += DJDZ[i];
}

void updateParams(double W[], double B[], double DJDW[], double DJDB[], int size_in, int size_out)
{
	for (int i = 0; i < size_out; i++)
	{
		B[i] -= ETA * DJDB[i];

		for (int j = 0; j < size_in; j++)
			W[i*size_in + j] -= ETA * DJDW[i*size_in + j];
	}
}

int getMaxVecElemIDX(double v[], int size_out)
{
	double max = -10000.0;
	int ret = 0;

	for (int i = 0; i < size_out; ++i)
		if (v[i] > max)
		{
			ret = i;
			max = v[i];
		}

	return ret;
}

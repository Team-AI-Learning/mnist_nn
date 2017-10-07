#ifndef __FILTER__
#define __FILTER__

#include<iostream>
#include<math.h>
#include<random>
#include<time.h>
#include<assert.h>
#include<stdlib.h>
using namespace std;

// 정의 : 다차원 배열을 쉽게 다루기 위한 클래스
#define UINT unsigned int
class Tensor
{
public:
	bool allocated;
	union
	{
		struct
		{
			UINT mRow; // i
			UINT mCol; // j
			UINT mFilterRow; // k
			UINT mFilterCol; // l
		};
		struct
		{
			UINT I;
			UINT J;
			UINT K;
			UINT L;
		};
	};

	double ****arr;
	unsigned int size;
	
public:
	// [row][col][filter row][filter col]
	explicit Tensor(UINT _i, UINT _j = 1, UINT _k = 1, UINT _l = 1)
		: mRow(_i), mCol(_j), mFilterRow(_k), mFilterCol(_l)
	{
		arr = alloc(I, J, K, L);
		setRandom();
		size = I*J*K*L;
		allocated = true;
	}
	// 복사생성자는?
	explicit Tensor()
	{
		arr = 0; 
		size = 0;
		allocated = false;
	}

	virtual ~Tensor()
	{
		if (allocated)
		{
			delete[] arr[0][0][0];
			delete[] arr[0][0];
			delete[] arr[0];
			delete[] arr;
			arr = 0;
		}
	}

	double***& operator[](unsigned int x)
	{
		return arr[x];
	}

	double& array(unsigned int i, unsigned int j = 0, unsigned int k = 0, unsigned int l = 0)
	{
		return arr[i][j][k][l];
	}
	// operator=
public:

	void setRandom()
	{
		default_random_engine generator;
		double variance = 1.0 / ((double)J*K*L);
		assert(J != 0 && K != 0 && L != 0);
		normal_distribution<double> distribution(0.0, variance);

		for (int i = 0; i < I; i++) for (int j = 0; j < J; j++)
		for (int k = 0; k < K; k++) for (int l = 0; l < L; l++)
		{
			arr[i][j][k][l] = distribution(generator);
		}
	}
protected:

	double ****alloc(int max_i, int max_j, int max_k, int max_l) {
		double *_l = new double[max_i*max_j*max_k*max_l]; // 실제 저장할 메모리
		double **_k = new double*[max_i*max_j*max_k]; // 
		double ***_j = new double**[max_i*max_j];
		double ****_i = new double***[max_i];

		for (int i = 0; i < max_i; i++) {
			_i[i] = _j;
			_j += max_j;
			for (int j = 0; j < max_j; j++) {
				_i[i][j] = _k;
				_k += max_k;
				for (int k = 0; k < max_k; k++) {
					_i[i][j][k] = _l;
					_l += max_l;
				}
			}
		}
		return _i;
	}

};

#endif

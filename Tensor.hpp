#ifndef __TENSOR__
#define __TENSOR__
#include<iostream>
#include<math.h>
#include<random>
#include<time.h>
#include<stdlib.h>

#include"common.h"
using namespace std;

// TODO:
// add dimension or Format info var
// operator= for shallow/deep copying

class Tensor
{
public:
	bool allocated;
	union
	{
		struct
		{
			UINT I;
			UINT J;
			UINT K;
			UINT L;
			UINT M;
		};
	};

	double *****arr;
	UINT size;
	
public:
	// [row][col][filter row][filter col]
	explicit Tensor(UINT _i, UINT _j = 1, UINT _k = 1, UINT _l = 1, UINT _m = 1)
		: I(_i), J(_j), K(_k), L(_l), M(_m)
	{
		arr = alloc(I, J, K, L, M);
		setRandom();
		size = I*J*K*L*M;
		allocated = true;
	}

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
			delete[] arr[0][0][0][0];
			delete[] arr[0][0][0];
			delete[] arr[0][0];
			delete[] arr[0];
			delete[] arr;
			arr = 0;
		}
	}

	double****& operator[](UINT x)
	{
		return arr[x];
	}

	double& array(UINT i, UINT j = 0, UINT k = 0, UINT l = 0, UINT m = 0)
	{
		return arr[i][j][k][l][m];
	}
public:

	void setRandom()
	{
		default_random_engine generator;
		double variance = 1.0 / ((double)J*K*L*M);
		
		normal_distribution<double> distribution(0.0, variance);

		FOR5D(i, j, k, l, m, I, J, K, L, M)
		{
			arr[i][j][k][l][m] = distribution(generator);
			if (arr[i][j][k][l][m] >= 1.0)
			{
				cout << "error\n";
				getchar();
			}
				
		}
			
	}

protected:
	double *****alloc(int max_i, int max_j, int max_k, int max_l, int max_m) {

		double *_m = new double[max_i*max_j*max_k*max_l*max_m];
		double **_l = new double*[max_i*max_j*max_k*max_l];
		double ***_k = new double**[max_i*max_j*max_k];
		double ****_j = new double***[max_i*max_j];
		double *****_i = new double****[max_i];
		for (int i = 0; i < max_i; i++) {
			_i[i] = _j;
			_j += max_j;
			for (int j = 0; j < max_j; j++) {
				_i[i][j] = _k;
				_k += max_k;
				for (int k = 0; k < max_k; k++) {
					_i[i][j][k] = _l;
					_l += max_l;
					for (int l = 0; l < max_l; l++) {
						_i[i][j][k][l] = _m;
						_m += max_m;
					}
				}
			}
		}
		return _i;
	}
};

#endif
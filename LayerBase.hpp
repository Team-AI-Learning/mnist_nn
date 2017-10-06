#ifndef __LAYER_BASE__
#define __LAYER_BASE__

#define max(a, b) (a > b) ? (a) : (b);
#include<iostream>
#include<math.h>
#include<random>
#include<time.h>
#include "Tensor.hpp"
using namespace std;

// activation functions
static void sigmoid(double z, double& out_x)
{
	out_x = 1 / (1 + exp(-z));
}

static void softmax(Tensor& const z, Tensor& out_x, int size)
{
	double z_max = 0;
	for (int i = 0; i < size; i++)
		z_max = z.array(i) > z_max ? z.array(i) : z_max;

	double sum = 0;
	for (int i = 0; i < size; i++)
		sum += exp(z.array(i) - z_max);

	for (int i = 0; i < size; i++)
		out_x.array(i) = exp(z.array(i) - z_max) / sum;
}

// J - loss functions
static double mean_square(Tensor& t, Tensor& x, int idx, int size)
{
	double sum = 0;
	for (int i = 0; i < size; i++)
		sum += 0.5 * (t.array(idx, i) - x.array(i)) * (t.array(idx, i) - x.array(i));

	return sum;
}


struct DJ // neuron data
{
	double *dx;
	double *dz;
	double *dw;
	double *db;
	
	DJ(int input_num_neurons, int num_neurons)
	{
		dx = new double[num_neurons];
		dz = new double[num_neurons];
		dw = new double[num_neurons*input_num_neurons];
		db = new double[num_neurons];
	}
	DJ() {}

	~DJ()
	{
		delete[] dx;
		delete[] dz;
		delete[] dw;
		delete[] db;
	}
};

static double learning_rate = 0.01;
class LayerBase
{
public:
	enum Activation {Identity, ReLU, Sigmoid, Softmax};
public:
	
	int numNeurons;
	bool isInputLayer;

protected:
	LayerBase* inputLayer;

	DJ dJ;
	Tensor *_x;
	Tensor z;
	Tensor w;
	Tensor b;
public:
	explicit LayerBase(LayerBase* i_inputLayer, int num_neurons)
		: inputLayer(i_inputLayer), numNeurons(num_neurons), isInputLayer(false),
		dJ(i_inputLayer->numNeurons, num_neurons),
		z(numNeurons),
		w(numNeurons, inputLayer->numNeurons),
		b(numNeurons)
	{
		_x = new Tensor(numNeurons);
	}
	explicit LayerBase(int num_neurons)
		: numNeurons(num_neurons)
	{
		isInputLayer = true;
		_x = new Tensor(numNeurons);
	}

	~LayerBase()
	{
		printf("~ %x\n",this);
		if (_x != 0)
		{
			delete _x;
			_x = 0;
		}
	}

	void forwardPropagation(Activation act)
	{
		LayerBase& in = (*inputLayer);
		// Calculate z values
		for (int i = 0; i < numNeurons; i++)
			z.array(i) = 0;

		for (int i = 0; i < numNeurons; i++)
		{
			for (int j = 0; j < in.numNeurons; j++)
				z.array(i) += w.array(i,j) * in._x->array(j);

			z.array(i) += b.array(i);
		}

		// Update x values
		switch (act)
		{
		case Identity:
			for (int i = 0; i < numNeurons; i++)
				_x->array(i) = z.array(i);
			break;
		case ReLU:
			for (int i = 0; i < numNeurons; i++)
				_x->array(i) = max(z.array(i), 0);
			break;
		case Sigmoid:
			for (int i = 0; i < numNeurons; i++)
				sigmoid(z.array(i), _x->array(i));
			break;
		case Softmax:
			softmax(z, *_x, numNeurons);
			break;
		}
	}

	void backPropagation(Activation act, int idx = 0, Tensor *ans = 0)
	{
		LayerBase& in = (*inputLayer);
		switch (act)
		{
		case Identity:
			for (int i = 0; i < numNeurons; i++)
				dJ.dz[i] = dJ.dx[i];
			break;
		case ReLU:
			for (int i = 0; i < numNeurons; i++)
				dJ.dz[i] = dJ.dx[i] * ((_x->array(i) > 0) ? 1 : 0);
			break;
		case Sigmoid:
			for (int i = 0; i < numNeurons; i++)
				dJ.dz[i] = dJ.dx[i] * _x->array(i) * (1 - _x->array(i));
			break;
		case Softmax: // Cross entropy
			for (int i = 0; i < numNeurons; i++)
				dJ.dz[i] = _x->array(i) - ans->array(idx,i);
			break;
		}

		if (!in.isInputLayer)
		{
			for (int j = 0; j < in.numNeurons; j++)
			{
				in.dJ.dx[j] = 0;
				for (int i = 0; i < numNeurons; i++)
					in.dJ.dx[j] += dJ.dz[i] * w.array(i,j);
			}
		}
		updateWeightBias();
	}

	void updateWeightBias()
	{
		LayerBase& in = (*inputLayer);
		for (int i = 0; i < numNeurons; i++)
		{
			dJ.db[i] = dJ.dz[i];
			b.array(i) -= learning_rate * dJ.db[i];
			for (int j = 0; j < in.numNeurons; j++)
			{
				dJ.dw[i * in.numNeurons + j] = dJ.dz[i] * in._x->array(j);
				w.array(i,j) -= learning_rate * dJ.dw[i * in.numNeurons + j];
			}
		}
	}

	Tensor& getOutput()	{ return *_x; }

	// InputLayer
	void updateInput(Tensor& input, int idx)
	{
		int i = 0;
		for (int j = 0; j < input.J; j++)
		{
			for (int k = 0; k < input.K; k++)
			{
				_x->array(i) = input.array(idx, j, k);
				i++;
			}
		}
	}
};

#endif
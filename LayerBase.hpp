#ifndef __LAYER_BASE__
#define __LAYER_BASE__

#define max(a, b) (a > b) ? (a) : (b);
#define UINT unsigned int
#include<iostream>
#include<math.h>
#include<random>
#include<time.h>
#include "Tensor.hpp"
using namespace std;

static double learning_rate = 0.01;
static UINT minibatch_size = 1;

struct LayerInfo
{
	UINT numNeurons;
	UINT x_row;
	UINT x_col;
	UINT w_row;
	UINT w_col;
	// stride
};

// activation functions
static void sigmoid(double z, double& out_x)
{
	out_x = 1 / (1 + exp(-z));
}

static void softmax(Tensor& const z, Tensor& out_x, LayerInfo info)
{
	for (int b = 0; b < minibatch_size; b++)
	{
		double z_max = 0;
		for (int i = 0; i < info.numNeurons; i++)
			for (int m = 0; m < info.x_row; m++)
				for (int n = 0; n < info.x_col; n++)
					z_max = z.array(b, i, m, n) > z_max ? z.array(b, i, m, n) : z_max;

		double sum = 0;
		for (int i = 0; i < info.numNeurons; i++)
			for (int m = 0; m < info.x_row; m++)
				for (int n = 0; n < info.x_col; n++)
					sum += exp(z.array(b, i, m, n) - z_max);

		for (int i = 0; i < info.numNeurons; i++)
			for (int m = 0; m < info.x_row; m++)
				for (int n = 0; n < info.x_col; n++)
					out_x.array(b, i, m, n) = exp(z.array(b, i, m, n) - z_max) / sum;
	}
}

// J - loss functions
static double mean_square(Tensor& t, Tensor& x, int idx, LayerInfo info)
{
	double sum = 0;

	for (int i = 0; i < info.numNeurons; i++)
		for (int m = 0; m < info.x_row; m++)
			for (int n = 0; n < info.x_col; n++)
				sum += 0.5 * (t.array(idx, i) - x.array(idx, i, m, n)) * (t.array(idx, i) - x.array(idx, i, m, n));

	/*
	for (int i = 0; i < size; i++)
		sum += 0.5 * (t.array(idx, i) - x.array(i)) * (t.array(idx, i) - x.array(i));
	*/
	return sum;
}


class LayerBase
{
public:
	enum Activation {Identity, ReLU, Sigmoid, Softmax};
public:
	bool isInputLayer;
	LayerInfo info;
protected:
	struct DJ // neuron data
	{
		Tensor *dx;
		Tensor *dz;
		Tensor *dw;
		Tensor *db;
		bool allocated;

		DJ(LayerBase& inputLayer, LayerInfo info)
			: allocated(true)
		{
			dx = new Tensor(minibatch_size, info.numNeurons, info.x_row, info.x_col);
			dz = new Tensor(minibatch_size, info.numNeurons, info.x_row, info.x_col);
			dw = new Tensor(info.numNeurons, inputLayer.info.numNeurons, info.w_row, info.w_col);
			db = new Tensor(minibatch_size, info.numNeurons, info.x_row, info.x_col);
		}
		DJ()
			: allocated(false)
		{}
		~DJ()
		{
			if (!allocated) return;
			delete[] dx;
			delete[] dz;
			delete[] dw;
			delete[] db;
			dx = 0;
			dz = 0;
			dw = 0;
			db = 0;
		}
	};
protected:
	LayerBase* inputLayer;

	DJ dJ;
	Tensor *_x;
	Tensor z;
	Tensor w;
	Tensor b;
public:
	explicit LayerBase(LayerBase* i_inputLayer, LayerInfo _info)
		: inputLayer(i_inputLayer), info(_info), isInputLayer(false),
		dJ(*i_inputLayer, info),
		z(minibatch_size, info.numNeurons, info.x_row, info.x_col),
		w(info.numNeurons, inputLayer->info.numNeurons, info.w_row, info.w_col),
		b(minibatch_size, info.numNeurons, info.x_row, info.x_col)
	{
		_x = new Tensor(minibatch_size, info.numNeurons, info.x_row, info.x_col);
	}
	explicit LayerBase(LayerInfo _info)
		: info(_info)
	{
		isInputLayer = true;
		_x = new Tensor(minibatch_size, info.numNeurons, info.x_row, info.x_col);
	}

	~LayerBase()
	{
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
		for (int b = 0; b < minibatch_size; b++)
			for (int i = 0; i < info.numNeurons; i++)
				for (int m = 0; m < info.x_row; m++)
					for (int n = 0; n < info.x_col; n++)
						z.array(b, i, m, n) = 0;

		for (int b = 0; b < minibatch_size; b++)
			for (int i = 0; i < info.numNeurons; i++)
				for (int m = 0; m < info.x_row; m++)
					for (int n = 0; n < info.x_col; n++)
						for (int j = 0; j < inputLayer->info.numNeurons; j++)
							for (int k = 0; k < info.w_row; k++)
								for (int l = 0; l < info.w_col; l++)
									z.array(b, i, m, n) += w.array(i, j, k, l) * in._x->array(b, j, k, l);

		// Update x values
		switch (act)
		{
		case Identity:
			for (int b = 0; b < minibatch_size; b++)
				for (int i = 0; i < info.numNeurons; i++)
					for (int m = 0; m < info.x_row; m++)
						for (int n = 0; n < info.x_col; n++)
							_x->array(b, i, m, n) = z.array(b, i, m, n);
			break;
		case ReLU:
			for (int b = 0; b < minibatch_size; b++)
				for (int i = 0; i < info.numNeurons; i++)
					for (int m = 0; m < info.x_row; m++)
						for (int n = 0; n < info.x_col; n++)
							_x->array(b, i, m, n) = max(z.array(b, i, m, n), 0);
			break;
		case Sigmoid:
			for (int b = 0; b < minibatch_size; b++)
				for (int i = 0; i < info.numNeurons; i++)
					for (int m = 0; m < info.x_row; m++)
						for (int n = 0; n < info.x_col; n++)
							sigmoid(z.array(b, i, m, n), _x->array(b, i, m, n));
			break;
		case Softmax:
			softmax(z, *_x, info);
			break;
		}
	}

	void backPropagation(Activation act, int idx = 0, Tensor *ans = 0)
	{
		LayerBase& in = (*inputLayer);
		switch (act)
		{
		case Identity:
			for (int b = 0; b < minibatch_size; b++)
				for (int i = 0; i < info.numNeurons; i++)
					for (int m = 0; m < info.x_row; m++)
						for (int n = 0; n < info.x_col; n++)
							dJ.dz->array(b, i, m, n) = dJ.dx->array(b, i, m, n);
			break;
		case ReLU:
			for (int b = 0; b < minibatch_size; b++)
				for (int i = 0; i < info.numNeurons; i++)
					for (int m = 0; m < info.x_row; m++)
						for (int n = 0; n < info.x_col; n++)
							dJ.dz->array(b, i, m, n) = dJ.dx->array(b, i, m, n) * ((_x->array(b, i, m, n) > 0) ? 1 : 0);
			break;
		case Sigmoid:
			for (int b = 0; b < minibatch_size; b++)
				for (int i = 0; i < info.numNeurons; i++)
					for (int m = 0; m < info.x_row; m++)
						for (int n = 0; n < info.x_col; n++)
							dJ.dz->array(b, i, m, n) = dJ.dx->array(b, i, m, n) * _x->array(b, i, m, n) * (1 - _x->array(b, i, m, n));
			break;
		case Softmax: // Cross entropy
			for (int b = 0; b < minibatch_size; b++)
				for (int i = 0; i < info.numNeurons; i++)
					for (int m = 0; m < info.x_row; m++)
						for (int n = 0; n < info.x_col; n++)
							dJ.dz->array(b, i, m, n) = dJ.dx->array(b, i, m, n) - ans->array(idx,i); // ans
			break;
		}

		if (!in.isInputLayer)
		{
			for (int b = 0; b < minibatch_size; b++)
			{
				for (int j = 0; j < in.info.numNeurons; j++)
					for (int k = 0; k < info.w_row; k++)
						for (int l = 0; l < info.w_col; l++)
						{
							in.dJ.dx->array(b, j, k, l) = 0;

							for (int i = 0; i < info.numNeurons; i++)
								for (int m = 0; m < info.x_row; m++)
									for (int n = 0; n < info.x_col; n++)
										in.dJ.dx->array(b, j, k, l) += dJ.dz->array(b, i, m, n) * w.array(i, j, k, l);
						}
			}
			/*
			for (int j = 0; j < in.numNeurons; j++)
			{
				in.dJ.dx[j] = 0;
				for (int i = 0; i < numNeurons; i++)
					in.dJ.dx[j] += dJ.dz[i] * w.array(i,j);
			}
			*/
		}
		updateWeightBias();
	}

	void updateWeightBias()
	{
		LayerBase& in = (*inputLayer);
		for (int bb = 0; bb < minibatch_size; bb++)
		{
			for (int i = 0; i < info.numNeurons; i++)
				for (int m = 0; m < info.x_row; m++)
					for (int n = 0; n < info.x_col; n++)
					{
						dJ.db->array(bb, i, m, n) = dJ.dz->array(bb, i, m, n);
						b.array(bb, i, m, n) -= learning_rate * dJ.db->array(bb, i, m, n);
					}
			// 배치를 고려하면 누적해서 나중에 한번에 파라미터 업데이트 해야 함. 나중에 고쳐야 함.
			for (int i = 0; i < info.numNeurons; i++)
				for (int j = 0; j < inputLayer->info.numNeurons; j++)
					for (int k = 0; k < info.w_row; k++)
						for (int l = 0; l < info.w_col; l++)
						{
							for (int m = 0; m < info.x_row; m++)
								for (int n = 0; n < info.x_col; n++)
									dJ.dw->array(i, j, k, l) = dJ.dz->array(bb, i, m, n) * in._x->array(bb, j, k, l);
						}

			for (int i = 0; i < info.numNeurons; i++)
				for (int j = 0; j < inputLayer->info.numNeurons; j++)
					for (int k = 0; k < info.w_row; k++)
						for (int l = 0; l < info.w_col; l++)
							w.array(i, j, k, l) -= learning_rate * dJ.dw->array(i, j, k, l);
		}

	/*

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
	*/

	}

	Tensor& getOutput()	{ return *_x; }

	// InputLayer
	void updateInput(Tensor& input, int idx)
	{
		for (int b = 0; b < minibatch_size; b++) // check with idx
		for (int ch = 0; ch < input.J; ch++)
		for (int k = 0; k < input.K; k++)
		{
			for (int l = 0; l < input.L; l++)
			{
				_x->array(b, ch, k, l) = input.array(idx, ch, k, l);
			}
		}
	}
};

#endif
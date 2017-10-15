#ifndef __LAYER__
#define __LAYER__

#include<assert.h>
#include "Layer.h"

#define MAX(a, b) (a > b) ? (a) : (b);
double Layer::learning_rate = 0.01;

// Softmax
static inline void softmax(Tensor& z, Tensor& out_x, int size)
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

void Layer::forwardPropagation(Activation _act)
{
	act = _act;
	Layer& in = (*inputLayer);
	// Calculate z values
	for (int i = 0; i < numNeurons; i++)
		z.array(i) = 0;

	for (int i = 0; i < numNeurons; i++)
	{
		for (int j = 0; j < in.numNeurons; j++)
			z.array(i) += w.array(i, j) * in.x.array(j);

		z.array(i) += b.array(i);
	}

	// Update x values
	switch (act)
	{
	case Identity:
		for (int i = 0; i < numNeurons; i++)
			x.array(i) = z.array(i);
		break;
	case ReLU:
		for (int i = 0; i < numNeurons; i++)
			x.array(i) = MAX(z.array(i), 0);
		break;
	case Sigmoid:
		for (int i = 0; i < numNeurons; i++)
			x.array(i) = 1 / (1 + exp(-z.array(i)));
		break;
	case Softmax:
		softmax(z, x, numNeurons);
		break;
	}
}

void Layer::backPropagation(Tensor* onehot, int _idx)
{
	Layer& in = (*inputLayer);
	// Update 'dJ/dz'
	switch (act)
	{
	case Identity:
		for (int i = 0; i < numNeurons; i++)
			dJ.dz[i] = dJ.dx[i];
		break;
	case ReLU:
		for (int i = 0; i < numNeurons; i++)
			dJ.dz[i] = dJ.dx[i] * ((x.array(i) > 0) ? 1 : 0);
		break;
	case Sigmoid:
		for (int i = 0; i < numNeurons; i++)
			dJ.dz[i] = dJ.dx[i] * x.array(i) * (1 - x.array(i));
		break;
	case Softmax: // cross entropy
		assert(onehot != 0);
		for (int i = 0; i < numNeurons; i++)
			dJ.dz[i] = x.array(i) - onehot->array(_idx, i);
		break;
	}

	if (!in.isInputLayer)
	{
		for (int j = 0; j < in.numNeurons; j++)
		{
			in.dJ.dx[j] = 0;
			for (int i = 0; i < numNeurons; i++)
				in.dJ.dx[j] += dJ.dz[i] * w.array(i, j);
		}
	}
	_updateWeightBias();
}

void Layer::_updateWeightBias()
{
	Layer& in = (*inputLayer);
	for (int i = 0; i < numNeurons; i++)
	{
		dJ.db[i] = dJ.dz[i];
		b.array(i) -= learning_rate * dJ.db[i];
		for (int j = 0; j < in.numNeurons; j++)
		{
			dJ.dw[i * in.numNeurons + j] = dJ.dz[i] * in.x.array(j);
			w.array(i, j) -= learning_rate * dJ.dw[i * in.numNeurons + j];
		}
	}
}

void Layer::updateInput(Tensor& input, int idx)
{
	int i = 0;
	for (UINT j = 0; j < input.J; j++) for (UINT k = 0; k < input.K; k++)
	{
		x.array(i) = input.array(idx, j, k);
		i++;
	}
}
#endif
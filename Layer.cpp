#ifndef __LAYER__
#define __LAYER__

#include<assert.h>
#include "Layer.h"
#include "common.h"

double Layer::learning_rate = 0.01;

void Layer::forwardPropagation(Activation _act)
{
	act = _act;
	Layer& in = (*inputLayer);
	// Calculate z values
	FOR4D(batch, i, xr, xc, info.minibatch_size, info.numNeurons, info.x_row, info.x_col)
		z[batch][i][xr][xc] = 0;

	FOR4D(batch, i, xr, xc, info.minibatch_size, info.numNeurons, info.x_row, info.x_col)
	{
		FOR3D(j, wr, wc, inputLayer->info.numNeurons, info.w_row, info.w_col)
			z[batch][i][xr][xc] += w[i][j][wr][wc] * in.x[batch][j][wr][wc];

		z[batch][i][xr][xc] += b[batch][i][xr][xc];
	}

	// Update x values
	switch (act)
	{
	case Identity:
		FOR4D(batch, i, xr, xc, info.minibatch_size, info.numNeurons, info.x_row, info.x_col)
			x[batch][i][xr][xc] = z[batch][i][xr][xc];
		break;
	case ReLU:
		FOR4D(batch, i, xr, xc, info.minibatch_size, info.numNeurons, info.x_row, info.x_col)
			x[batch][i][xr][xc] = CMP_MAX(z[batch][i][xr][xc], 0);
		break;
	case Sigmoid:
		FOR4D(batch, i, xr, xc, info.minibatch_size, info.numNeurons, info.x_row, info.x_col)
			x[batch][i][xr][xc] = 1 / (1 + exp(-z[batch][i][xr][xc]));
		break;
	case Softmax:
		FOR(batch, info.minibatch_size)
		{
			double z_max = 0;
			FOR3D(i, xr, xc, info.numNeurons, info.x_row, info.x_col)
				z_max = CMP_MAX(z[batch][i][xr][xc], z_max);

			double sum = 0;
			FOR3D(i, xr, xc, info.numNeurons, info.x_row, info.x_col)
				sum += exp(z[batch][i][xr][xc] - z_max);

			FOR3D(i, xr, xc, info.numNeurons, info.x_row, info.x_col)
				x[batch][i][xr][xc] = exp(z[batch][i][xr][xc] - z_max) / sum;
		}
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
		FOR4D(batch, i, xr, xc, info.minibatch_size, info.numNeurons, info.x_row, info.x_col)
			dJ.dz[batch][i][xr][xc] = dJ.dx[batch][i][xr][xc];
		break;
	case ReLU:
		FOR4D(batch, i, xr, xc, info.minibatch_size, info.numNeurons, info.x_row, info.x_col)
			dJ.dz[batch][i][xr][xc] = dJ.dx[batch][i][xr][xc] * ((x[batch][i][xr][xc] > 0) ? 1 : 0);
		break;
	case Sigmoid:
		FOR4D(batch, i, xr, xc, info.minibatch_size, info.numNeurons, info.x_row, info.x_col)
			dJ.dz[batch][i][xr][xc] = dJ.dx[batch][i][xr][xc] * x[batch][i][xr][xc] * (1 - x[batch][i][xr][xc]);
		break;
	case Softmax: // cross entropy
		assert(onehot != 0);
		FOR4D(batch, i, xr, xc, info.minibatch_size, info.numNeurons, info.x_row, info.x_col)
			dJ.dz[batch][i][xr][xc] = x[batch][i][xr][xc] - onehot->array(_idx, i);
		break;
	}

	if (!in.isInputLayer)
	{
		FOR4D(batch, j, wr, wc, info.minibatch_size, in.info.numNeurons, info.w_row, info.w_col)
		{
			in.dJ.dx[batch][j][wr][wc] = 0;

			FOR3D(i, xr, xc, info.numNeurons, info.x_row, info.x_col)
				in.dJ.dx[batch][j][wr][wc] += dJ.dz[batch][i][xr][xc] * w[i][j][wr][wc];
		}
	}
	_updateWeightBias();
}

void Layer::_updateWeightBias()
{
	Layer& in = (*inputLayer);
	FOR(batch, info.minibatch_size)
	{
		FOR3D(i, xr, xc, info.numNeurons, info.x_row, info.x_col)
		{
			dJ.db[batch][i][xr][xc] = dJ.dz[batch][i][xr][xc];
			b[batch][i][xr][xc] -= learning_rate * dJ.db[batch][i][xr][xc];
		}

		FOR4D(i, j, wr, wc, info.numNeurons, in.info.numNeurons, info.w_row, info.w_col)
		{
			dJ.dw[i][j][wr][wc] = 0;
			
			FOR2D(xr, xc, info.x_row, info.x_col)
				dJ.dw[i][j][wr][wc] = dJ.dz[batch][i][xr][xc] * in.x[batch][i][wr][wc];
			
			w[i][j][wr][wc] -= learning_rate * dJ.dw[i][j][wr][wc];
		}
	}
}

void Layer::updateInput(Tensor& input, int idx)
{
	FOR4D(i, j, k, l, info.minibatch_size, input.J, input.K, input.L)
		x[i][j][k][l] = input[idx][j][k][l];
}
#endif
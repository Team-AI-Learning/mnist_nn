#ifndef __CONV_LAYER__
#define __CONV_LAYER__

#include<assert.h>
#include "ConvLayer.h"
#include "common.h"

double ConvLayer::learning_rate = 0.01;
void ConvLayer::forwardPropagation(Activation _act)
{
	act = _act;
	ConvLayer& in = (*inputLayer);
	// Calculate z values
	FOR4D(i, xd, xr, xc, info.numNeurons, info.x_cha, info.x_row, info.x_col)
		z[i][xd][xr][xc] = 0;

	FOR4D(i, xd, xr, xc, info.numNeurons, info.x_cha, info.x_row, info.x_col)
	{
		//stride
		UINT offset_str_r = 0;
		UINT& stride = info.stride;
		info.getSizeOfZ(0, offset_str_r);
		UINT offset_str_c = offset_str_r;
		
		FOR2D(offset_r, offset_c, offset_str_r, offset_str_c)
		{
			FOR4D(j, wd, wr, wc, inputLayer->info.numNeurons, info.w_cha, info.w_row, info.w_col)
				z[i][xd][xr][xc] += w[i][j][wd][wr][wc] * in.x[j][wd][wr+offset_r*stride][wc+offset_c*stride];
		}

		z[i][xd][xr][xc] += b[i][xd][xr][xc];
	}

	// Update x values
	switch (act)
	{
	case Identity:
		break;
	case ReLU:
		FOR4D(i, xd, xr, xc, info.numNeurons, info.x_cha, info.x_row, info.x_col)
			x[i][xd][xr][xc] = CMP_MAX(z[i][xd][xr][xc], 0);
		break;
	case Sigmoid:
		FOR4D(i, xd, xr, xc, info.numNeurons, info.x_cha, info.x_row, info.x_col)
			x[i][xd][xr][xc] = 1 / (1 + exp(-z[i][xd][xr][xc]));
		break;
	case Softmax:
		double z_max = 0;
		FOR4D(i, xd, xr, xc, info.numNeurons, info.x_cha, info.x_row, info.x_col)
			z_max = CMP_MAX(z[i][xd][xr][xc], z_max);

		double sum = 0;
		FOR4D(i, xd, xr, xc, info.numNeurons, info.x_cha, info.x_row, info.x_col)
			sum += exp(z[i][xd][xr][xc] - z_max);

		FOR4D(i, xd, xr, xc, info.numNeurons, info.x_cha, info.x_row, info.x_col)
			x[i][xd][xr][xc] = exp(z[i][xd][xr][xc] - z_max) / sum;
		break;
	}
}

void ConvLayer::backPropagation(Tensor* onehot, int _idx)
{
	ConvLayer& in = (*inputLayer);

	switch (act)
	{
	case Identity:
		break;
	case ReLU:
		FOR4D(i, xd, xr, xc, info.numNeurons, info.x_cha, info.x_row, info.x_col)
			dJ.dz[i][xd][xr][xc] = dJ.dx[i][xd][xr][xc] * ((x[i][xd][xr][xc] > 0) ? 1 : 0);
		break;
	case Sigmoid:
		FOR4D(i, xd, xr, xc, info.numNeurons, info.x_cha, info.x_row, info.x_col)
			dJ.dz[i][xd][xr][xc] = dJ.dz[i][xd][xr][xc] * x[i][xd][xr][xc] * (1 - x[i][xd][xr][xc]);
		break;
	case Softmax:
		assert(onehot != 0);
		FOR4D(i, xd, xr, xc, info.numNeurons, info.x_cha, info.x_row, info.x_col)
			dJ.dz[i][xd][xr][xc] = x[i][xd][xr][xc] - onehot->array(_idx, i);
		break;
	}

	if (!in.isInputLayer)
	{
		FOR4D(j, wd, wr, wc, in.info.numNeurons, info.w_cha, info.w_row, info.w_col)
		{
			in.dJ.dx[j][wd][wr][wc] = 0;

			FOR4D(i, xd, xr, xc, info.numNeurons, info.x_cha, info.x_row, info.x_col)
				in.dJ.dx[j][wd][wr][wc] += dJ.dz[i][xd][xr][xc] * w[i][j][wd][wr][wc];
		}
	}
	_updateWeightBias();
}

void ConvLayer::_updateWeightBias()
{
	ConvLayer& in = (*inputLayer);
	FOR4D(i, xd, xr, xc, info.numNeurons, info.x_cha, info.x_row, info.x_col)
	{
		dJ.db[i][xd][xr][xc] = dJ.dz[i][xd][xr][xc];
		b[i][xd][xr][xc] -= learning_rate * dJ.db[i][xd][xr][xc];
	}

	FOR5D(i, j, wd, wr, wc, info.numNeurons, in.info.numNeurons, info.w_cha, info.w_row, info.w_col)
	{
		dJ.dw[i][j][wd][wr][wc] = 0;

		FOR3D(xd, xr, xc, info.x_cha, info.x_row, info.x_col)
			dJ.dw[i][j][wd][wr][wc] = dJ.dz[i][xd][xr][xc] * in.x[j][wd][wr][wc];

		w[i][j][wd][wr][wc] -= learning_rate * dJ.dw[i][j][wd][wr][wc];
	}
}

void ConvLayer::updateInput(Tensor& input, int idx)
{
	FOR4D(i, k, l, m, info.numNeurons, input.K, input.L, input.M)
		x[i][k][l][m] = input[0][idx][k][l][m];
}

#endif
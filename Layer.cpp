#ifndef __LAYER__
#define __LAYER__

#include<assert.h>
#include<ctime>
#include "Layer.h"
#include "common.h"

double Layer::learning_rate = 0.01;

void Layer::forwardPropagation(Activation _act)
{
	act = _act;
	Layer& in = (*inputLayer);
	// Calculate z values
#pragma omp parallel for 
	FOR4D(batch, f, xr, xc, info.minibatch_size, info.numFilters, info.x_row, info.x_col)
		z[batch][f][xr][xc] = 0;

	FOR4D(batch, f, xr, xc, info.minibatch_size, info.numFilters, info.x_row, info.x_col)
	{
		UINT offset_str_r = 0;
		offset_str_r = info.getSizeOfZ(in.info.x_row);
		UINT offset_str_c = offset_str_r;
		UINT& stride = info.stride;

		double local_z = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+:local_z) // TODO: job to 1 but this value is calculated by stride&filter size
		FOR2D_OMP(offset_r, offset_c, offset_str_r, offset_str_c, stride_calculation)
		
			FOR3D(ch, wr, wc, inputLayer->info.numChannels, info.w_row, info.w_col)
				local_z += w[f][ch][wr][wc] * in.x[batch][ch][wr+offset_r*stride][wc+offset_c*stride];

		FOR_OMP_END
		
		local_z += b[batch][f][xr][xc];

		z[batch][f][xr][xc] += local_z;
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
#pragma omp parallel for 
			FOR3D(i, xr, xc, info.numNeurons, info.x_row, info.x_col)
				z_max = CMP_MAX(z[batch][i][xr][xc], z_max);

			double sum = 0;
#pragma omp parallel for reduction(+:sum)
			FOR3D(i, xr, xc, info.numNeurons, info.x_row, info.x_col)
				sum += exp(z[batch][i][xr][xc] - z_max);

#pragma omp parallel for 
			FOR3D(i, xr, xc, info.numNeurons, info.x_row, info.x_col)
				x[batch][i][xr][xc] = exp(z[batch][i][xr][xc] - z_max) / sum;
		}
		break;
	}
}

Tensor* Layer::maxPooling(bool require)
{
	Layer& in = *inputLayer;
	Tensor* idxInfo = 0;
	if(require)
		idxInfo = new Tensor(in.info.minibatch_size, in.info.numNeurons, in.info.x_row, in.info.x_col, false);
	
	FOR3D(batch, xr, xc, info.minibatch_size, info.x_row, info.x_col)
	{
		int offset_r = info.x_row*xr;
		int offset_c = info.x_col*xc;
		FOR(i, info.numNeurons)
		{
			double maxPool = -INFINITY;
			int idx_r = 0;
			int idx_c = 0;
			FOR2D(_xr, _xc, info.x_row, info.x_col)
			{
				double in_x = in.x[batch][i][_xr + offset_r][_xc + offset_c];
				if (in_x > maxPool)
				{
					maxPool = in_x;
					idx_r = _xr + offset_r;
					idx_c = _xc + offset_c;
				}
			}

			if(require)
				(*idxInfo)[batch][i][idx_r][idx_c] = 1;
			x[batch][i][xr][xc] = maxPool;
		}
	}
	return idxInfo;
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
	case MaxPooling:
		FOR4D(batch, i, xr, xc, info.minibatch_size, info.numNeurons, info.x_row, info.x_col)
		{
			Tensor& idxInfo = *onehot;
			if (idxInfo[batch][i][xr][xc] >= 1)
				in.dJ.dx[batch][i][xr][xc] = dJ.dx[batch][i][xr][xc];
			else
				in.dJ.dx[batch][i][xr][xc] = 0;
		}
		break;
	}

	if (!in.isInputLayer)
	{
		if(in.act != MaxPooling) 
		FOR4D(batch, j, wr, wc, info.minibatch_size, in.info.numNeurons, info.w_row, info.w_col)
		{
			in.dJ.dx[batch][j][wr][wc] = 0;
			double local_in_x = 0;

#pragma omp parallel for reduction(+:local_in_x)
			FOR3D(i, xr, xc, info.numNeurons, info.x_row, info.x_col)
				local_in_x += dJ.dz[batch][i][xr][xc] * w[i][j][wr][wc];

			in.dJ.dx[batch][j][wr][wc] = local_in_x;
		}
	}
	if(act != MaxPooling)
		_updateWeightBias();
}

void Layer::_updateWeightBias()
{
	Layer& in = (*inputLayer);
	FOR(batch, info.minibatch_size)
	{
#pragma omp parallel for
		FOR3D(i, xr, xc, info.numNeurons, info.x_row, info.x_col)
		{
			dJ.db[batch][i][xr][xc] = dJ.dz[batch][i][xr][xc];
			b[batch][i][xr][xc] -= learning_rate * dJ.db[batch][i][xr][xc];
		}
#pragma omp parallel for
		FOR4D(i, j, wr, wc, info.numNeurons, in.info.numNeurons, info.w_row, info.w_col)
		{
			dJ.dw[i][j][wr][wc] = 0;
			
			FOR2D(xr, xc, info.x_row, info.x_col)
				dJ.dw[i][j][wr][wc] = dJ.dz[batch][i][xr][xc] * in.x[batch][j][wr][wc];
			
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
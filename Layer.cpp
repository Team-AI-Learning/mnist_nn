#ifndef __LAYER__
#define __LAYER__

#include<assert.h>
#include<ctime>
#include "Layer.h"
#include "common.h"

double Layer::learning_rate = 0.01;

void Layer::forwardPropagation()
{
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
	switch (info.act)
	{
	case Act::Identity:
		FOR4D(batch, i, xr, xc, info.minibatch_size, info.numNeurons, info.x_row, info.x_col)
			x[batch][i][xr][xc] = z[batch][i][xr][xc];
		break;
	case Act::ReLU:
		FOR4D(batch, i, xr, xc, info.minibatch_size, info.numNeurons, info.x_row, info.x_col)
			x[batch][i][xr][xc] = CMP_MAX(z[batch][i][xr][xc], 0);
		break;
	case Act::Sigmoid:
		FOR4D(batch, i, xr, xc, info.minibatch_size, info.numNeurons, info.x_row, info.x_col)
			x[batch][i][xr][xc] = 1 / (1 + exp(-z[batch][i][xr][xc]));
		break;
	case Act::Softmax:
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

void Layer::maxPooling()
{
	if (maxpool_idx == 0)
		maxpool_idx = new MaxPoolingIdxInfo(info.minibatch_size,
			info.numChannels, info.x_row, info.x_col, false);
	else
		maxpool_idx->setZero<pair<int,int>>();
	
	// Calculate stride & size of filter of maxpooling
	Layer& in = (*inputLayer);
	info.w_col = info.w_row = in.info.x_row / info.x_row;
	// You have to initialize size of correct x_row value.
	assert(in.info.x_row % info.x_row == 0);
	UINT& stride = info.w_row;

	double local_z = 0;
	FOR2D(batch, ch, info.minibatch_size, info.numChannels)
	{
#pragma omp parallel for schedule(dynamic, 1) reduction(+:local_z) // TODO: job to 1 but this value is calculated by stride&filter size
		FOR2D_OMP(xr, xc, info.x_row, info.x_col, stride_calculation)
			double max_local_x = -INFINITY;
			int local_idx_r = 0;
			int local_idx_c = 0;
			// Find Max value and the index
			FOR2D(wr, wc, info.w_row, info.w_col)
			{
				double in_x = in.x[batch][ch][wr + xr*stride][wc + xc*stride];
				if (in_x >= max_local_x)
				{
					max_local_x = in_x;
					local_idx_r = wr + xr*stride;
					local_idx_c = wc + xc*stride;
				}
			}
			// store the indice of input x of maximum value.
			(*maxpool_idx)[batch][ch][xr][xc] = std::make_pair(local_idx_r, local_idx_c);
		FOR_OMP_END

	}
}

void Layer::backPropagation(Tensor<>* onehot, int _idx)
{
	Layer& in = (*inputLayer);
	// Update 'dJ/dz'
	switch (info.act)
	{
	case Act::Identity:
		FOR4D(batch, i, xr, xc, info.minibatch_size, info.numNeurons, info.x_row, info.x_col)
			dJ.dz[batch][i][xr][xc] = dJ.dx[batch][i][xr][xc];
		break;
	case Act::ReLU:
		FOR4D(batch, i, xr, xc, info.minibatch_size, info.numNeurons, info.x_row, info.x_col)
			dJ.dz[batch][i][xr][xc] = dJ.dx[batch][i][xr][xc] * ((x[batch][i][xr][xc] > 0) ? 1 : 0);
		break;
	case Act::Sigmoid:
		FOR4D(batch, i, xr, xc, info.minibatch_size, info.numNeurons, info.x_row, info.x_col)
			dJ.dz[batch][i][xr][xc] = dJ.dx[batch][i][xr][xc] * x[batch][i][xr][xc] * (1 - x[batch][i][xr][xc]);
		break;
	case Act::Softmax: // cross entropy
		assert(onehot != 0);
		FOR4D(batch, i, xr, xc, info.minibatch_size, info.numNeurons, info.x_row, info.x_col)
			dJ.dz[batch][i][xr][xc] = x[batch][i][xr][xc] - onehot->array(_idx, i);
		break;
	case Act::MaxPooling:
		FOR4D(batch, in_i, in_xr, in_xc, in.info.minibatch_size, in.info.numNeurons, in.info.x_row, in.info.x_col)
			in.dJ.dx[batch][in_i][in_xr][in_xc] = 0;

		FOR4D(batch, i, xr, xc, info.minibatch_size, info.numNeurons, info.x_row, info.x_col)
		{
			pair<int, int> &p = (*maxpool_idx)[batch][i][xr][xc];
			in.dJ.dx[batch][i][p.first][p.second] = dJ.dx[batch][i][xr][xc];
		}
		return;
		break;
	}

	if (!in.isInputLayer)
	{
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

void Layer::updateInput(Tensor<>& input, int idx)
{
	FOR4D(i, j, k, l, info.minibatch_size, input.J, input.K, input.L)
		x[i][j][k][l] = input[idx][j][k][l];
}
#endif
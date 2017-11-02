#ifndef __LAYER__H
#define __LAYER__H

#include <vector>
#include <utility>
#include "Tensor.hpp"
#include "common.h"

// numNeurons, x_row, x_col, w_row, w_col, stride, minibatch_size
struct LayerInfo
{
	enum Activation { None, Identity, ReLU, Sigmoid, Softmax, MaxPooling };
	Activation act;
	// Note that All of numbers of Neurons, Channels and Filters are actually same.
	UINT numNeurons;
	UINT& numChannels;
	UINT& numFilters;
	
	UINT x_row;
	UINT x_col;
	UINT w_row;
	UINT w_col;

	UINT stride;
	UINT minibatch_size;
	LayerInfo(Activation _act, UINT _numNeurons, UINT _x_row, UINT _x_col, UINT _w_row = 0, UINT _w_col = 0, UINT _stride = 0, UINT _batch = 1 ) 
		: act(_act), numNeurons(_numNeurons), numChannels(numNeurons), numFilters(numNeurons),
		x_row(_x_row), x_col(_x_col), w_row(_w_row), w_col(_w_col), stride(_stride), minibatch_size(_batch)
		{}

	UINT getSizeOfZ(UINT in_x_size, UINT* out_padding = 0)
	{
		if (stride == 0)
			return 1;

		UINT& filter_size = w_row;
		UINT ret = (in_x_size - filter_size) / stride + 1;
		UINT r = (in_x_size - filter_size) % stride;
		UINT pad = 0;

		if (r != 0) // padding is required !
		{
			ret++;
			pad = stride - r;
			if (pad % 2 != 0)
				pad++;
		}
		if(out_padding != 0)
			*out_padding = pad;
		return ret;
	}

};

struct DJ // pointers must be declared first
{
	Tensor<> *_dx;
	Tensor<> *_dz;
	Tensor<> *_dw;
	Tensor<> *_db;
	bool allocated;

	Tensor<>& dx;
	Tensor<>& dz;
	Tensor<>& dw;
	Tensor<>& db;

	// DJ(number of input neurons, number of current neurons)
	DJ(UINT input_numNeurons, LayerInfo& info)
		:
		_dx(new Tensor<>(info.minibatch_size, info.numNeurons, info.x_row, info.x_col)),
		_dz(new Tensor<>(info.minibatch_size, info.numNeurons, info.x_row, info.x_col)),
		_dw( new Tensor<>(info.numNeurons,	input_numNeurons, info.w_row, info.w_col)),
		_db(new Tensor<>(info.minibatch_size, info.numNeurons, info.x_row, info.x_col)),
		dx(*_dx), dz(*_dz), dw(*_dw), db(*_db)
	{
		
	}
	// Workaround code for Maxpooling
	DJ(UINT input_numNeurons, LayerInfo& info, bool is_maxpooling)
		:
		_dx(new Tensor<>(info.minibatch_size, info.numNeurons, info.x_row, info.x_col)),
		_dz(0),
		_dw(0),
		_db(0),
		dx(*_dx), dz(*_dz), dw(*_dw), db(*_db)
	{

	}
	~DJ()
	{
		// Check whether DJ has been allocated.
		if (_dx != 0)
		{
			delete _dx;
			delete _dz;
			delete _dw;
			delete _db;
		}
		_dx = 0; _dz = 0; _dw = 0; _db = 0;
	}
};

class Layer
{
public:
	friend struct DJ;
	static double learning_rate;
public:
	bool isInputLayer;
	Layer* inputLayer;
	LayerInfo info;
	typedef LayerInfo::Activation Act;
protected:
	Tensor<> *_x;
	DJ *_dJ;
	typedef Tensor<pair<int, int>> MaxPoolingIdxInfo;
	MaxPoolingIdxInfo *maxpool_idx;

	Tensor<>& x;
	Tensor<> z;
	Tensor<> w;
	Tensor<> b;
	DJ& dJ;
public:
	// Constructor for Hidden Layer
	// if allocation fails, mem leak would be occurred.
	explicit Layer(Layer* _inputLayer, LayerInfo _info)
		: inputLayer(_inputLayer), info(_info), 
		_x(new Tensor<>(info.minibatch_size, info.numFilters, info.x_row, info.x_col)),
		_dJ(new DJ(inputLayer->info.numFilters, info)),
		maxpool_idx(0),
		x(*_x), dJ(*_dJ),
		z(info.minibatch_size, info.numFilters, info.x_row, info.x_col),
		w(info.numFilters, inputLayer->info.numChannels, info.w_row, info.w_col),
		b(info.minibatch_size, info.numFilters, info.x_row, info.x_col)
	{
		isInputLayer = false;
	}

	// Workaround code for Maxpooling
	explicit Layer(Layer* _inputLayer, LayerInfo _info, bool maxpooling)
		: inputLayer(_inputLayer), info(_info),
		_x(new Tensor<>(info.minibatch_size, info.numFilters, info.x_row, info.x_col)),
		_dJ(new DJ(inputLayer->info.numFilters, info, maxpooling)),
		maxpool_idx(0),
		x(*_x), dJ(*_dJ)
	{
		isInputLayer = false;
	}

	// Constructor for Input Layer.
	explicit Layer(LayerInfo _info)
		: inputLayer(0), info(_info), 
		_x(new Tensor<>(info.minibatch_size, info.numChannels, info.x_row, info.x_col)),
		_dJ(0), 
		maxpool_idx(0),
		x(*_x), dJ(*_dJ)
	{
		isInputLayer = true;
	}

	virtual ~Layer()
	{
		delete _x; // _x should be always allocated.
		_x = 0;
		if (maxpool_idx != 0)
			delete maxpool_idx;
		maxpool_idx = 0;
	}

	// Set Activation method and propagate
	void forwardPropagation();

	// Pooling
	void maxPooling();

	// Parameters are for Output layer.
	void backPropagation(Tensor<>* onehot = 0, int _idx = 0);

	// Returns a pointer of x
	Tensor<>& getOutput() { return x; }
	
	// To switch input data for Input Layer
	void updateInput(Tensor<>& input, int idx);
protected:
	// Calculate dw, db and Update w, b values after dz, dx.
	void _updateWeightBias();
};

#endif
#ifndef __CONV_LAYER__H
#define __CONV_LAYER__H

#include "Tensor.hpp"
#include "common.h"

// numNeurons, x, w, stride
struct ConvLayerInfo
{
	UINT numNeurons; // is numChannel if only inputLayer

	UINT x_cha; // channel (depth) of x
	UINT x_row;
	UINT x_col;

	UINT w_nFilter;
	UINT w_cha; // channel (depth) of w
	UINT w_row;
	UINT w_col;

	UINT stride;

	// returns additional padding and size of z.
	void getSizeOfZ(UINT* out_padding, UINT& z_size) 
	{
		if (stride == 0)
		{
			if(out_padding != 0)
				*out_padding = 0;
			z_size = 1;
			return;
		}

		z_size = (x_row - w_nFilter) / stride + 1;
		UINT r = (x_row - w_nFilter) % stride;
		UINT q = (x_row - w_nFilter) / stride;
		if(out_padding != 0)
			*out_padding = (stride - r) / 2;
		if ((stride - 2) % 2 != 0)
		{
			if (out_padding != 0)
				(*out_padding)++; // padding is required.
			z_size++;
		}

	}
};

struct DJ // pointers must be declared first
{
	Tensor *_dx;
	Tensor *_dz;
	Tensor *_dw;
	Tensor *_db;
	bool allocated;

	double****& dx;
	double****& dz;
	Tensor& dw;
	double****& db;

	// DJ(number of input neurons, number of current neurons)
	DJ(UINT input_numNeurons, ConvLayerInfo& info)
		:
		_dx(new Tensor(1, info.numNeurons, info.x_cha, info.x_row, info.x_col)),
		_dz(new Tensor(1, info.numNeurons, info.x_cha, info.x_row, info.x_col)),
		_dw(new Tensor(info.numNeurons, input_numNeurons, info.w_cha, info.w_row, info.w_col)),
		_db(new Tensor(1, info.numNeurons, info.x_cha, info.x_row, info.x_col)),
		dx((*_dx)[0]), dz((*_dz)[0]), dw(*_dw), db((*_db)[0])
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

class ConvLayer
{
public:
	friend struct DJ;
	static double learning_rate;
public:
	enum Activation { Identity, ReLU, Sigmoid, Softmax };
	bool isInputLayer;
	ConvLayer* inputLayer;
	ConvLayerInfo info;
protected:
	Activation act;
protected:
	Tensor* _x;
	Tensor* _z;
	Tensor* _w;
	Tensor* _b;
	DJ *_dJ;

	double****& x;
	double****& z;
	Tensor& w;
	double****& b;
	DJ& dJ;
public:
	// Constructor for Hidden Layer
	// note that if allocation fails, mem leak would be occurred.
	explicit ConvLayer(ConvLayer* _inputLayer, ConvLayerInfo _info)
		: inputLayer(_inputLayer), info(_info),
		_x(new Tensor(1, info.numNeurons, info.x_cha, info.x_row, info.x_col)),
		_z(new Tensor(1, info.numNeurons, info.x_cha, info.x_row, info.x_col)),
		_w(new Tensor(info.numNeurons, inputLayer->info.numNeurons, info.w_cha, info.w_row, info.w_col)),
		_b(new Tensor(1, info.numNeurons, info.x_cha, info.x_row, info.x_col)),
		_dJ(new DJ(inputLayer->info.numNeurons, info)),
		x((*_x)[0]), z((*_z)[0]), w(*_w), b((*_b)[0]),
		dJ(*_dJ)
	{
		isInputLayer = false;
	}

	// Constructor for Input Layer.
	explicit ConvLayer(ConvLayerInfo _info)
		: inputLayer(0), info(_info),
		_x(new Tensor(info.numNeurons, info.x_cha, info.x_row, info.x_col)),
		_z(0), _w(0), _b(0), _dJ(0), x((*_x)[0]), z(x), w(*_x), b(x),
		dJ(*_dJ)
	{
		isInputLayer = true;
	}

	virtual ~ConvLayer()
	{
		delete _x; // _x should be always allocated.
		_x = 0;
	}

	// Set Activation method and propagate
	virtual void forwardPropagation(Activation _act);

	// Parameters are for Output layer.
	virtual void backPropagation(Tensor* onehot = 0, int _idx = 0);

	// Returns a pointer of x
	double****& getOutput() { return x; }

	// To switch input data for Input Layer
	virtual void updateInput(Tensor& input, int idx);
protected:
	// Calculate dw, db and Update w, b values after dz, dx.
	virtual void _updateWeightBias();

};

#endif

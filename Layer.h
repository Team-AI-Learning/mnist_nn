#ifndef __LAYER__H
#define __LAYER__H

#include "Tensor.hpp"
#include "common.h"

struct LayerInfo
{
	UINT numNeurons;
	UINT x_row;
	UINT x_col;
	UINT w_row;
	UINT w_col;
	UINT minibatch_size;
	//stride
};

struct DJ // pointers must be declared first
{
	Tensor *_dx;
	Tensor *_dz;
	Tensor *_dw;
	Tensor *_db;
	bool allocated;

	Tensor& dx;
	Tensor& dz;
	Tensor& dw;
	Tensor& db;

	// DJ(number of input neurons, number of current neurons)
	DJ(UINT input_numNeurons, LayerInfo& info)
		:
		_dx(new Tensor(info.minibatch_size, info.numNeurons, info.x_row, info.x_col)),
		_dz(new Tensor(info.minibatch_size, info.numNeurons, info.x_row, info.x_col)),
		_dw(new Tensor(info.numNeurons, input_numNeurons, info.w_row, info.w_col)),
		_db(new Tensor(info.minibatch_size, info.numNeurons, info.x_row, info.x_col)),
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
	friend class DJ;
	static double learning_rate;
public:
	enum Activation {Identity, ReLU, Sigmoid, Softmax};
	bool isInputLayer;
	Layer* inputLayer;
	LayerInfo info;
protected:
	Activation act;
protected:
	Tensor *_x; 
	DJ *_dJ;

	Tensor& x;
	Tensor z;
	Tensor w;
	Tensor b;
	DJ& dJ;
public:
	// Constructor for Hidden Layer
	// note that if allocation fails, mem leak would be occurred.
	explicit Layer(Layer* _inputLayer, LayerInfo _info)
		: inputLayer(_inputLayer), info(_info), 
		_dJ(new DJ(inputLayer->info.numNeurons, info)), dJ(*_dJ),
		_x(new Tensor(info.minibatch_size, info.numNeurons, info.x_row, info.x_col)), x(*_x),
		z(info.minibatch_size, info.numNeurons, info.x_row, info.x_col),
		w(info.numNeurons, inputLayer->info.numNeurons, info.w_row, info.w_col),
		b(info.minibatch_size, info.numNeurons, info.x_row, info.x_col)
	{
		isInputLayer = false;
	}

	// Constructor for Input Layer.
	explicit Layer(LayerInfo _info)
		: info(_info),
		_dJ(0), dJ(*_dJ),
		_x(new Tensor(info.minibatch_size, info.numNeurons, info.x_row, info.x_col)), x(*_x)
	{
		isInputLayer = true;
	}

	virtual ~Layer()
	{
		delete _x; // _x should be always allocated.
		_x = 0;
	}

	// Set Activation method and propagate
	void forwardPropagation(Activation _act);

	// Parameters are for Output layer.
	void backPropagation(Tensor* onehot = 0, int _idx = 0);

	// Returns a pointer of x
	Tensor& getOutput() { return x; }
	
	// To switch input data for Input Layer
	void updateInput(Tensor& input, int idx);
protected:
	// Calculate dw, db and Update w, b values after dz, dx.
	void _updateWeightBias();
};

#endif
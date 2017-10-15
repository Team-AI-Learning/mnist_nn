#ifndef __LAYER__H
#define __LAYER__H

#include "Tensor.hpp"

struct DJ
{
	double *dx;
	double *dz;
	double *dw;
	double *db;

	// DJ(number of input neurons, number of current neurons)
	DJ(int _in_num_neurons, int _num_neurons)
	{
		dx = new double[_num_neurons];
		dz = new double[_num_neurons];
		dw = new double[_num_neurons*_in_num_neurons];
		db = new double[_num_neurons];
	}

	~DJ()
	{
		// Check whether DJ has been allocated.
		if (dx != 0)
		{
			delete[] dx;
			delete[] dz;
			delete[] dw;
			delete[] db;
		}
		dx = 0; dz = 0; dw = 0; db = 0;
	}
};

class Layer
{
public:
	static double learning_rate;
public:
	enum Activation {Identity, ReLU, Sigmoid, Softmax};
	bool isInputLayer;
	Layer* inputLayer;
protected:
	Activation act;
	int numNeurons;

	Tensor *_x; // pointers must be declared first
	DJ *_dJ;

	Tensor& x;
	Tensor z;
	Tensor w;
	Tensor b;
	DJ& dJ;
private:
public:
	// Constructor for Hidden Layer
	// note that if allocation fails, mem leak would be occurred.
	explicit Layer(Layer* _inputLayer, int _numNeurons)
		: inputLayer(_inputLayer), numNeurons(_numNeurons),
		_dJ(new DJ(inputLayer->numNeurons, numNeurons)), dJ(*_dJ),
		_x(new Tensor(numNeurons)), x(*_x), 
		z(numNeurons), 
		w(numNeurons, inputLayer->numNeurons), 
		b(numNeurons)
	{
		isInputLayer = false;
	}

	// Constructor for Input Layer.
	explicit Layer(int _numNeurons)
		: numNeurons(_numNeurons),
		_dJ(0), dJ(*_dJ),
		_x(new Tensor(numNeurons)), x(*_x)
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
#ifndef __LAYER_BASE__
#define __LAYER_BASE__

#include<math.h>
#include<random>
#include<time.h>
using namespace std;

// activation functions
static void sigmoid(double z, double& out_x)
{
	out_x = 1 / (1 + exp(-z));
}

static void softmax(double *z, double*& const out_x, int size)
{
	double z_max = 0;
	for (int i = 0; i < size; i++)
		z_max = z[i] > z_max ? z[i] : z_max;

	double sum = 0;
	for (int i = 0; i < size; i++)
		sum += exp(z[i] - z_max);

	for (int i = 0; i < size; i++)
		out_x[i] = exp(z[i] - z_max) / sum;
}

// J - loss functions
static double mean_square(double *t, double* x, int size)
{
	double sum = 0;
	for (int i = 0; i < size; i++)
		sum += 0.5 * (t[i] - x[i]) * (t[i] - x[i]);

	return sum;
}

static void setRandomVec(double v[], int size_in)
{
	default_random_engine generator;
	double variance = 1.0 / (double)size_in;
	normal_distribution<double> distribution(0.0, variance);

	for (int i = 0; i < size_in; ++i)
		v[i] = distribution(generator);
}

static void setRandomMat(double m[], int size_in, int size_out)
{
	default_random_engine generator;
	double variance = 1.0 / (double)size_in;
	normal_distribution<double> distribution(0.0, variance);

	for (int i = 0; i < size_out; ++i)
		for (int j = 0; j < size_in; ++j)
			m[size_in * i + j] = distribution(generator);
}

struct DJ // neuron data
{
	double *dx;
	double *dz;
	double *dw;
	double *db;
	
	DJ(int input_numNeurons, int numNeurons)
	{
		dx = new double[numNeurons];
		dz = new double[numNeurons];
		dw = new double[numNeurons*input_numNeurons];
		db = new double[numNeurons];
	}
	DJ() {}

	~DJ()
	{
		delete[] dx;
		delete[] dz;
		delete[] dw;
		delete[] db;
	}
};

static double learning_rate = 0.01;
class LayerBase
{
public:
	enum Activation {Sigmoid=1, Softmax};
public:
	
	int nNeuron;
	bool isFirstLayer;
protected:
	LayerBase* inputLayer;

	DJ dJ;
	double *x;
	double *z;
	double *w;
	double *b;

public:
	explicit LayerBase(LayerBase* i_inputLayer, int numNeurons)
		: inputLayer(i_inputLayer), nNeuron(numNeurons), isFirstLayer(false), 
		dJ(i_inputLayer->nNeuron, numNeurons)
	{
		x = new double[nNeuron];
		z = new double[nNeuron];
		w = new double[nNeuron * inputLayer->nNeuron];
		b = new double[nNeuron];

		setRandomVec(b, nNeuron);
		setRandomMat(w, inputLayer->nNeuron, nNeuron);
	}
	~LayerBase()
	{
		if (!isFirstLayer)
		{
			delete[] x;
			delete[] z;
			delete[] w;
			delete[] b;
		}
	}

	void forwardPropagation(Activation act)
	{
		LayerBase& in = (*inputLayer);
		// Calculate z values
		for (int i = 0; i < nNeuron; i++)
			z[i] = 0;

		for (int i = 0; i < nNeuron; i++)
		{
			for (int j = 0; j < in.nNeuron; j++)
				z[i] += w[i*in.nNeuron + j] * in.x[j];

			z[i] += b[i];
		}

		// Update x values
		switch (act)
		{
		case Sigmoid:
			for (int i = 0; i < nNeuron; i++)
				sigmoid(z[i], x[i]);
			break;
		case Softmax:
			softmax(z, x, nNeuron);
			break;
		}
	}

	void backPropagation(Activation act, double *ans = 0)
	{
		LayerBase& in = (*inputLayer);
		switch (act)
		{
		case Sigmoid:
			for (int i = 0; i < nNeuron; i++)
				dJ.dz[i] = dJ.dx[i] * x[i] * (1 - x[i]);
			break;
		case Softmax: // Cross entropy
			for (int i = 0; i < nNeuron; i++)
				dJ.dz[i] = x[i] - ans[i];
			break;
		}

		if (!in.isFirstLayer)
		{
			for (int j = 0; j < in.nNeuron; j++)
			{
				in.dJ.dx[j] = 0;
				for (int i = 0; i < nNeuron; i++)
					in.dJ.dx[j] += dJ.dz[i] * w[in.nNeuron * i + j];
			}
		}
		updateWeightBias();
	}

	void updateWeightBias()
	{
		LayerBase& in = (*inputLayer);
		for (int i = 0; i < nNeuron; i++)
		{
			dJ.db[i] = dJ.dz[i];
			b[i] -= learning_rate * dJ.db[i];
			for (int j = 0; j < in.nNeuron; j++)
			{
				dJ.dw[i * in.nNeuron + j] = dJ.dz[i] * in.x[j];
				w[i * in.nNeuron + j] -= learning_rate * dJ.dw[i * in.nNeuron + j];
			}
		}
	}

	// InputLayer
	static LayerBase CreateInputLayerBase(double*& arr, int size)
	{
		LayerBase ret;
		ret.isFirstLayer = true;
		ret.x = arr;
		ret.nNeuron = size;
		return ret;
	}

	void updateInput(double *new_input_arr)
	{
		x = new_input_arr;
	}
	
	double* getOutput() { return x; }
protected:
	LayerBase()
		:inputLayer(0)
	{

	}
};

#endif
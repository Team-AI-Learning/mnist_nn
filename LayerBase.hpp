#ifndef __LAYER_BASE__
#define __LAYER_BASE__

#define max(a, b) (a > b) ? (a) : (b);

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
	
	DJ(int input_num_neurons, int num_neurons)
	{
		dx = new double[num_neurons];
		dz = new double[num_neurons];
		dw = new double[num_neurons*input_num_neurons];
		db = new double[num_neurons];
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
	enum Activation {Identity, ReLU, Sigmoid, Softmax};
public:
	
	int numNeurons;
	bool isFirstLayer;
protected:
	LayerBase* inputLayer;

	DJ dJ;
	double *x;
	double *z;
	double *w;
	double *b;

public:
	explicit LayerBase(LayerBase* i_inputLayer, int num_neurons)
		: inputLayer(i_inputLayer), numNeurons(num_neurons), isFirstLayer(false),
		dJ(i_inputLayer->numNeurons, num_neurons)
	{
		x = new double[numNeurons];
		z = new double[numNeurons];
		w = new double[numNeurons * inputLayer->numNeurons];
		b = new double[numNeurons];

		setRandomVec(b, numNeurons);
		setRandomMat(w, inputLayer->numNeurons, numNeurons);
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
		for (int i = 0; i < numNeurons; i++)
			z[i] = 0;

		for (int i = 0; i < numNeurons; i++)
		{
			for (int j = 0; j < in.numNeurons; j++)
				z[i] += w[i*in.numNeurons + j] * in.x[j];

			z[i] += b[i];
		}

		// Update x values
		switch (act)
		{
		case Identity:
			for (int i = 0; i < numNeurons; i++)
				x[i] = z[i];
			break;
		case ReLU:
			for (int i = 0; i < numNeurons; i++)
				x[i] = max(z[i], 0);
			break;
		case Sigmoid:
			for (int i = 0; i < numNeurons; i++)
				sigmoid(z[i], x[i]);
			break;
		case Softmax:
			softmax(z, x, numNeurons);
			break;
		}
	}

	void backPropagation(Activation act, double *ans = 0)
	{
		LayerBase& in = (*inputLayer);
		switch (act)
		{
		case Identity:
			for (int i = 0; i < numNeurons; i++)
				dJ.dz[i] = dJ.dx[i];
			break;
		case ReLU:
			for (int i = 0; i < numNeurons; i++)
				dJ.dz[i] = dJ.dx[i] * ((x[i] > 0) ? 1 : 0);
			break;
		case Sigmoid:
			for (int i = 0; i < numNeurons; i++)
				dJ.dz[i] = dJ.dx[i] * x[i] * (1 - x[i]);
			break;
		case Softmax: // Cross entropy
			for (int i = 0; i < numNeurons; i++)
				dJ.dz[i] = x[i] - ans[i];
			break;
		}

		if (!in.isFirstLayer)
		{
			for (int j = 0; j < in.numNeurons; j++)
			{
				in.dJ.dx[j] = 0;
				for (int i = 0; i < numNeurons; i++)
					in.dJ.dx[j] += dJ.dz[i] * w[in.numNeurons * i + j];
			}
		}
		updateWeightBias();
	}

	void updateWeightBias()
	{
		LayerBase& in = (*inputLayer);
		for (int i = 0; i < numNeurons; i++)
		{
			dJ.db[i] = dJ.dz[i];
			b[i] -= learning_rate * dJ.db[i];
			for (int j = 0; j < in.numNeurons; j++)
			{
				dJ.dw[i * in.numNeurons + j] = dJ.dz[i] * in.x[j];
				w[i * in.numNeurons + j] -= learning_rate * dJ.dw[i * in.numNeurons + j];
			}
		}
	}

	// InputLayer
	static LayerBase CreateInputLayerBase(double*& arr, int size)
	{
		LayerBase ret;
		ret.isFirstLayer = true;
		ret.x = arr;
		ret.numNeurons = size;
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
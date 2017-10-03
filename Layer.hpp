#include<math.h>
#include<random>
#include<time.h>

// activation functions
void sigmoid(double z, double& out_x)
{
	out_x = 1 / (1 + exp(-z));
}

void softmax(double *z, double* out_x, int size)
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
double mean_square(double *t, double* x, int size)
{
	double sum = 0;
	for (int i = 0; i < size; i++)
		sum += 0.5 * (t[i] - x[i]) * (t[i] - x[i]);

	return sum;
}

double cross_entropy(double *t, double* x, int size)
{
	double sum = 0;
	for (int i = 0; i < size; i++)
		sum += t[i] * log(x[i]);

	return -sum;
}

struct DJ // neuron data
{
	double *dx_lower;
	double *dz;
	double *db;
	double *dw;
	DJ(int input_size, int n_neuron)
	{
		dw = new double[n_neuron*input_size];
		for (int i = 0; i < n_neuron*input_size; i++)
			dw[i] = 0;

		db = new double[n_neuron];
		for (int i = 0; i < n_neuron; i++)
			db[i] = 0;

		dz = new double[n_neuron];
		for (int i = 0; i < n_neuron; i++)
			dz[i] = 0;

		dx_lower = new double[n_neuron];
		for (int i = 0; i < n_neuron; i++)
			dx_lower[i] = 0;
	}

	~DJ()
	{
		delete[] dw;
		delete[] dx_lower;
		delete[] db;
		delete[] dz;
	}
};

class Layer
{
public:
	enum Activation{Sigmoid = 2, Softmax};
public:
	static double learning_rate;
	double *x;
	DJ dJ;
protected:
	int nNeuron;
	double *w;
	double *b;
	double *z;
	double *ref_input_arr;
	int input_size;
	
public:
	Layer(double* input_arr, int input_size, int n_neuron)
		: nNeuron(n_neuron), w(0), b(0), z(0), x(0), ref_input_arr(input_arr), input_size(input_size), 
		dJ(input_size, n_neuron)
	{
		w = new double[nNeuron*input_size];
		for (int i = 0; i < nNeuron*input_size; i++)
			w[i] = 0;

		b = new double[nNeuron];
		for (int i = 0; i < nNeuron; i++)
			b[i] = 0;

		z = new double[nNeuron];
		for (int i = 0; i < nNeuron; i++)
			z[i] = 0;

		x = new double[nNeuron];
		for (int i = 0; i < nNeuron; i++)
			x[i] = 0;

		initialize();
		cout << "init finished\n";
	}
	// ¼Ò¸êÀÚ
	~Layer()
	{
		delete[] w;
		delete[] x;
		delete[] b;
		delete[] z;
	}

	void updateInput(double *new_input_arr)
	{
		ref_input_arr = new_input_arr;
	}

	// generate initial weight and bias values.
	void initialize() 
	{
		srand((unsigned int)time(NULL));
		
		for (int i = 0; i < nNeuron; i++)
		{
			b[i] = (double)rand() / RAND_MAX; // (0~1)
			for (int j = 0; j < input_size; j++)
				w[i*nNeuron+j] = (double)rand() / RAND_MAX;
		}
	}

	// propate and get result of neuron
	double* forwardPropagation(Activation activation)
	{
		for (int i = 0; i < nNeuron; i++)
		{
			for (int j = 0; j < input_size; j++)
			{
				z[i] += w[i*nNeuron + j] * ref_input_arr[j];
			}
			z[i] += b[i];
		}
		switch (activation)
		{
		case Sigmoid:
			for (int i = 0; i < nNeuron; i++)
				sigmoid(z[i], x[i]);
			break;
		case Softmax:
			softmax(z, x, nNeuron);
			break;
		}

		return x;
	}

	void backPropagation(double* dJdx_upper, Activation activation, double *t = 0, bool isFirstLayer = false)
	{		
		switch (activation)
		{
		case Sigmoid: 
			if (t == 0) for (int i = 0; i < nNeuron; i++)
				dJ.dz[i] = dJdx_upper[i] * x[i] * (1 - x[i]);
			else for (int i = 0; i < nNeuron; i++) // mean_square
				dJ.dz[i] = (x[i] - t[i]) * x[i] * (1 - x[i]);
			break;

		case Softmax: // cross_entropy
			for (int i = 0; i < nNeuron; i++)
				dJ.dz[i] = x[i] - t[i];
			break;
		}
		if (!isFirstLayer)
		{
			for (int j = 0; j < input_size; j++)
			{
				dJ.dx_lower[j] = 0;
				for (int i = 0; i < nNeuron; i++)
					dJ.dx_lower[j] += dJ.dz[i] * w[nNeuron*i + j];
			}
		}

		for (int i = 0; i < nNeuron; i++)
		{
			dJ.db[i] = dJ.dz[i];
			b[i] -= learning_rate * dJ.db[i];
			for (int j = 0; j < input_size; j++)
			{
				dJ.dw[i*nNeuron + j] = dJ.dz[i] * ref_input_arr[j];
				w[i*nNeuron + j] -= learning_rate * dJ.dw[i*nNeuron + j];
			}
		}
	}
};
double Layer::learning_rate=0.1;
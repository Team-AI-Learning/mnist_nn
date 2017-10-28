#include<iostream>
#include"FileReader.hpp"
#include"Layer.h"
#include"common.h"
using namespace std;

// Simple C++ Project for MNIST NN.

// TODO: CNN
// 1. Batch applications
// 2. Batch-normalization (w/ or w/o some parameters)
// 3. Additional initialization/optimization schemes
// 4. Other activations

// TODO: SW architecture
// 1. Application of Strategy on Activation & Loss Functions
// 2. Tensor Architecture(Tensor multiplication, TensorInfo) & Interface (Element product, operator*) update
// 3. LayerInfo structure
// 4. FileReader read(maxRead);
// 5. Parallel computation on for loop by OMP

static inline double cross_entropy(Tensor& t, Tensor& x, UINT idx, LayerInfo info)
{
	double sum = 0;

	FOR3D(i, xr, xc, info.numNeurons, info.x_row, info.x_col)
	{
		sum += t.array(idx, i) * log(x[0][i][xr][xc]) + // batch = 0
			(1 - t.array(idx, i)) * log(1 - x[0][i][xr][xc]); // add an epsilon value
	}

	return -sum;
}
#define L1_NUM_NEURONS 50
#define L2_NUM_NEURONS 10
#define TEST_SIZE 10000

int main(void)
{
	Layer::learning_rate = 0.01;
	ImageReader trainData("train-images.idx3-ubyte");
	trainData.read(TEST_SIZE);
	LabelReader inputLabel("train-labels.idx1-ubyte");
	inputLabel.read(TEST_SIZE);
	ImageReader testData("t10k-images.idx3-ubyte");
	testData.read(TEST_SIZE);
	LabelReader testLabel("t10k-labels.idx1-ubyte");
	testLabel.read(TEST_SIZE);

	LayerInfo inputLayerInfo = { 1, 28, 28, 0, 0, 0, 1 };
	LayerInfo layer1Info	= { L1_NUM_NEURONS, 2, 2, 27, 27, 1, 1 };
	LayerInfo layer2Info	= { L2_NUM_NEURONS, 1, 1, 2, 2, 0, 1 };
	
	Layer inputLayer(inputLayerInfo);
	Layer layer1(&inputLayer, layer1Info);
	Layer layer2(&layer1, layer2Info);

	for (UINT i = 0; i < trainData.nImages; i++)
	{
		inputLayer.updateInput(*trainData.images, i);
		layer1.forwardPropagation(Layer::Activation::Sigmoid);
		layer2.forwardPropagation(Layer::Activation::Softmax);

		double j = cross_entropy(*inputLabel.onehot_label, layer2.getOutput(), i, layer2Info);
		cout << i << " entropy " << j << endl;

		layer2.backPropagation(inputLabel.onehot_label, i);
		layer1.backPropagation();
	}

	int cnt = 0;
	for (UINT i = 0; i < testData.nImages; i++)
	{
		inputLayer.updateInput(*testData.images, i);
		layer1.forwardPropagation(Layer::Activation::Sigmoid);
		layer2.forwardPropagation(Layer::Activation::Softmax);

		double pred_max = 0;
		int pred_max_idx = 0;
		Tensor& pred_x = layer2.getOutput();
		for (UINT ca = 0; ca < inputLabel.nCategory; ca++)
		{
			// minibatch size 1
			if (pred_x[0][ca][0][0] > pred_max)
			{
				pred_max = pred_x[0][ca][0][0];
				pred_max_idx = ca;
			}
		}

		if (testLabel.onehot_label->array(i,pred_max_idx) != 0)
		{ 
			cout << i << ": matched (" << cnt << ")\n";
			cnt++;
		}
	}
	cout << "Total Test: " << testData.nImages << ", count " << cnt << endl;

	return 0;
}
#include<iostream>
#include"FileReader.hpp"
#include"ConvLayer.h"
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

static inline double cross_entropy(Tensor& t, double****& x, UINT idx, ConvLayerInfo info)
{
	double sum = 0;
	FOR(i, info.numNeurons)
	{
		FOR3D(xd, xr, xc, info.x_cha, info.x_row, info.x_col)
		{
			sum += t.array(idx, i) * log(x[i][xd][xr][xc]) + // batch = 0
				(1 - t.array(idx, i)) * log(1 - x[i][xd][xr][xc]); // add an epsilon value
		}
	}
	/*
	FOR4D(i, xd, xr, xc, info.numNeurons, info.x_cha, info.x_row, info.x_col)
	{
		
	}
	*/
	return -sum;
}
#define L1_NUM_NEURONS 50
#define L2_NUM_NEURONS 10

int main(void)
{
	ConvLayer::learning_rate = 0.01;
	ImageReader trainData("train-images.idx3-ubyte");
	trainData.read();
	LabelReader inputLabel("train-labels.idx1-ubyte");
	inputLabel.read();
	ImageReader testData("t10k-images.idx3-ubyte");
	testData.read();
	LabelReader testLabel("t10k-labels.idx1-ubyte");
	testLabel.read();

	ConvLayerInfo inputLayerInfo= { 
		1, 
		1, 28, 28,	// x 
		0, 0, 0, 0, // w
		0 };
	ConvLayerInfo layer1Info	= { 
		L1_NUM_NEURONS, 
		1, 1, 1, 
		1, 1, 28, 28, 
		0 };
	ConvLayerInfo layer2Info	= { 
		L2_NUM_NEURONS, 
		1, 1, 1, 
		1, 1, 1, 1, 
		0 };
	
	ConvLayer inputLayer(inputLayerInfo);
	ConvLayer layer1(&inputLayer, layer1Info);
	ConvLayer layer2(&layer1, layer2Info);

	for (UINT i = 0; i < trainData.nImages; i++)
	{
		inputLayer.updateInput(*trainData.images, i);
		layer1.forwardPropagation(ConvLayer::Activation::Sigmoid);
		layer2.forwardPropagation(ConvLayer::Activation::Softmax);

		double j = cross_entropy(*inputLabel.onehot_label, layer2.getOutput(), i, layer2Info);
		cout << i << " entropy " << j << endl;

		layer2.backPropagation(inputLabel.onehot_label, i);
		layer1.backPropagation();
	}

	int cnt = 0;
	for (UINT i = 0; i < testData.nImages; i++)
	{
		inputLayer.updateInput(*testData.images, i);
		layer1.forwardPropagation(ConvLayer::Activation::Sigmoid);
		layer2.forwardPropagation(ConvLayer::Activation::Softmax);

		double pred_max = 0;
		int pred_max_idx = 0;
		double****& pred_x = layer2.getOutput();
		for (UINT ca = 0; ca < inputLabel.nCategory; ca++)
		{
			if (pred_x[ca][0][0][0] > pred_max)
			{
				pred_max = pred_x[ca][0][0][0];
				pred_max_idx = ca;
			}
		}

		if (testLabel.onehot_label->array(i, pred_max_idx) != 0)
		{ 
			cout << i << ": matched (" << cnt << ")\n";
			cnt++;
		}
	}
	cout << "Total Test: " << testData.nImages << ", count " << cnt << endl;

	return 0;
}
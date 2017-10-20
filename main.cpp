#include<iostream>
#include"FileReader.hpp"
#include"Layer.h"
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

static inline double cross_entropy(Tensor& t, Tensor& x, int idx, int size)
{
	double sum = 0;

	for (int i = 0; i < size; i++)
	{
		sum += t.array(idx, i) * log(x.array(i)) + 
			(1 - t.array(idx, i)) * log(1 - x.array(i)); // add an epsilon value
	}

	return -sum;
}
#define L1_NUM_NEURONS 50
#define L2_NUM_NEURONS 10

int main(void)
{
	Layer::learning_rate = 0.01;
	ImageReader trainData("train-images.idx3-ubyte");
	trainData.read();
	LabelReader inputLabel("train-labels.idx1-ubyte");
	inputLabel.read();
	ImageReader testData("t10k-images.idx3-ubyte");
	testData.read();
	LabelReader testLabel("t10k-labels.idx1-ubyte");
	testLabel.read();

	Layer inputLayer(trainData.getImageSize());
	Layer layer1(&inputLayer, L1_NUM_NEURONS);
	Layer layer2(&layer1, L2_NUM_NEURONS);
	for (int i = 0; i < trainData.nImages; i++)
	{
		inputLayer.updateInput(*trainData.images, i);
		layer1.forwardPropagation(Layer::Activation::Sigmoid);
		layer2.forwardPropagation(Layer::Activation::Softmax);

		double j = cross_entropy(*inputLabel.onehot_label, layer2.getOutput(), i, L2_NUM_NEURONS);
		cout << i << " entropy " << j << endl;

		layer2.backPropagation(inputLabel.onehot_label, i);
		layer1.backPropagation();
	}

	int cnt = 0;
	for (int i = 0; i < testData.nImages; i++)
	{
		inputLayer.updateInput(*testData.images, i);
		layer1.forwardPropagation(Layer::Activation::Sigmoid);
		layer2.forwardPropagation(Layer::Activation::Softmax);

		double pred_max = 0;
		int pred_max_idx = 0;
		Tensor& pred_x = layer2.getOutput();
		for (int ii = 0; ii < inputLabel.nCategory; ii++)
		{
			if (pred_x.array(ii) > pred_max)
			{
				pred_max = pred_x.array(ii);
				pred_max_idx = ii;
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
#include<iostream>
#include"FileReader.hpp"
#include"LayerBase.hpp"
using namespace std;

double cross_entropy(double *t, double* x, int size)
{
	double sum = 0;

	for (int i = 0; i < size; i++)
	{
		sum += t[i] * log(x[i]) + (1 - t[i]) * log(1 - x[i]); // add an epsilon value
	}

	return -sum;
}

int main(void)
{
	ImageReader inputImage("train-images.idx3-ubyte");
	inputImage.read();
	LabelReader inputLabel("train-labels.idx1-ubyte");
	inputLabel.read();
	ImageReader testImage("t10k-images.idx3-ubyte");
	testImage.read();
	LabelReader testLabel("t10k-labels.idx1-ubyte");
	testLabel.read();
	
	LayerBase inputLayer = LayerBase::CreateInputLayerBase(
		inputImage.images[0], inputImage.getImageSize());

	LayerBase layer1(&inputLayer, 50);
	LayerBase layer2(&layer1, 10);
	for (int i = 0; i < 60000; i++)
	{
		inputLayer.updateInput(inputImage.images[i]);
		layer1.forwardPropagation(LayerBase::Activation::Relu);
		layer2.forwardPropagation(LayerBase::Activation::Softmax);

		double j = cross_entropy(inputLabel.ans[i], layer2.getOutput(), layer2.numNeurons);
		cout << i << " entropy " << j << endl;

		layer2.backPropagation(LayerBase::Activation::Softmax, inputLabel.ans[i]);
		layer1.backPropagation(LayerBase::Activation::Relu);
	}

	double cnt = 0;
	for (int i = 0; i < 1000; i++)
	{
		inputLayer.updateInput(testImage.images[i]);
		layer1.forwardPropagation(LayerBase::Activation::Relu);
		layer2.forwardPropagation(LayerBase::Activation::Softmax);

		double pred_max = 0;
		int pred_max_idx = 0;
		double *pred_x = layer2.getOutput();
		for (int ii = 0; ii < inputLabel.nCategory; ii++)
		{
			if (pred_x[ii] > pred_max)
			{
				pred_max = pred_x[ii];
				pred_max_idx = ii;
			}
		}

		if (testLabel.ans[i][pred_max_idx] != 0)
			cnt += 1.0;
	}
	cout << "count " << cnt;

	return 0;
}
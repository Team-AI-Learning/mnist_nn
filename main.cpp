#include<iostream>
#include"FileReader.hpp"
#include"LayerBase.hpp"
using namespace std;

double cross_entropy(Tensor& t, Tensor& x, int idx, int size)
{
	double sum = 0;

	for (int i = 0; i < size; i++)
	{
		sum += t.array(idx, i) * log(x.array(i)) + (1 - t.array(idx, i)) * log(1 - x.array(i)); // add an epsilon value
	}

	return -sum;
}

int main(void)
{
	cout << "CNN\n";
	ImageReader trainData("train-images.idx3-ubyte");
	trainData.read();
	
	LabelReader inputLabel("train-labels.idx1-ubyte");
	inputLabel.read();
	ImageReader testData("t10k-images.idx3-ubyte");
	testData.read();
	LabelReader testLabel("t10k-labels.idx1-ubyte");
	testLabel.read();
	int imageSize = trainData.images->J*trainData.images->K;
	LayerBase inputLayer(imageSize);
	
	inputLayer.updateInput(*trainData.images, 0);

	LayerBase layer1(&inputLayer, 50);
	LayerBase layer2(&layer1, 10);
	for (int i = 0; i < 60000; i++)
	{
		inputLayer.updateInput(*trainData.images, i);
		layer1.forwardPropagation(LayerBase::Activation::Sigmoid);
		layer2.forwardPropagation(LayerBase::Activation::Softmax);

		double j = cross_entropy(*inputLabel.onehot_label, layer2.getOutput(), i, layer2.numNeurons);
		cout << i << " entropy " << j << endl;

		layer2.backPropagation(LayerBase::Activation::Softmax, i, inputLabel.onehot_label);
		layer1.backPropagation(LayerBase::Activation::Sigmoid);
	}

	double cnt = 0;
	for (int i = 0; i < 1000; i++)
	{
		inputLayer.updateInput(*testData.images, i);
		layer1.forwardPropagation(LayerBase::Activation::Sigmoid);
		layer2.forwardPropagation(LayerBase::Activation::Softmax);

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

		double tmp = testLabel.onehot_label->array(i, pred_max_idx);
		if (testLabel.onehot_label->array(i,pred_max_idx) != 0)
			cnt += 1.0;
	}
	printf("layer %x\n", &layer2);
	printf("layer input %x\n", &inputLayer);
	cout << "count " << cnt;

	return 0;
}
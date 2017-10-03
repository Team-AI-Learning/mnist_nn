#include<iostream>
#include"FileReader.hpp"
#include"Layer.hpp"
using namespace std;

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
	int image_size = inputImage.nCol*inputImage.nRow;
	Layer::learning_rate = 0.1;
	Layer layer1(inputImage.images[0], image_size, 256);
	Layer layer1_1(layer1.x, 256, 256);
	Layer layer2(layer1_1.x, 256, 10);
	for (int i = 0; i < 600; i++)
	{
		while(true)
		{
			layer1.updateInput(inputImage.images[i]);
			layer1.forwardPropagation(Layer::Activation::Sigmoid);
			layer1_1.forwardPropagation(Layer::Activation::Sigmoid);
			layer2.forwardPropagation(Layer::Activation::Softmax);

			// J
			double j = cross_entropy(inputLabel.ans[i], layer2.x, inputLabel.nCategory);
			if (i % 10 == 0)
				cout << i << " cross entropy: " << j << endl;
			
			if (j < 0.01)
				break;

			// Back
			layer2.backPropagation(0, Layer::Activation::Softmax, inputLabel.ans[i]);
			layer1_1.backPropagation(layer2.dJ.dx_lower, Layer::Activation::Sigmoid);
			layer1.backPropagation(layer1_1.dJ.dx_lower, Layer::Activation::Sigmoid, 0, true);
		}
	}

	double cnt = 0;
	for (int i = 0; i < 100; i++)
	{
		layer1.updateInput(inputImage.images[i]);
		layer1.forwardPropagation(Layer::Activation::Sigmoid);
		layer1_1.forwardPropagation(Layer::Activation::Sigmoid);
		layer2.forwardPropagation(Layer::Activation::Softmax);

		double pred_max = 0;
		int pred_max_idx = 0;
		for (int ii = 0; ii < inputLabel.nCategory; ii++)
		{
			if (layer2.x[ii] > pred_max)
			{
				pred_max = layer2.x[ii];
				pred_max_idx = ii;
			}
		}

		if (inputLabel.ans[i][pred_max_idx] != 0)
			cnt += 1.0;
	}
	cout << "count " << cnt;

	getchar();
	return 0;
}
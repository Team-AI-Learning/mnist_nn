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
// 1. Tensor Architecture(Tensor multiplication, TensorInfo) & Interface (Element product, operator*) update

static inline double cross_entropy(Tensor<>& t, Tensor<>& x, UINT idx, LayerInfo info)
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
#define TEST_SIZE 100


namespace OMP
{
	void init(int num_threads)
	{
		omp_set_num_threads(num_threads);
	}

	void display()
	{
		printf(
			"num procs : %d\n"
			"max threads : %d\n"
			, omp_get_num_procs(), omp_get_max_threads());
	}
}

int main(void)
{
	// OMP
	OMP::init(8);
	OMP::display();
	
	Layer::learning_rate = 0.01;
	
	ImageReader trainData("train-images.idx3-ubyte");
	trainData.read(TEST_SIZE);
	LabelReader inputLabel("train-labels.idx1-ubyte");
	inputLabel.read(TEST_SIZE);
	ImageReader testData("t10k-images.idx3-ubyte");
	testData.read(TEST_SIZE);
	LabelReader testLabel("t10k-labels.idx1-ubyte");
	testLabel.read(TEST_SIZE);

	typedef LayerInfo::Activation Act;
	// activation, channel, x size, w size, stride, batch size
	LayerInfo inputLayerInfo = { Act::None, 1, 28, 28, 0, 0, 0, 1 };
	LayerInfo layer1Info = { Act::ReLU, L1_NUM_NEURONS, 14, 14, 14, 14, 1, 1 };
	LayerInfo pooling1Info = { Act::MaxPooling, L1_NUM_NEURONS, 7, 7 }; // 1/4 size of original
	LayerInfo layer2Info = { Act::Softmax, L2_NUM_NEURONS, 1, 1, 7, 7, 0, 1 };

	// TensorFlow.addLayer() -> linkedlist<Layer>
	Layer inputLayer(inputLayerInfo);
	Layer layer1(&inputLayer, layer1Info);
	Layer pooling1(&layer1, pooling1Info, true);
	Layer layer2(&pooling1, layer2Info);

	double fptime = 0;
	double bptime = 0;
	// TensorFlow.run()
	TIMESTAMP_START(trainingTime)
	for (UINT i = 0; i < trainData.nImages; i++)
	{
		inputLayer.updateInput(*trainData.images, i);
		TIMESTAMP_START(fp)
		layer1.forwardPropagation();
		pooling1.maxPooling();
		layer2.forwardPropagation();
		TIMESTAMP_SAVE(fp, fptime)

		double j = cross_entropy(*inputLabel.onehot_label, layer2.getOutput(), i, layer2Info);
		cout << i << " entropy " << j << endl;
		TIMESTAMP_START(bp)
		layer2.backPropagation(inputLabel.onehot_label, i);
		pooling1.backPropagation();
		layer1.backPropagation();
		TIMESTAMP_SAVE(bp, bptime)
	}
	TIMESTAMP_END(trainingTime, "elapsed time on Training")
	TIMESTAMP_PRINT(fptime, "forward propagation time")
	TIMESTAMP_PRINT(bptime, "back propagation time")

	int cnt = 0;
	for (UINT i = 0; i < testData.nImages; i++)
	{
		inputLayer.updateInput(*testData.images, i);
		layer1.forwardPropagation();
		pooling1.maxPooling();
		layer2.forwardPropagation();

		double pred_max = 0;
		int pred_max_idx = 0;
		Tensor<>& pred_x = layer2.getOutput();
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
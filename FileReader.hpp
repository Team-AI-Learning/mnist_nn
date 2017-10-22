#ifndef __IMAGE_READER__
#define __IMAGE_READER__

#define DEBUG 1

#include<iostream>
#include<fstream>
#include"Tensor.hpp"
#include"common.h"
using namespace std;

static const int Byte4 = 4;
static const int Byte1 = 1;
static int ReverseInt(int i) // byte translation
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;

	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

// MNIST Image & Label Reader for CNN
class ImageReader
{
public:
	const string TAG = "[ImageReader] ";
	UINT nDummy;
	UINT nImages; // 60000
	UINT nChannel; // 1 for mnist nn.
	UINT nRow; // 28
	UINT nCol; // 28
	Tensor *images; // [numImages][channel][row][col]
	ifstream in;
private:
	int maxRead;
public:
	ImageReader(string fileName) :
		nDummy(0), nImages(0), nChannel(1), nRow(0), nCol(0), images(0), maxRead(0)
	{
		in.open(fileName, ios::binary);
	}
	~ImageReader() 
	{
		if (images != 0)
		{
			delete images;
			images = 0;
		}
		in.close();
	}

	// parameter maxRead is for Debugging
	void read(int _maxRead = 0)
	{
		maxRead = _maxRead;
		if (!in.is_open())
		{
			cout << "image read fail.\n";
			return;
		}
#ifdef DEBUG
		printf(string(TAG).append("read() start\n").c_str());
#endif
		read_header();
		read_pixels();
#ifdef DEBUG
		printf(string(TAG).append("read() end\n").c_str());
#endif
	}

	unsigned int getImageSize() { return nRow*nCol; }

	void printImage(int idx)
	{
		double***& img = (*images)[0][idx];
		
		FOR2D(ch, r, nChannel, nRow) 
		{
			FOR(c, nCol)
			{
				UINT dot = (UINT)(img[ch][r][c] * 10);
				cout << dot << " ";
			}
			cout << endl;
		}
	}

protected:
	// Read header
	void read_header()
	{
		in.read((char*)&nDummy, Byte4); // Magic number
		in.read((char*)&nImages, Byte4); // nImages
		in.read((char*)&nRow, Byte4);
		in.read((char*)&nCol, Byte4);

		nDummy = ReverseInt(nDummy);
		nImages = ReverseInt(nImages);
		if (maxRead != 0) nImages = maxRead;
		nRow = ReverseInt(nRow);
		nCol = ReverseInt(nCol);
#if DEBUG
		printf(string(TAG).append("dummy %d, nImage %d, nRow %d, nCol %d\n").c_str(),
			nDummy, nImages, nRow, nCol);
#endif
		images = new Tensor(1, nImages, nChannel, nRow, nCol);
	}
	// Read an image
	void read_pixels()
	{
		FOR4D(img, channel, row, col, 
			nImages, nChannel, nRow, nCol)
		{
			unsigned char buff = 0;
			in.read((char*)&buff, Byte1);
			(*images)[0][img][channel][row][col] = (double)buff / 255.0;
		}
	}
};

class LabelReader
{
public:
	const string TAG = "[LabelReader] ";
	UINT nDummy;
	UINT nLabels; // 60000
	Tensor* label; // answer of the image
	UINT nCategory; // 10
	Tensor* onehot_label;
	ifstream in;

public:
	LabelReader(string fileName, int n_category = 10)
		: label(0), nLabels(0), nCategory(n_category), onehot_label(0)
	{
		in.open(fileName, ios::binary);
	}

	~LabelReader()
	{
		if (label != 0)
		{
			delete label;
			label = 0;
		}
			
		if (onehot_label != 0)
		{
			delete onehot_label;
			onehot_label = 0;
		}
		in.close();
	}

	void read(int _maxRead = 0)
	{
		if (!in.is_open())
		{
			printf(string(TAG).append("read() failed.\n").c_str());
			return;
		}
#ifdef DEBUG
		printf(string(TAG).append("read() start\n").c_str());
#endif
		in.read((char*)&nDummy, Byte4); // dummy
		in.read((char*)&nLabels, Byte4);
		nDummy = ReverseInt(nDummy);
		nLabels = ReverseInt(nLabels); 
		if (_maxRead != 0) nLabels = _maxRead;
		label = new Tensor(nLabels);
		for (UINT i = 0; i < nLabels; i++)
		{
			unsigned char buff = 0;
			in.read((char*)&buff, Byte1);
			label->array(i) = (double)buff;
		}
		generateOnehot();
#ifdef DEBUG
		printf(string(TAG).append("dummy %d label %d\n").c_str(), nDummy, nLabels);
		printf(string(TAG).append("read() end\n").c_str());
#endif
	}

	void printLabel(UINT idx, bool answerVector = false)
	{
		cout << "label " << label->array(idx) << endl;
		if (answerVector)
		{
			cout << "onehot_label ";
			for (UINT i = 0; i < nCategory; i++)
				cout << onehot_label->array(idx,i) << " ";
			cout << endl;
		}			
	}

protected:
	void generateOnehot()
	{
		if (onehot_label != 0)
		{
			printf(string(TAG).append("error! already generated!!").c_str());
			return;
		}
		onehot_label = new Tensor(nLabels, nCategory);

		FOR(i, nLabels)
		{
			FOR(j, nCategory)
				onehot_label->array(i,j) = 0;
			
			onehot_label->array(i, (UINT)label->array(i) ) = 1;
		}
	}
};
#endif
#ifndef __IMAGE_READER__
#define __IMAGE_READER__

#define DEBUG 1

#include<iostream>
#include<fstream>
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

// MNIST Image & Label Reader
class ImageReader
{
public:
	const string TAG = "[ImageReader]";
	int nDummy;
	int nImages; // 60000
	int nRow; // 28
	int nCol; // 28
	double **images;
	ifstream in;
public:
	ImageReader(string fileName) :
		nDummy(0), nImages(0), nRow(0), nCol(0), images(0)
	{
		in.open(fileName, ios::binary);
	}
	~ImageReader() 
	{
		in.close();
		for (int i = 0; i < nImages; i++)
		{
			delete[] images[i];
			images[i] = 0;
		}
		delete[] images;
		images = 0;
	}

	void read()
	{
		if (!in.is_open() || images != 0)
		{
			cout << "image read fail.\n";
			return;
		}
#ifdef DEBUG
		printf(string(TAG).append(" read() start\n").c_str());
#endif
		read_header();
		read_pixels();
#ifdef DEBUG
		printf(string(TAG).append(" read() end\n").c_str());
#endif
	}

	void printImage(int idx)
	{
		double*& img = images[idx];
		for (int r = 0; r < nRow; r++)
		{
			for (int c = 0; c < nCol; c++)
			{
				int dot = (int)(img[r*nRow + c] * 10);
				cout << dot << " ";
			}
			cout << endl;
		}
		cout << "press enter.";
		getchar();
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
		nImages = ReverseInt(nImages); nImages = 10000;
		nRow = ReverseInt(nRow);
		nCol = ReverseInt(nCol);
#if DEBUG
		printf(string(TAG).append(" dummy %d, nImage %d, nRow %d, nCol %d\n").c_str(),
			nDummy, nImages, nRow, nCol);
#endif
	}

	// Read an image
	void read_pixels()
	{
		images = new double*[nImages];
		for (int i = 0; i < nImages; i++)
			images[i] = new double[nRow*nCol];

		for (int i = 0; i < nImages; ++i)
		{
			for (int r = 0; r < nRow; ++r) for (int c = 0; c < nCol; ++c)
			{
				unsigned char buff = 0;
				in.read((char*)&buff, Byte1);
				images[i][(nRow*r) + c] = (double)buff / 255.0;
			}
		}
	}
};


class LabelReader
{
public:
	const string TAG = "[LabelReader]";
	int nDummy;
	int nLabels; // 60000
	double* label; // answer of the image
	int nCategory; // 10
	double** ans;

	ifstream in;
public:
	LabelReader(string fileName, int n_category = 10)
		: label(0), ans(0), nLabels(0), nCategory(n_category)
	{
		in.open(fileName, ios::binary);
	}
	~LabelReader()
	{
		in.close();
		delete[] label;
		label = 0;

		for (int i = 0; i < nLabels; i++)
		{
			delete[] ans[i];
			ans[i] = 0;
		}
		delete[] ans;
		ans = 0;
	}

	void read()
	{
		if (!in.is_open() || label != 0)
		{
			cout << "read fail\n";
			return;
		}
#ifdef DEBUG
		printf(string(TAG).append(" read() start\n").c_str());
#endif
		in.read((char*)&nDummy, Byte4); // dummy
		in.read((char*)&nLabels, Byte4);
		nDummy = ReverseInt(nDummy);
		nLabels = ReverseInt(nLabels);
		label = new double[nLabels];
		for (int i = 0; i < nLabels; i++)
		{
			unsigned char buff = 0;
			in.read((char*)&buff, Byte1);
			label[i] = (double)buff;
		}
		generateAnswerVector();
#ifdef DEBUG
		printf(string(TAG).append(" dummy %d label %d\n").c_str(), nDummy, nLabels);
		printf(string(TAG).append(" read() end\n").c_str());
#endif
	}
protected:
	void generateAnswerVector()
	{
		if (ans != 0)
		{
			cout << "cannot generate answer vector.\n";
			return;
		}
		
		ans = new double*[nLabels];
		for (int i = 0; i < nLabels; i++)
		{
			ans[i] = new double[nCategory];

			for (int j = 0; j < nCategory; j++)
				ans[i][j] = 0;
			
			ans[i][(int)label[i]] = 1;
		}
	}
};
#endif
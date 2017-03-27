#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <string>
#include <cstring>
#include <unordered_map>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <random>
#include <assert.h>

#define _SIGMOID 1
#define _TANH 2
#define _RELU 3
#define _LINEAR 4
#define _SOFTMAX 5

#define _CONV 10
#define _POOL_MAX 11
#define _POOL_AVERAGE 12
#define _POOL_LINEAR 13
#define _FC 14

#define LAYER_SIZE 100
#define LAYER_WIDTH 101
#define LAYER_HEIGHT 102
#define LAYER_DEPTH 103

#define CONNECTION_TYPE 200
#define CONNECTION_WIDTH 201
#define CONNECTION_STRIDE 202
#define CONNECTION_PADDING 203
#define CONNECTION_RELU 204
#define CONNECTION_SOFTMAX 205

#define CNN_TOTAL_LAYER_NUM 300
#define CNN_TRAINING_RATE 301
#define CNN_BATCH_SIZE 302
#define CNN_MOMENTUM_FACTOR 303
#define CNN_REGULARIZATION_STRENGTH 304
#define CNN_DROPOUT_RATE 305

using namespace std;

class LAYER;
class CONNECTION;
class CONVOLUTIONAL_NEURAL_NETWORK;

class LAYER
{
public:
	bool is3d;
	int size;
	int width, height, depth;
	double *neuron;
	double *delta;
	bool *isdropout;
	CONNECTION *prev_connection, *next_connection;
	LAYER();
	~LAYER();
	void initialization(unordered_map<int, int>*);
	double getval(const int, const int, const int);
	void setval(const int, const int, const int, const double);
	void calcoutputdelta(const double*);
	double getdelta(const int, const int, const int);
	void adddelta(const int, const int, const int, const double);
};

LAYER::LAYER()
{
	neuron = delta = NULL;
	isdropout = NULL;
	prev_connection = next_connection = NULL;
}

LAYER::~LAYER()
{
	if (neuron != NULL) delete []neuron;
	if (delta != NULL) delete []delta;
	if (isdropout != NULL) delete []isdropout;
}

void LAYER::initialization(unordered_map<int, int> *param)
{
	int i;
	assert(param != NULL);
	if (param->find(LAYER_SIZE) != param->end())
	{
		size = (*param)[LAYER_SIZE];
		assert(size > 0);
		neuron = new double[size];
		delta = new double[size];
		isdropout = new bool[size];
		is3d = false;
	}
	else
	{
		assert(param->find(LAYER_WIDTH) != param->end() && param->find(LAYER_HEIGHT) != param->end() && param->find(LAYER_DEPTH) != param->end());
		width = (*param)[LAYER_WIDTH];
		height = (*param)[LAYER_HEIGHT];
		depth = (*param)[LAYER_DEPTH];
		assert(width > 0 && height > 0 && depth > 0);
		size = width*height*depth;
		neuron = new double[size];
		delta = new double[size];
		isdropout = new bool[size];
		is3d = true;		
	}
	for (i = 0; i < size; i++)
	{
		neuron[i] = 0;
		delta[i] = 0;
		isdropout[i] = false;
	}
}

double LAYER::getval(const int w, const int h, const int d)
{
	assert(is3d && d >= 0 && d < depth);
	if (w < 0 || w >= width || h < 0 || h >= height) return 0;
	return neuron[w*height*depth+h*depth+d];
}

void LAYER::setval(const int w, const int h, const int d, const double newval)
{
	assert(is3d && w < width && w >= 0 && h < height && h >= 0 && d < depth && d >= 0);
	neuron[w*height*depth+h*depth+d] = newval;
}

class CONNECTION
{
public:
	int type;
	int size1, size2;
	int width, stride, padding;
	double **weight;
	double *bias;
	double **dw;
	double *db;
	double relu_param, softmax_param;
	LAYER *prev_layer, *next_layer;
	CONNECTION();
	~CONNECTION();
	void initialization(unordered_map<int, double>*, LAYER*, LAYER*);
	double getweight(const int, const int, const int, const int);
	void adddw(const int, const int, const int, const int, const double);
	void weight_initialization(mt19937&);
	void forward_propagation(const double, const bool);
	void back_propagation();
};

CONNECTION::CONNECTION()
{
	weight = dw = NULL;
	bias = db = NULL;
	prev_layer = next_layer = NULL;
	size1 = size2 = width = stride = padding = 0;
	relu_param = 0;
	softmax_param = 0;
}

CONNECTION::~CONNECTION()
{
	int i;
	if (weight != NULL)
	{
		for (i = 0; i < size1; i++)
		{
			delete []weight[i];
			delete []dw[i];
		}
		delete []weight;
		delete []dw;
	}
	if (bias != NULL) delete []bias;
	if (db != NULL) delete []db;
}

void CONNECTION::initialization(unordered_map<int, double> *param, LAYER *prev, LAYER *next)
{
	int i, j;
	assert(param != NULL && param->find(CONNECTION_TYPE) != param->end());
	type = (*param)[CONNECTION_TYPE];
	prev_layer = prev;
	next_layer = next;
	prev->next_connection = this;
	next->prev_connection = this;
	size1 = size2 = 0;
	if (type == _RELU)
	{
		relu_param = 0;
		if (param->find(CONNECTION_RELU) != param->end()) relu_param = (*param)[CONNECTION_RELU];
		assert(relu_param >= 0 && relu_param < 1);
	}
	else if (type == _SOFTMAX)
	{
		softmax_param = 1;
		if (param->find(CONNECTION_SOFTMAX) != param->end()) relu_param = (*param)[CONNECTION_SOFTMAX];
		assert(softmax_param > 0);
	}
	else if (type == _CONV)
	{
		assert(prev->is3d && next->is3d);
		width = 3;
		stride = 1;
		padding = 1;
		if (param->find(CONNECTION_WIDTH) != param->end()) width = (*param)[CONNECTION_WIDTH];
		if (param->find(CONNECTION_STRIDE) != param->end()) stride = (*param)[CONNECTION_STRIDE];
		if (param->find(CONNECTION_PADDING) != param->end()) padding = (*param)[CONNECTION_PADDING];
		assert(width > 0 && stride > 0 && padding >= 0);
		size1 = width*width*prev->depth;
		size2 = next->depth;
	}
	else if (type == _POOL_MAX || type == _POOL_AVERAGE)
	{
		assert(prev->is3d && next->is3d && prev->depth > 0 && prev->depth == next->depth);
		width = 2;
		stride = 2;
		if (param->find(CONNECTION_WIDTH) != param->end()) width = (*param)[CONNECTION_WIDTH];
		if (param->find(CONNECTION_STRIDE) != param->end()) stride = (*param)[CONNECTION_STRIDE];
		assert(width > 0 && stride > 0);		
	}
	else if (type == _POOL_LINEAR)
	{
		assert(prev->is3d && next->is3d && prev->depth > 0 && prev->depth == next->depth && next->height == 1);
		size1 = prev->width*prev->height;
		size2 = next->width*next->depth;
	}
	else if (type == _FC)
	{
		size1 = prev->size;
		size2 = next->size;
	}
	if (size1 > 0 && size2 > 0)
	{
		weight = new double*[size1];
		dw = new double*[size1];
		for (i = 0; i < size1; i++)
		{
			weight[i] = new double[size2];
			dw[i] = new double[size2];
			for (j = 0; j < size2; j++)
			{
				weight[i][j] = 0;
				dw[i][j] = 0;
			}
		}
		bias = new double[size2];
		db = new double[size2];
		for (j = 0; j < size2; j++)
		{
			bias[j] = 0;
			db[j] = 0;
		}
	}
}

double CONNECTION::getweight(const int w, const int h, const int d1, const int d2)
{
	assert(weight != NULL && type == _CONV && w >= 0 && w < width && h >= 0 && h < width && d1 >= 0 && d1 < prev_layer->depth && d2 >= 0 && d2 < size2);
	return weight[w*width*prev_layer->depth+h*prev_layer->depth+d1][d2];
}

void CONNECTION::adddw(const int w, const int h, const int d1, const int d2, const double addval)
{
	assert(weight != NULL && type == _CONV && w >= 0 && w < width && h >= 0 && h < width && d1 >= 0 && d1 < prev_layer->depth && d2 >= 0 && d2 < size2);
	weight[w*width*prev_layer->depth+h*prev_layer->depth+d1][d2] += addval;
}

class CONVOLUTIONAL_NEURAL_NETWORK
{
	int total_layer_num;
	int connection_num;
	LAYER *layer;
	LAYER *input_layer;
	LAYER *output_layer;
	CONNECTION *connection;
	double training_rate;
	int batch_size;
	double momentum_factor;
	double regularization_strength;
	double dropout_rate;
	long long training_cnt;
	mt19937 mt19937generator;
	void forward_propagation(const double*, double*, const bool);
	void back_propagation();
	void update(CONNECTION*);
	void dropout();
public:
	CONVOLUTIONAL_NEURAL_NETWORK();
	CONVOLUTIONAL_NEURAL_NETWORK(unordered_map<int, int>*, unordered_map<int, double>*, unordered_map<int, double>*);
	CONVOLUTIONAL_NEURAL_NETWORK(ifstream&);
	~CONVOLUTIONAL_NEURAL_NETWORK();
	void initialization(unordered_map<int, int>*, unordered_map<int, double>*, unordered_map<int, double>*);
	void initialization(ifstream&);
	void weight_initialization();
	void train(const double*, double*, const double*);
	void train(const double*);
	void test(const double*, double*);
	void print(ofstream&);
	void copyto(CONVOLUTIONAL_NEURAL_NETWORK&);
	void getneuronfromlayer(const int, double*);
};

CONVOLUTIONAL_NEURAL_NETWORK::CONVOLUTIONAL_NEURAL_NETWORK()
{
	layer = input_layer = output_layer = NULL;
	connection = NULL;
}

CONVOLUTIONAL_NEURAL_NETWORK::CONVOLUTIONAL_NEURAL_NETWORK(unordered_map<int, int> *layer_param, unordered_map<int, double> *connection_param, unordered_map<int, double> *param)
{
	layer = input_layer = output_layer = NULL;
	connection = NULL;
	initialization(layer_param, connection_param, param);
}

void CONVOLUTIONAL_NEURAL_NETWORK::initialization(unordered_map<int, int> *layer_param, unordered_map<int, double> *connection_param, unordered_map<int, double> *param)
{
	int i;
	assert(layer_param != NULL && connection_param != NULL && param != NULL && param->find(CNN_TOTAL_LAYER_NUM) != param->end());
	assert(layer == NULL && input_layer == NULL && output_layer == NULL && connection == NULL);
	total_layer_num = (*param)[CNN_TOTAL_LAYER_NUM];
	connection_num = total_layer_num-1;
	assert(connection_num > 0);
	training_rate = 0.1;
	batch_size = 1;
	momentum_factor = 0;
	regularization_strength = 0;
	dropout_rate = 1;
	training_cnt = 0;
	mt19937generator = mt19937(time(0));
	if (param->find(CNN_TRAINING_RATE) != param->end()) training_rate = (*param)[CNN_TRAINING_RATE];
	if (param->find(CNN_BATCH_SIZE) != param->end()) batch_size = (*param)[CNN_BATCH_SIZE];
	if (param->find(CNN_MOMENTUM_FACTOR) != param->end()) momentum_factor = (*param)[CNN_MOMENTUM_FACTOR];
	if (param->find(CNN_REGULARIZATION_STRENGTH) != param->end()) regularization_strength = (*param)[CNN_REGULARIZATION_STRENGTH];
	if (param->find(CNN_DROPOUT_RATE) != param->end()) dropout_rate = (*param)[CNN_DROPOUT_RATE];
	layer = new LAYER[total_layer_num];
	connection = new CONNECTION[connection_num];
	input_layer = layer;
	output_layer = layer+total_layer_num-1;
	for (i = 0; i < total_layer_num; i++)
	{
		layer[i].initialization(layer_param+i);
	}
	for (i = 0; i < connection_num; i++)
	{
		connection[i].initialization(connection_param+i, layer+i, layer+i+1);
	}
	assert(!output_layer->is3d);
	assert(output_layer->prev_connection->type == _SIGMOID || output_layer->prev_connection->type == _TANH || output_layer->prev_connection->type == _RELU || output_layer->prev_connection->type == _LINEAR || output_layer->prev_connection->type == _SOFTMAX);
	weight_initialization();
}

CONVOLUTIONAL_NEURAL_NETWORK::~CONVOLUTIONAL_NEURAL_NETWORK()
{
	if (layer != NULL) delete []layer;
	if (connection != NULL) delete []connection;
}

void CONNECTION::weight_initialization(mt19937 &mt19937generator)
{
	int i, j;
	normal_distribution<double> distribution(0.0, 1.0);
	for (i = 0; i < size1; i++)
		for (j = 0; j < size2; j++)
		{
			double randnum = distribution(mt19937generator);
			weight[i][j] = randnum/sqrt(size1);
		}
}

void CONVOLUTIONAL_NEURAL_NETWORK::weight_initialization()
{
	assert(layer != NULL && input_layer != NULL && output_layer != NULL && connection != NULL);
	int i;
	for (i = 0; i < connection_num; i++)
	{
		if (connection[i].weight != NULL) connection[i].weight_initialization(mt19937generator);
	}
}

void CONNECTION::forward_propagation(const double dropout_rate, const bool testing)
{
	int i, j, k, l;
	if (type == _SIGMOID)
	{
		assert(prev_layer->size > 0 && prev_layer->size == next_layer->size);
		for (i = 0; i < prev_layer->size; i++)
		{
			next_layer->neuron[i] = 1/(1+exp(-(prev_layer->neuron[i])));
		}
	}
	else if (type == _TANH)
	{
		assert(prev_layer->size > 0 && prev_layer->size == next_layer->size);
		for (i = 0; i < prev_layer->size; i++)
		{
			next_layer->neuron[i] = 2/(1+exp(-2*prev_layer->neuron[i]))-1;
		}		
	}
	else if (type == _RELU)
	{
		assert(prev_layer->size > 0 && prev_layer->size == next_layer->size && relu_param >= 0 && relu_param < 1);
		for (i = 0; i < prev_layer->size; i++)
		{
			if (prev_layer->neuron[i] <= 0) next_layer->neuron[i] = prev_layer->neuron[i]*relu_param;
			else next_layer->neuron[i] = prev_layer->neuron[i];
		}		
	}
	else if (type == _LINEAR)
	{
		assert(prev_layer->size > 0 && prev_layer->size == next_layer->size);
		for (i = 0; i < prev_layer->size; i++)
		{
			next_layer->neuron[i] = prev_layer->neuron[i];
		}
	}
	else if (type == _SOFTMAX)
	{
		assert(prev_layer->size > 0 && prev_layer->size == next_layer->size);
		double sum = 0;
		double maxneuron = -(1e+307);
		for (i = 0; i < prev_layer->size; i++)
		{
			if (prev_layer->neuron[i] > maxneuron)
			{
				maxneuron = softmax_param*prev_layer->neuron[i];
			}
		}
		for (i = 0; i < prev_layer->size; i++)
		{
			sum += exp(softmax_param*prev_layer->neuron[i]-maxneuron);
		}
		for (i = 0; i < prev_layer->size; i++)
		{
			next_layer->neuron[i] = exp(softmax_param*prev_layer->neuron[i]-maxneuron)/sum;
		}
	}
	else if (type == _CONV)
	{
		assert(prev_layer->is3d && next_layer->is3d && prev_layer->depth > 0 && next_layer->depth > 0 && size2 == next_layer->depth);
		for (l = 0; l < next_layer->depth; l++)
		{
			for (i = 0; i < next_layer->width; i++)
				for (j = 0; j < next_layer->height; j++)
				{
					double newval = 0;
					for (k = 0; k < prev_layer->depth; k++)
					{
						int i1 = stride*i-padding;
						int j1 = stride*j-padding;
						int ii, jj;
						for (ii = 0; ii < width; ii++)
							for (jj = 0; jj < width; jj++)
							{
								newval += prev_layer->getval(i1+ii, j1+jj, k)*getweight(ii, jj, k, l);
							}
					}
					newval += bias[l];
					next_layer->setval(i, j, l, newval);
				}
		}
	}
	else if (type == _POOL_MAX || type == _POOL_AVERAGE)
	{
		assert(prev_layer->is3d && next_layer->is3d && prev_layer->depth > 0 && prev_layer->depth == next_layer->depth);
		for (k = 0; k < prev_layer->depth; k++)
		{
			for (i = 0; i < next_layer->width; i++)
				for (j = 0; j < next_layer->height; j++)
				{
					int i1 = stride*i;
					int j1 = stride*j;
					int ii, jj;
					if (type == _POOL_MAX)
					{
						double maxval = -(1e+307);
						for (ii = 0; ii < width; ii++)
							for (jj = 0; jj < width; jj++)
							{
								if (prev_layer->getval(i1+ii, j1+jj, k) > maxval) maxval = prev_layer->getval(i1+ii, j1+jj, k);
							}
						next_layer->setval(i, j, k, maxval);
					}
					else if (type == _POOL_AVERAGE)
					{
						double avgval = 0;
						for (ii = 0; ii < width; ii++)
							for (jj = 0; jj < width; jj++)
							{
								avgval += prev_layer->getval(i1+ii, j1+jj, k)/(width*width);
							}
						next_layer->setval(i, j, k, avgval);
					}
					else assert(false);
				}
		}
	}
	else if (type == _POOL_LINEAR)
	{
		assert(prev_layer->is3d && next_layer->is3d && prev_layer->depth > 0 && prev_layer->depth == next_layer->depth && next_layer->height == 1);
		for (l = 0; l < size2; l++)
		{
			next_layer->neuron[l] = 0;
			int d2 = size2%next_layer->depth;
			for (i = 0; i < prev_layer->width; i++)
				for (j = 0; j < prev_layer->height; j++)
				{
					next_layer->neuron[l] += prev_layer->getval(i, j, d2)*weight[i*prev_layer->height+j][l];
				}
			next_layer->neuron[l] += bias[l];
		}
	}
	else if (type == _FC)
	{
		assert(dropout_rate > 0 && dropout_rate <= 1);	
		for (j = 0; j < size2; j++)
		{
			next_layer->neuron[j] = 0;
			if (!testing && next_layer->isdropout[j]) continue;
			for (i = 0; i < size1; i++)
			{
				if (!testing && prev_layer->isdropout[i]) ;
				else next_layer->neuron[j] += prev_layer->neuron[i]*weight[i][j];
			}
			if (testing)
			{
				next_layer->neuron[j] *= dropout_rate;
			}
			next_layer->neuron[j] += bias[j];
		}
	}
	else
	{
		assert(false);
	}
}

void CONVOLUTIONAL_NEURAL_NETWORK::forward_propagation(const double *input, double *output, const bool testing)
{
	int i;
	assert(input != NULL && output != NULL);
	for (i = 0; i < input_layer->size; i++)
	{
		assert(!std::isnan(input[i]));
		input_layer->neuron[i] = input[i];
	}
	for (i = 0; i < connection_num; i++)
	{
		connection[i].forward_propagation(dropout_rate, testing);
	}
	for (i = 0; i < output_layer->size; i++)
	{
		output[i] = output_layer->neuron[i];
		assert(!std::isnan(output[i]));
	}
}

void LAYER::calcoutputdelta(const double *label)
{
	assert(label != NULL && next_connection != NULL && next_connection->next_layer->next_connection == NULL && size == next_connection->next_layer->size);
	int i;
	if (next_connection->type == _SIGMOID)
	{
		for (i = 0; i < size; i++)
		{
			delta[i] = next_connection->next_layer->neuron[i]*(1-next_connection->next_layer->neuron[i])*(label[i]-next_connection->next_layer->neuron[i]);
		}
	}
	else if (next_connection->type == _TANH)
	{
		for (i = 0; i < size; i++)
		{
			delta[i] = (1-next_connection->next_layer->neuron[i]*next_connection->next_layer->neuron[i])*(label[i]-next_connection->next_layer->neuron[i]);
		}
	}
	else if (next_connection->type == _RELU)
	{
		for (i = 0; i < size; i++)
		{
			if (next_connection->next_layer->neuron[i] > 0) delta[i] = label[i]-next_connection->next_layer->neuron[i];
			else delta[i] = next_connection->relu_param*(label[i]-next_connection->next_layer->neuron[i]); 
		}
	}
	else if (next_connection->type == _LINEAR || _SOFTMAX)
	{
		for (i = 0; i < size; i++)
		{
			delta[i] = (label[i]-next_connection->next_layer->neuron[i]);
		}
	}
	else
	{
		assert(false);
	}
}

double LAYER::getdelta(const int w, const int h, const int d)
{
	assert(is3d && w < width && w >= 0 && h < height && h >= 0 && d < depth && d >= 0);
	return delta[w*height*depth+h*depth+d];	
}

void LAYER::adddelta(const int w, const int h, const int d, const double addval)
{
	assert(is3d && d < depth && d >= 0);
	if (w >= width || w < 0 || h >= height || h < 0) return;
	delta[w*height*depth+h*depth+d] += addval;
}

void CONNECTION::back_propagation()
{
	assert(prev_layer->prev_connection != NULL && next_layer->next_connection != NULL);
	int i, j, k, l;
	for (i = 0; i < prev_layer->size; i++)
	{
		prev_layer->delta[i] = 0;
	}
	if (type == _SIGMOID || type == _TANH || type == _RELU || type == _FC)
	{
		for (i = 0; i < prev_layer->size; i++)
		{
			if (prev_layer->isdropout[i]) continue;
			if (type == _SIGMOID)
			{
				assert(prev_layer->size == next_layer->size);
				prev_layer->delta[i] = next_layer->delta[i]*(next_layer->neuron[i])*(1-next_layer->neuron[i]);
			}
			else if (type == _TANH)
			{
				assert(prev_layer->size == next_layer->size);
				prev_layer->delta[i] = next_layer->delta[i]*(1-next_layer->neuron[i]*next_layer->neuron[i]);
			}
			else if (type == _RELU)
			{
				assert(prev_layer->size == next_layer->size);
				prev_layer->delta[i] = next_layer->delta[i];
				if (next_layer->neuron[i] <= 0) prev_layer->delta[i] *= relu_param;
			}
			else if (type == _FC)
			{
				assert(weight != NULL && size1 == prev_layer->size && size2 == next_layer->size);
				prev_layer->delta[i] = 0;
				for (j = 0; j < next_layer->size; j++)
				{
					if (next_layer->isdropout[j]) continue;
					prev_layer->delta[i] += weight[i][j]*next_layer->delta[j];
				}
			}
			else
			{
				assert(false);
			}
		}
	}
	else if (type == _CONV)
	{		
		for (l = 0; l < next_layer->depth; l++)
		{
			for (i = 0; i < next_layer->width; i++)
				for (j = 0; j < next_layer->height; j++)
				{
					for (k = 0; k < prev_layer->depth; k++)
					{
						int i1 = stride*i-padding;
						int j1 = stride*j-padding;
						int ii, jj;
						for (ii = 0; ii < width; ii++)
							for (jj = 0; jj < width; jj++)
							{
								prev_layer->adddelta(i1+ii, j1+jj, k, next_layer->getdelta(i, j, l)*getweight(ii, jj, k, l));
							}
					}
				}
		}		
	}
	else if (type == _POOL_MAX || type == _POOL_AVERAGE)
	{
		for (k = 0; k < prev_layer->depth; k++)
		{
			for (i = 0; i < next_layer->width; i++)
				for (j = 0; j < next_layer->height; j++)
				{
					int i1 = stride*i;
					int j1 = stride*j;
					int ii, jj;
					if (type == _POOL_MAX)
					{
						double maxval = -(1e+307);
						int selecti = i1+ii;
						int selectj = j1+jj;
						for (ii = 0; ii < width; ii++)
							for (jj = 0; jj < width; jj++)
							{
								if (prev_layer->getval(i1+ii, j1+jj, k) > maxval)
								{
									selecti = i1+ii;
									selectj = j1+jj;
									maxval = prev_layer->getval(i1+ii, j1+jj, k);
								}
							}
						prev_layer->adddelta(selecti, selectj, k, next_layer->getdelta(i, j, k));
					}
					else if (type == _POOL_AVERAGE)
					{
						for (ii = 0; ii < width; ii++)
							for (jj = 0; jj < width; jj++)
							{
								prev_layer->adddelta(i1+ii, j1+jj, k, next_layer->getdelta(i, j, k)/(width*width));
							}
					}
					else assert(false);
				}
		}
	}
	else if (type == _POOL_LINEAR)
	{
		for (l = 0; l < size2; l++)
		{
			int d2 = size2%next_layer->depth;
			for (i = 0; i < prev_layer->width; i++)
				for (j = 0; j < prev_layer->height; j++)
				{
					prev_layer->adddelta(i, j, d2, weight[i*prev_layer->height+j][l]*next_layer->delta[l]);
				}
		}
	}
	else
	{
		assert(false);
	}
}

void CONVOLUTIONAL_NEURAL_NETWORK::back_propagation()
{
	int i;
	for (i = connection_num-2; i >= 1; i--)
	{
		connection[i].back_propagation();
	}
}

void CONVOLUTIONAL_NEURAL_NETWORK::update(CONNECTION *conn)
{
	assert(conn->type == _CONV || conn->type == _FC || conn->type == _POOL_LINEAR);
	assert(conn->weight != NULL && conn->dw != NULL && conn->bias != NULL && conn->db != NULL);
	int i, j, k, l;
	if (conn->type == _CONV)
	{
		for (l = 0; l < conn->next_layer->depth; l++)
		{
			for (i = 0; i < conn->next_layer->width; i++)
				for (j = 0; j < conn->next_layer->height; j++)
				{
					for (k = 0; k < conn->prev_layer->depth; k++)
					{
						int i1 = conn->stride*i-conn->padding;
						int j1 = conn->stride*j-conn->padding;
						int ii, jj;
						for (ii = 0; ii < conn->width; ii++)
							for (jj = 0; jj < conn->width; jj++)
							{
								conn->adddw(ii, jj, k, l, training_rate*conn->next_layer->getdelta(i, j, l)*conn->prev_layer->getval(i1+ii, j1+jj, k)/batch_size);
							}
					}
					conn->db[l] += training_rate*conn->next_layer->getdelta(i, j, l)/batch_size;
				}
		}		
	}
	else if (conn->type == _POOL_LINEAR)
	{
		for (l = 0; l < conn->size2; l++)
		{
			int d2 = conn->size2%conn->next_layer->depth;
			for (i = 0; i < conn->prev_layer->width; i++)
				for (j = 0; j < conn->prev_layer->height; j++)
				{
					conn->dw[i*conn->prev_layer->height+j][l] += training_rate*conn->next_layer->delta[l]*conn->prev_layer->getval(i, j, d2)/batch_size;
				}
			conn->db[l] += training_rate*conn->next_layer->delta[l]/batch_size;
		}
	}
	else if (conn->type == _FC)
	{
		for (i = 0; i < conn->size1; i++)
		{
			if (conn->prev_layer->isdropout[i]) continue;
			for (j = 0; j < conn->size2; j++)
			{
				if (conn->next_layer->isdropout[j]) continue;
				conn->dw[i][j] += training_rate*conn->next_layer->delta[j]*conn->prev_layer->neuron[i]/batch_size;
			}
		}
		for (j = 0; j < conn->size2; j++)
		{
			if (conn->next_layer->isdropout[j]) continue;
			conn->db[j] += training_rate*conn->next_layer->delta[j]/batch_size;
		}
	}
	if (training_cnt % batch_size == 0)
	{
		for (i = 0; i < conn->size1; i++)
		{
			if (conn->type == _FC && conn->prev_layer->isdropout[i]) continue;
			for (j = 0; j < conn->size2; j++)
			{
				if (conn->type == _FC && conn->next_layer->isdropout[j]) continue;
				conn->dw[i][j] -= training_rate*regularization_strength*conn->weight[i][j];
				conn->weight[i][j] += conn->dw[i][j];
				conn->dw[i][j] *= momentum_factor;
			}
		}
		for (j = 0; j < conn->size2; j++)
		{
			if (conn->type == _FC && conn->next_layer->isdropout[j]) continue;
			conn->bias[j] += conn->db[j];
			conn->db[j] *= momentum_factor;
		}
	}
}

void CONVOLUTIONAL_NEURAL_NETWORK::dropout()
{
	int i;
	CONNECTION *cur_conn = connection;
	uniform_real_distribution<> distribution(0, 1);
	while (cur_conn != NULL)
	{
		if (cur_conn->type == _FC)
		{
			LAYER *need_to_dropout_layer = cur_conn->next_layer;
			LAYER *end_dropout_layer = need_to_dropout_layer;
			bool need_to_dropout = false;
			while (end_dropout_layer != NULL)
			{
				if (end_dropout_layer->next_connection == NULL) break;
				else if (end_dropout_layer->next_connection->type == _FC)
				{
					need_to_dropout = true;
					break;
				}
				else
				{
					end_dropout_layer = end_dropout_layer->next_connection->next_layer;
				}
			}
			cur_conn = end_dropout_layer->next_connection;
			if (need_to_dropout)
			{
				for (i = 0; i < need_to_dropout_layer->size; i++)
				{
					need_to_dropout_layer->isdropout[i] = distribution(mt19937generator) >= dropout_rate;
				}
				LAYER *l = need_to_dropout_layer;
				while (l != end_dropout_layer)
				{
					l = l->next_connection->next_layer;
					assert(l->size == need_to_dropout_layer->size);
					for (i = 0; i < l->size; i++)
					{
						l->isdropout[i] = need_to_dropout_layer->isdropout[i];
					}
				}
			}
		}
		else
		{
			cur_conn = cur_conn->next_layer->next_connection;
		}
	}
}

void CONVOLUTIONAL_NEURAL_NETWORK::train(const double *input_data, double *output_result, const double *label)
{
	int i;
	assert(layer != NULL && input_layer != NULL && output_layer != NULL && connection != NULL);
	training_cnt++;
	if (dropout_rate < 1 && training_cnt % batch_size == 1 % batch_size)
	{
		dropout();
	}
	forward_propagation(input_data, output_result, false);
	output_layer->prev_connection->prev_layer->calcoutputdelta(label);
	back_propagation();
	for (i = 0; i < connection_num; i++)
	{
		if (connection[i].type == _CONV || connection[i].type == _FC || connection[i].type == _POOL_LINEAR)
		{
			update(connection+i);
		}
	}
}

void CONVOLUTIONAL_NEURAL_NETWORK::train(const double *delta)
{
	int i;
	assert(delta != NULL);
	training_cnt++;
	for (i = 0; i < output_layer->prev_connection->prev_layer->size; i++)
	{
		output_layer->prev_connection->prev_layer->delta[i] = delta[i];
	}
	back_propagation();
	for (i = 0; i < connection_num; i++)
	{
		if (connection[i].type == _CONV || connection[i].type == _FC || connection[i].type == _POOL_LINEAR)
		{
			update(connection+i);
		}
	}	
}

void CONVOLUTIONAL_NEURAL_NETWORK::test(const double *input_data, double *output_result)
{
	assert(layer != NULL && input_layer != NULL && output_layer != NULL && connection != NULL);
	forward_propagation(input_data, output_result, true);
}

CONVOLUTIONAL_NEURAL_NETWORK::CONVOLUTIONAL_NEURAL_NETWORK(ifstream &fin)
{
	layer = input_layer = output_layer = NULL;
	connection = NULL;
	initialization(fin);
} 

void CONVOLUTIONAL_NEURAL_NETWORK::initialization(ifstream &fin)
{
	assert(layer == NULL && input_layer == NULL && output_layer == NULL && connection == NULL);
	fin >> total_layer_num >> connection_num;
	fin >> training_rate >> batch_size >> momentum_factor >> regularization_strength >> dropout_rate >> training_cnt;
	layer = new LAYER[total_layer_num];
	connection = new CONNECTION[connection_num];
	input_layer = layer;
	output_layer = layer+total_layer_num-1;
	int i, j, k;
	for (i = 0; i < total_layer_num; i++)
	{
		unordered_map<int, int> tmpmap;
		int tag, tmpw, tmph, tmpd, tmpsize;
		fin >> tag;
		if (tag == 1)
		{
			fin >> tmpw >> tmph >> tmpd;
			tmpmap.insert(pair<int, int>(LAYER_WIDTH, tmpw));
			tmpmap.insert(pair<int, int>(LAYER_HEIGHT, tmph));
			tmpmap.insert(pair<int, int>(LAYER_DEPTH, tmpd));
		}
		else
		{
			fin >> tmpsize;
			tmpmap.insert(pair<int, int>(LAYER_SIZE, tmpsize));
		}
		layer[i].initialization(&tmpmap);
	}
	for (i = 0; i < connection_num; i++)
	{
		unordered_map<int, double> tmpmap;
		int tmptype, tmpw, tmps, tmpp;
		double tmpreluparam, tmpsoftmaxparam;
		fin >> tmptype >> tmpw >> tmps >> tmpp >> tmpreluparam >> tmpsoftmaxparam;
		tmpmap.insert(pair<int, double>(CONNECTION_TYPE, tmptype));
		tmpmap.insert(pair<int, double>(CONNECTION_WIDTH, tmpw));
		tmpmap.insert(pair<int, double>(CONNECTION_STRIDE, tmps));
		tmpmap.insert(pair<int, double>(CONNECTION_PADDING, tmpp));
		tmpmap.insert(pair<int, double>(CONNECTION_RELU, tmpreluparam));
		tmpmap.insert(pair<int, double>(CONNECTION_SOFTMAX, tmpsoftmaxparam));
		connection[i].initialization(&tmpmap, layer+i, layer+i+1);
	}
	assert(output_layer->prev_connection->type == _SIGMOID || output_layer->prev_connection->type == _TANH || output_layer->prev_connection->type == _RELU || output_layer->prev_connection->type == _LINEAR || output_layer->prev_connection->type == _SOFTMAX);
	while (fin >> i)
	{
		assert(connection[i].weight != NULL && connection[i].bias != NULL && connection[i].dw != NULL && connection[i].db != NULL);
		for (j = 0; j < connection[i].size1; j++)
			for (k = 0; k < connection[i].size2; k++)
				fin >> connection[i].weight[j][k];
		for (k = 0; k < connection[i].size2; k++)
			fin >> connection[i].bias[k];
		for (j = 0; j < connection[i].size1; j++)
			for (k = 0; k < connection[i].size2; k++)
				fin >> connection[i].dw[j][k];
		for (k = 0; k < connection[i].size2; k++)
			fin >> connection[i].db[k];
	}
	mt19937generator = mt19937(time(0));
}

void CONVOLUTIONAL_NEURAL_NETWORK::print(ofstream &fout)
{
	assert(layer != NULL && input_layer != NULL && output_layer != NULL && connection != NULL);
	fout.precision(16);
	fout << total_layer_num << " " << connection_num << " " << endl;
	fout << training_rate << " " << batch_size << " " << momentum_factor << " " << regularization_strength << " " << dropout_rate << " " << training_cnt << endl;
	int i, j, k;
	for (i = 0; i < total_layer_num; i++)
	{
		if (layer[i].is3d) fout << "1 " << layer[i].width << " " << layer[i].height << " " << layer[i].depth << endl;
		else fout << "0 " << layer[i].size << endl;
	}
	for (i = 0; i < connection_num; i++)
	{
		fout << connection[i].type << " " << connection[i].width << " " << connection[i].stride << " " << connection[i].padding << " " << connection[i].relu_param << " " << connection[i].softmax_param << endl;
	}
	for (i = 0; i < connection_num; i++)
	{
		if (connection[i].weight != NULL)
		{
			assert(connection[i].bias != NULL && connection[i].dw != NULL && connection[i].db != NULL);
			fout << i << endl;
			for (j = 0; j < connection[i].size1; j++)
			{
				for (k = 0; k < connection[i].size2; k++)
				{
					fout << connection[i].weight[j][k] << " ";
				}
				fout << endl;
			}
			for (k = 0; k < connection[i].size2; k++)
			{
				fout << connection[i].bias[k] << " ";
			}
			fout << endl;
			for (j = 0; j < connection[i].size1; j++)
			{
				for (k = 0; k < connection[i].size2; k++)
				{
					fout << connection[i].dw[j][k] << " ";
				}
				fout << endl;
			}
			for (k = 0; k < connection[i].size2; k++)
			{
				fout << connection[i].db[k] << " ";
			}
			fout << endl;			
		}
	}
}

void CONVOLUTIONAL_NEURAL_NETWORK::copyto(CONVOLUTIONAL_NEURAL_NETWORK &cnn)
{
	assert(total_layer_num == cnn.total_layer_num && connection_num == cnn.connection_num);
	int i, j, k;
	for (i = 0; i < connection_num; i++)
	{
		assert((connection[i].weight == NULL) == (cnn.connection[i].weight == NULL) && (connection[i].bias == NULL) == (cnn.connection[i].bias == NULL));
		if (connection[i].weight != NULL)
		{
			assert(connection[i].bias != NULL && connection[i].size1 == cnn.connection[i].size1 && connection[i].size2 == cnn.connection[i].size2);
			for (j = 0; j < connection[i].size1; j++)
				for (k = 0; k < connection[i].size2; k++)
					cnn.connection[i].weight[j][k] = connection[i].weight[j][k];
			for (k = 0; k < connection[i].size2; k++)
				cnn.connection[i].bias[k] = connection[i].bias[k];
		}
	}
}

void CONVOLUTIONAL_NEURAL_NETWORK::getneuronfromlayer(const int layer_index, double *neuron)
{
	assert(neuron != NULL && layer_index >= 0 && layer_index < total_layer_num);
	int i;
	for (i = 0; i < layer[layer_index].size; i++)
	{
		neuron[i] = layer[layer_index].neuron[i];
	}
}

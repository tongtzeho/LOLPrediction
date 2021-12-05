#pragma once

#include <cmath>
#include <vector>
#include <unordered_map>
#include <ctime>
#include <random>
#include <cassert>
#include <cstdio>
#include <thread>
#include <atomic>
#include <functional>
#include <DirectXMath.h>

struct neural_network_layer;
struct neural_network_connection;
struct neural_network_thread_arg;
class neural_network;

struct neural_network_layer
{
	neural_network_layer() = default;
	~neural_network_layer() = default;
	void initialization(const std::unordered_map<int, int>&, int);
	void calculate_output_delta(const std::vector<float>&, int);

	void deserialize(const std::vector<unsigned char>&, size_t&);
	void serialize(std::vector<unsigned char>&, size_t&) const;
	inline size_t estimate_byte() const { return sizeof(int); }

	neural_network_connection* prev_connection{ nullptr };
	neural_network_connection* next_connection{ nullptr };
	std::vector<std::vector<float>> neuron;
	std::vector<std::vector<float>> delta;
	int size{ 0 };

	static constexpr int layer_size_key = 100;
};

struct neural_network_connection
{
	neural_network_connection() = default;
	~neural_network_connection() { if (weight2d_root) delete[]weight2d_root; }
	void initialization(const std::unordered_map<int, float>&, neural_network_layer*, neural_network_layer*, int);
	void weight_initialization(std::mt19937&);
	void forward_propagation(std::pair<std::vector<float>, std::vector<float>>&)const;
	void activation(std::vector<float>&)const;
	void back_propagation(int, int, float);

	void deserialize(const std::vector<unsigned char>&, size_t&);
	void serialize(std::vector<unsigned char>&, size_t&) const;
	inline size_t estimate_byte() const { return sizeof(int) * 3 + sizeof(float) * (size1 * size2 + bias.size()); }

	neural_network_layer* prev_layer{ nullptr };
	neural_network_layer* next_layer{ nullptr };
	float* weight2d_root{ nullptr };
	float* weight2d{ nullptr };
	std::vector<float> bias; // [size2]
	std::vector<float> delta_weight2d; // [size1][size2]
	std::vector<float> delta_bias; // [size2]
	std::vector<std::vector<float>> delta_weight2d_per_thread;
	std::vector<std::vector<float>> delta_bias_per_thread;
	int type{ 0 };
	int size1{ 0 };
	int size2{ 0 };

	static constexpr int activation_sigmoid = 1;
	static constexpr int activation_tanh = 2;
	static constexpr int activation_relu = 3;
	static constexpr int output_linear = 4;
	static constexpr int output_softmax = 5;
	static constexpr int full_connection = 10;

	static constexpr int connection_type_key = 200;
};

struct neural_network_thread_arg
{
	std::function<void(neural_network*, std::pair<std::vector<float>, std::vector<float>>*, const std::vector<std::vector<float>>*, int, int)> job_func;
	const std::vector<std::vector<float>>* shared_arg{ nullptr };
	int begin_job_id{ 0 };
	int end_job_id{ 0 };
	int thread_id{ 0 };
	volatile bool work_flag{ false };
	volatile bool exit_flag{ false };
};

class neural_network
{
public:
	neural_network() = default;
	~neural_network() = default;
	void initialization(const std::unordered_map<int, int>*, const std::unordered_map<int, float>*, const std::unordered_map<int, float>&, const std::vector<float>&);
	static void shuffle(std::vector<std::vector<float>>&, std::vector<std::vector<float>>&);
	void train(std::vector<std::vector<float>>&, std::vector<std::vector<float>>&, int);
	void train(std::pair<std::vector<float>, std::vector<float>>&, const std::vector<float>&);
	void test(std::pair<std::vector<float>, std::vector<float>>&)const;

	void deserialize(const std::vector<unsigned char>&);
	void serialize(std::vector<unsigned char>&) const;
	size_t estimate_byte() const;
	void read_from_file(const char*);
	void write_to_file(const char*) const;

	int get_input_layer_size() const { return input_layer->size; }
	int get_output_layer_size() const { return output_layer->size; }
	int get_max_layer_size() const { return max_layer_size; }

private:
	void weight_initialization();
	void preprocess(std::vector<float>&)const;
	void update_weight(neural_network_connection*);

	static void forward_propagation_job(neural_network*, std::pair<std::vector<float>, std::vector<float>>*, const std::vector<std::vector<float>>*, int, int);
	static void backward_propagation_job(neural_network*, std::pair<std::vector<float>, std::vector<float>>*, const std::vector<std::vector<float>>*, int, int);
	static void update_weight_job(neural_network*, std::pair<std::vector<float>, std::vector<float>>*, const std::vector<std::vector<float>>*, int, int);
	static void worker_thread(neural_network*, std::pair<std::vector<float>, std::vector<float>>*, neural_network_thread_arg*);

	std::vector<neural_network_layer> layers;
	std::vector<neural_network_connection> connections;
	std::vector<float> average_standarderrors;
	neural_network_layer* input_layer{ nullptr };
	neural_network_layer* output_layer{ nullptr };
	float training_rate{ 0.1f };
	float decay_rate{ 1.0f };
	float min_training_rate{ 0.001f };
	float momentum_factor{ 0.0f };
	float regularization_strength{ 0 };
	int training_count_per_batch{ 0 };
	int total_layer_count{ 0 };
	int connection_count{ 0 };
	int batch_size{ 1 };
	int max_layer_size{ 0 };
	int thread_count{ 1 };
	std::mt19937 mt19937generator;

public:
	static constexpr int nn_total_layer_key = 300;
	static constexpr int nn_training_rate_key = 301;
	static constexpr int nn_decay_rate_key = 302;
	static constexpr int nn_min_training_rate_key = 303;
	static constexpr int nn_batch_size_key = 304;
	static constexpr int nn_momentum_factor_key = 305;
	static constexpr int nn_regularization_strength_key = 306;
	static constexpr int nn_thread_count_key = 307;
};

void neural_network_layer::initialization(const std::unordered_map<int, int>& param, int batch_size)
{
	auto it = param.find(layer_size_key);
	if (it != param.end())
	{
		size = it->second;
	}
	assert(size > 0);
	neuron.resize((size_t)batch_size);
	for (std::vector<float>& n : neuron)
		n.resize((size_t)size, 0.0f);
	delta.resize((size_t)batch_size);
	for (std::vector<float>& d : delta)
		d.resize((size_t)size, 0.0f);
}

void neural_network_connection::initialization(const std::unordered_map<int, float>& param, neural_network_layer* prev, neural_network_layer* next, int thread_count)
{
	auto it = param.find(connection_type_key);
	assert(it != param.end());
	type = (int)it->second;
	prev_layer = prev;
	next_layer = next;
	prev->next_connection = this;
	next->prev_connection = this;
	size1 = size2 = 0;
	if (type == full_connection)
	{
		size1 = prev->size;
		size2 = next->size;
	}
	if (size1 > 0 && size2 > 0)
	{
		weight2d_root = new float[size1 * size2 + 4];
		weight2d = weight2d_root;
		while (((size_t)weight2d) % 16)
			++weight2d;
		bias.resize((size_t)size2, 0.0f);
		delta_weight2d.resize((size_t)(size1 * size2), 0.0f);
		delta_bias.resize((size_t)size2, 0.0f);
		delta_weight2d_per_thread.resize((size_t)thread_count);
		for (std::vector<float>& dw : delta_weight2d_per_thread)
			dw.resize((size_t)(size1 * size2), 0.0f);
		delta_bias_per_thread.resize((size_t)thread_count);
		for (std::vector<float>& db : delta_bias_per_thread)
			db.resize((size_t)size2, 0.0f);
	}
}

void neural_network::initialization(const std::unordered_map<int, int>* layer_param, const std::unordered_map<int, float>* connection_param, const std::unordered_map<int, float>& param, const std::vector<float>& average_sdes)
{
	assert(layer_param && connection_param);
	assert(layers.empty() && !input_layer && !output_layer && connections.empty());
	srand((unsigned int)time(0));
	mt19937generator = std::mt19937((unsigned int)time(0));
	auto it = param.find(nn_total_layer_key);
	assert(it != param.end());
	total_layer_count = (int)it->second;
	connection_count = total_layer_count - 1;
	assert(connection_count > 0);
	it = param.find(nn_training_rate_key);
	if (it != param.end())
		training_rate = it->second;
	it = param.find(nn_decay_rate_key);
	if (it != param.end())
		decay_rate = it->second;
	it = param.find(nn_min_training_rate_key);
	if (it != param.end())
		min_training_rate = it->second;
	it = param.find(nn_batch_size_key);
	if (it != param.end())
		batch_size = (int)std::max(1.0f, it->second);
	it = param.find(nn_momentum_factor_key);
	if (it != param.end())
		momentum_factor = it->second;
	it = param.find(nn_regularization_strength_key);
	if (it != param.end())
		regularization_strength = it->second;
	it = param.find(nn_thread_count_key);
	if (it != param.end())
		thread_count = (int)std::max(1.0f, it->second);
	layers.resize((size_t)total_layer_count);
	connections.resize((size_t)connection_count);
	input_layer = &layers.front();
	output_layer = &layers.back();
	for (int i = 0; i < total_layer_count; ++i)
	{
		layers[i].initialization(layer_param[i], batch_size);
		max_layer_size = std::max(max_layer_size, layers[i].size);
	}
	max_layer_size += 4;
	for (int i = 0; i < connection_count; ++i)
	{
		connections[i].initialization(connection_param[i], &layers[i], &layers[i + 1], thread_count);
	}
	average_standarderrors = average_sdes;
	assert(output_layer->prev_connection->type == neural_network_connection::activation_sigmoid || output_layer->prev_connection->type == neural_network_connection::output_linear || output_layer->prev_connection->type == neural_network_connection::output_softmax);
	assert(average_standarderrors.size() == input_layer->size * 2);
	weight_initialization();
}

void neural_network_connection::weight_initialization(std::mt19937& mt19937generator)
{
	if (size1 == 0 || size2 == 0)
		return;
	std::normal_distribution<float> distribution(0.0f, sqrtf(1.0f / size1));
	float* p_weight = &weight2d[0];
	const float* p_weight_end = p_weight + size1 * size2;
	while (p_weight != p_weight_end)
	{
		*(p_weight++) = distribution(mt19937generator);
	}
}

void neural_network::weight_initialization()
{
	assert(!layers.empty() && input_layer && output_layer && !connections.empty());
	for (int i = 0; i < connection_count; ++i)
	{
		connections[i].weight_initialization(mt19937generator);
	}
}

void neural_network_connection::forward_propagation(std::pair<std::vector<float>, std::vector<float>>& flow)const
{
	assert(type == full_connection && weight2d && size1 == prev_layer->size && size2 == next_layer->size);
	int s1 = size1;
	int s2 = size2;
	if (s2 % 4 == 0)
	{
		float* p_temp_start = &flow.second[0];
		while (((size_t)p_temp_start) % 16)
			++p_temp_start;
		memcpy(p_temp_start, &bias[0], sizeof(float) * s2);
		const float* p_weight = &weight2d[0];
		for (int i = 0; i < s1; ++i)
		{
			float n = flow.first[i];
			if (n == 0.0f)
			{
				p_weight += s2;
				continue;
			}
			DirectX::XMVECTOR nnnn = DirectX::XMVectorReplicate(n);
			float* p_temp = p_temp_start;
			const float* p_temp_end = p_temp + s2;
			for (; p_temp != p_temp_end; p_temp += 4, p_weight += 4)
			{
				DirectX::XMVECTOR weight = DirectX::XMLoadFloat4A(reinterpret_cast<const DirectX::XMFLOAT4A*>(p_weight));
				DirectX::XMVECTOR temp = DirectX::XMLoadFloat4A(reinterpret_cast<const DirectX::XMFLOAT4A*>(p_temp));
				DirectX::XMStoreFloat4A(reinterpret_cast<DirectX::XMFLOAT4A*>(p_temp), DirectX::XMVectorMultiplyAdd(nnnn, weight, temp));
			}
		}
		memcpy(&flow.first[0], p_temp_start, sizeof(float) * s2);
	}
	else
	{
		memcpy(&flow.second[0], &bias[0], sizeof(float) * s2);
		const float* p_weight = &weight2d[0];
		for (int i = 0; i < s1; ++i)
		{
			float n = flow.first[i];
			if (n == 0.0f)
			{
				p_weight += s2;
				continue;
			}
			float* p_temp = &flow.second[0];
			const float* p_temp_end = p_temp + s2;
			for (; p_temp != p_temp_end; ++p_temp)
			{
				(*p_temp) += n * (*(p_weight++));
			}
		}
		memcpy(&flow.first[0], &flow.second[0], sizeof(float) * s2);
	}
}

void neural_network_connection::activation(std::vector<float>& flow)const
{
	assert(prev_layer->size > 0 && prev_layer->size == next_layer->size);
	float* p_flow = &flow[0];
	const float* p_flow_end = p_flow + prev_layer->size;
	switch (type)
	{
	case activation_sigmoid:
	{
		for (; p_flow != p_flow_end; ++p_flow)
		{
			*p_flow = 1.0f / (1.0f + exp(-(*p_flow)));
		}
		return;
	}
	case activation_tanh:
	{
		for (; p_flow != p_flow_end; ++p_flow)
		{
			*p_flow = tanh(*p_flow);
		}
		return;
	}
	case activation_relu:
	{
		for (; p_flow != p_flow_end; ++p_flow)
		{
			if ((*p_flow) < 0.0f)
				(*p_flow) = 0.0f;
		}
		return;
	}
	case output_linear:
	{
		return;
	}
	case output_softmax:
	{
		float sum = 0.0f;
		float maximum = -FLT_MAX;
		for (p_flow = &flow[0]; p_flow != p_flow_end; ++p_flow)
		{
			maximum = std::max(maximum, (*p_flow));
		}
		for (p_flow = &flow[0]; p_flow != p_flow_end; ++p_flow)
		{
			(*p_flow) = exp(*p_flow - maximum);
			sum += (*p_flow);
		}
		for (p_flow = &flow[0]; p_flow != p_flow_end; ++p_flow)
		{
			(*p_flow) /= sum;
		}
		return;
	}
	default:
		assert(false);
	}
}

void neural_network::preprocess(std::vector<float>& flow)const
{
	assert(input_layer->size * 2 == average_standarderrors.size());
	const float* p_average_standarderror = &average_standarderrors[0];
	for (int i = 0; i < input_layer->size; ++i)
	{
		flow[i] -= *(p_average_standarderror++);
		float sde = *(p_average_standarderror++);
		if (sde > 1e-10f)
		{
			flow[i] /= sde;
		}
	}
}

void neural_network_layer::calculate_output_delta(const std::vector<float>& label, int slot)
{
	assert(label.size() == (size_t)size && next_connection && !next_connection->next_layer->next_connection && size == next_connection->next_layer->size);
	const float* p_neuron = &next_connection->next_layer->neuron[slot][0];
	const float* p_neuron_end = p_neuron + size;
	float* p_delta = &delta[slot][0];
	const float* p_label = &label[0];
	switch (next_connection->type)
	{
	case neural_network_connection::activation_sigmoid:
	{
		while (p_neuron != p_neuron_end)
		{
			float n = *(p_neuron++);
			*(p_delta++) = n * (1.0f - n) * (*(p_label++) - n);
		}
		return;
	}
	case neural_network_connection::output_linear:
	case neural_network_connection::output_softmax:
	{
		while (p_neuron != p_neuron_end)
		{
			*(p_delta++) = (*(p_label++) - *(p_neuron++));
		}
		return;
	}
	default:
		assert(false);
	}
}

void neural_network_connection::back_propagation(int thread_id, int slot, float training_rate_per_batch)
{
	if (type == full_connection)
	{
		assert(weight2d && size1 == prev_layer->size && size2 == next_layer->size && thread_id < (int)delta_weight2d_per_thread.size() && thread_id < (int)delta_bias_per_thread.size());
		int s1 = size1;
		int s2 = size2;
		float* p_delta_weight = &delta_weight2d_per_thread[thread_id][0];
		if (prev_layer->prev_connection)
		{
			float* p_weight = &weight2d[0];
			if (s2 % 4 == 0)
			{
				for (int i = 0; i < s1; ++i)
				{
					DirectX::XMVECTOR dddd = DirectX::XMVectorZero();
					const float* p_next_delta = &next_layer->delta[slot][0];
					const float* p_next_delta_end = p_next_delta + s2;
					float n = prev_layer->neuron[slot][i] * training_rate_per_batch;
					if (n != 0.0f)
					{
						DirectX::XMVECTOR nnnn = DirectX::XMVectorReplicate(n);
						for (; p_next_delta != p_next_delta_end; p_next_delta += 4, p_weight += 4, p_delta_weight += 4)
						{
							DirectX::XMVECTOR weight = DirectX::XMLoadFloat4(reinterpret_cast<const DirectX::XMFLOAT4A*>(p_weight));
							DirectX::XMVECTOR next_delta = DirectX::XMLoadFloat4(reinterpret_cast<const DirectX::XMFLOAT4*>(p_next_delta));
							DirectX::XMVECTOR dot = DirectX::XMVector4Dot(weight, next_delta);
							dddd = DirectX::XMVectorAdd(dddd, dot);
							DirectX::XMVECTOR delta_weight = DirectX::XMLoadFloat4(reinterpret_cast<const DirectX::XMFLOAT4*>(p_delta_weight));
							DirectX::XMStoreFloat4(reinterpret_cast<DirectX::XMFLOAT4*>(p_delta_weight), DirectX::XMVectorMultiplyAdd(next_delta, nnnn, delta_weight));
						}
					}
					else
					{
						for (; p_next_delta != p_next_delta_end; p_next_delta += 4, p_weight += 4)
						{
							DirectX::XMVECTOR weight = DirectX::XMLoadFloat4(reinterpret_cast<const DirectX::XMFLOAT4A*>(p_weight));
							DirectX::XMVECTOR next_delta = DirectX::XMLoadFloat4(reinterpret_cast<const DirectX::XMFLOAT4*>(p_next_delta));
							DirectX::XMVECTOR dot = DirectX::XMVector4Dot(weight, next_delta);
							dddd = DirectX::XMVectorAdd(dddd, dot);
						}
						p_delta_weight += s2;
					}
					prev_layer->delta[slot][i] = DirectX::XMVectorGetX(dddd);
				}
			}
			else
			{
				for (int i = 0; i < s1; ++i)
				{
					float d = 0.0f;
					const float* p_next_delta = &next_layer->delta[slot][0];
					const float* p_next_delta_end = p_next_delta + s2;
					float n = prev_layer->neuron[slot][i] * training_rate_per_batch;
					if (n != 0.0f)
					{
						for (; p_next_delta != p_next_delta_end; ++p_next_delta)
						{
							d += (*(p_weight++)) * (*p_next_delta);
							(*(p_delta_weight++)) += (*p_next_delta) * n;
						}
					}
					else
					{
						for (; p_next_delta != p_next_delta_end; ++p_next_delta)
						{
							d += (*(p_weight++)) * (*p_next_delta);
						}
						p_delta_weight += s2;
					}
					prev_layer->delta[slot][i] = d;
				}
			}
		}
		else
		{
			for (int i = 0; i < s1; ++i)
			{
				float n = prev_layer->neuron[slot][i] * training_rate_per_batch;
				const float* p_next_delta = &next_layer->delta[slot][0];
				const float* p_next_delta_end = p_next_delta + s2;
				for (; p_next_delta != p_next_delta_end; ++p_next_delta)
				{
					(*(p_delta_weight++)) += (*p_next_delta) * n;
				}
			}
		}
		float* p_delta_bias = &delta_bias_per_thread[thread_id][0];
		const float* p_delta_bias_end = p_delta_bias + s2;
		const float* p_delta = &next_layer->delta[slot][0];
		while (p_delta_bias != p_delta_bias_end)
		{
			*(p_delta_bias++) += *(p_delta++) * training_rate_per_batch;
		}
		return;
	}

	assert(prev_layer->prev_connection && next_layer->next_connection && prev_layer->size == next_layer->size);
	const float* p_neuron = &next_layer->neuron[slot][0];
	const float* p_neuron_end = p_neuron + prev_layer->size;
	const float* p_next_delta = &next_layer->delta[slot][0];
	float* p_prev_delta = &prev_layer->delta[slot][0];
	switch (type)
	{
	case activation_sigmoid:
	{
		while (p_neuron != p_neuron_end)
		{
			float n = *(p_neuron++);
			*(p_prev_delta++) = *(p_next_delta++) * n * (1.0f - n);
		}
		return;
	}
	case activation_tanh:
	{
		while (p_neuron != p_neuron_end)
		{
			float n = *(p_neuron++);
			*(p_prev_delta++) = *(p_next_delta++) * (1.0f - n * n);
		}
		return;
	}
	case activation_relu:
	{
		while (p_neuron != p_neuron_end)
		{
			float n = *(p_neuron++);
			*(p_prev_delta++) = (n > 0.0f) ? (*p_next_delta) : 0.0f;
			++p_next_delta;
		}
		return;
	}
	default:
		assert(false);
	}
}

void neural_network::update_weight(neural_network_connection* conn)
{
	assert(conn->type == neural_network_connection::full_connection && conn->weight2d && !conn->delta_weight2d.empty() && !conn->bias.empty() && !conn->delta_bias.empty());
	float r = training_rate * regularization_strength;
	int s1 = conn->size1;
	int s2 = conn->size2;
	int s = s1 * s2;
	for (size_t k = 0; k < conn->delta_weight2d_per_thread.size(); ++k)
	{
		float* p_delta_weight = &conn->delta_weight2d[0];
		float* p_delta_weight_per_thread = &conn->delta_weight2d_per_thread[k][0];
		for (int i = 0; i < s; ++i)
		{
			*(p_delta_weight++) += (*p_delta_weight_per_thread);
			*(p_delta_weight_per_thread++) = 0.0f;
		}
		float* p_delta_bias_per_thread = &conn->delta_bias_per_thread[k][0];
		for (int j = 0; j < s2; ++j)
		{
			conn->delta_bias[j] += (*p_delta_bias_per_thread);
			*(p_delta_bias_per_thread++) = 0.0f;
		}
	}
	float* p_weight = &conn->weight2d[0];
	const float* p_weight_end = p_weight + s;
	float* p_delta_weight = &conn->delta_weight2d[0];
	while (p_weight != p_weight_end)
	{
		*p_delta_weight -= r * *(p_weight);
		*(p_weight++) += *p_delta_weight;
		*(p_delta_weight++) *= momentum_factor;
	}
	for (int j = 0; j < s2; ++j)
	{
		conn->bias[j] += conn->delta_bias[j];
		conn->delta_bias[j] *= momentum_factor;
	}
}

void neural_network::shuffle(std::vector<std::vector<float>>& inputs, std::vector<std::vector<float>>& labels)
{
	assert(inputs.size() == labels.size());
	for (size_t i = 0; i < inputs.size(); ++i)
	{
		size_t swap_i = (size_t)((rand() % 4096 * 4096 + rand() % 4096) % inputs.size());
		inputs[i].swap(inputs[swap_i]);
		labels[i].swap(labels[swap_i]);
	}
}

void neural_network::forward_propagation_job(neural_network* nn, std::pair<std::vector<float>, std::vector<float>>* flow, const std::vector<std::vector<float>>* inputs, int thread_id, int job_id)
{
	const std::vector<float>& input = (*inputs)[job_id];
	assert(input.size() == nn->input_layer->size);
	memcpy(&flow->first[0], &input[0], sizeof(float) * nn->input_layer->size);
	nn->preprocess(flow->first);
	int slot = job_id % nn->batch_size;
	memcpy(&nn->input_layer->neuron[slot][0], &flow->first[0], sizeof(float) * nn->input_layer->size);
	for (int i = 0; i < nn->connection_count; ++i)
	{
		neural_network_connection* conn = &nn->connections[i];
		if (conn->type == neural_network_connection::full_connection)
			conn->forward_propagation(*flow);
		else
			conn->activation(flow->first);
		memcpy(&conn->next_layer->neuron[slot][0], &flow->first[0], sizeof(float) * conn->next_layer->size);
	}
}

void neural_network::backward_propagation_job(neural_network* nn, std::pair<std::vector<float>, std::vector<float>>* flow, const std::vector<std::vector<float>>* labels, int thread_id, int job_id)
{
	const std::vector<float>& label = (*labels)[job_id];
	assert(label.size() == nn->output_layer->size);
	float training_rate_per_batch = nn->training_rate / nn->batch_size;
	int slot = job_id % nn->batch_size;
	nn->output_layer->prev_connection->prev_layer->calculate_output_delta(label, slot);
	for (int i = nn->connection_count - 2; i >= 0; --i)
	{
		nn->connections[i].back_propagation(thread_id, slot, training_rate_per_batch);
	}
}

void neural_network::update_weight_job(neural_network* nn, std::pair<std::vector<float>, std::vector<float>>* flow, const std::vector<std::vector<float>>* unused, int thread_id, int job_id)
{
	assert(job_id >= 0 && job_id < nn->connection_count&& nn->connections[job_id].type == neural_network_connection::full_connection);
	nn->update_weight(&nn->connections[job_id]);
}

void neural_network::worker_thread(neural_network* nn, std::pair<std::vector<float>, std::vector<float>>* flow, neural_network_thread_arg* arg)
{
	while (!(arg->exit_flag))
	{
		if (arg->work_flag)
		{
			for (int i = arg->begin_job_id; i < arg->end_job_id; ++i)
				arg->job_func(nn, flow, arg->shared_arg, arg->thread_id, i);
			arg->work_flag = false;
		}
	}
}

void neural_network::train(std::vector<std::vector<float>>& inputs, std::vector<std::vector<float>>& labels, int loop)
{
	assert(!layers.empty() && input_layer && output_layer && !connections.empty());
	
	std::vector<std::pair<std::vector<float>, std::vector<float>>> flows;
	flows.resize((size_t)thread_count);
	for (std::pair<std::vector<float>, std::vector<float>>& flow : flows)
	{
		flow.first.resize((size_t)max_layer_size);
		flow.second.resize((size_t)max_layer_size);
	}

	std::vector<neural_network_thread_arg> thread_args(thread_count - 1);
	for (int i = 0; i < thread_count - 1; ++i)
		thread_args[i].thread_id = i;
	std::vector<std::thread> threads;
	threads.reserve((size_t)thread_count - 1);
	for (int i = 0; i < thread_count - 1; ++i)
		threads.emplace_back(worker_thread, this, &flows[i], &thread_args[i]);

	while ((loop--) > 0)
	{
		shuffle(inputs, labels);
		int begin_job_id = 0;
		int final_job_id = (int)inputs.size();
		while (begin_job_id < final_job_id)
		{
			int end_job_id = std::min(final_job_id, begin_job_id + batch_size - training_count_per_batch);
			int job_count = end_job_id - begin_job_id;
			assert(job_count > 0);
			int job_per_thread = job_count / thread_count;

			if (job_per_thread > 0 && thread_count > 1)
			{
				const std::function<void(neural_network*, std::pair<std::vector<float>, std::vector<float>>*, const std::vector<std::vector<float>>*, int, int)> job_func[2] = { forward_propagation_job , backward_propagation_job };
				const std::vector<std::vector<float>>* shared_arg[2] = { &inputs, &labels };

				for (int j = 0; j < 2; ++j)
				{
					for (int i = 0; i < thread_count - 1; ++i)
					{
						thread_args[i].job_func = job_func[j];
						thread_args[i].shared_arg = shared_arg[j];
						thread_args[i].begin_job_id = begin_job_id + i * job_per_thread;
						thread_args[i].end_job_id = thread_args[i].begin_job_id + job_per_thread;
						thread_args[i].work_flag = true;
					}
					for (int i = begin_job_id + (thread_count - 1) * job_per_thread; i < end_job_id; ++i)
					{
						job_func[j](this, &flows.back(), shared_arg[j], thread_count - 1, i);
					}
					while (true)
					{
						int i;
						for (i = 0; i < thread_count - 1; ++i)
							if (thread_args[i].work_flag)
								break;
						if (i == thread_count - 1)
							break;
					}
				}
			}
			else
			{
				for (int i = begin_job_id; i < end_job_id; ++i)
				{
					forward_propagation_job(this, &flows.back(), &inputs, thread_count - 1, i);
					backward_propagation_job(this, &flows.back(), &labels, thread_count - 1, i);
				}
			}

			training_count_per_batch += job_count;
			if (training_count_per_batch >= batch_size)
			{
				int t = 0;
				for (int i = 0; i < connection_count; ++i)
				{
					if (connections[i].type == neural_network_connection::full_connection)
					{
						if (t < thread_count - 1)
						{
							thread_args[t].job_func = update_weight_job;
							thread_args[t].begin_job_id = i;
							thread_args[t].end_job_id = i + 1;
							thread_args[t].work_flag = true;
							++t;
						}
						else
						{
							update_weight(&connections[i]);
						}
					}
				}

				while (true)
				{
					int i;
					for (i = 0; i < t; ++i)
						if (thread_args[i].work_flag)
							break;
					if (i == t)
						break;
				}

				training_rate = std::max(training_rate * decay_rate, min_training_rate);
				training_count_per_batch = 0;
			}

			begin_job_id = end_job_id;
		}
	}

	for (int i = 0; i < thread_count - 1; ++i)
	{
		assert(!thread_args[i].work_flag);
		thread_args[i].exit_flag = true;
	}
	for (int i = 0; i < thread_count - 1; ++i)
	{
		threads[i].join();
	}
}

void neural_network::train(std::pair<std::vector<float>, std::vector<float>>& flow, const std::vector<float>& label)
{
	assert(!layers.empty() && input_layer && output_layer && !connections.empty() && (int)flow.first.size() >= max_layer_size && (int)flow.second.size() >= max_layer_size);

	preprocess(flow.first);
	memcpy(&input_layer->neuron[0][0], &flow.first[0], sizeof(float) * input_layer->size);
	for (int i = 0; i < connection_count; ++i)
	{
		neural_network_connection* conn = &connections[i];
		if (conn->type == neural_network_connection::full_connection)
			conn->forward_propagation(flow);
		else
			conn->activation(flow.first);
		memcpy(&conn->next_layer->neuron[0][0], &flow.first[0], sizeof(float) * conn->next_layer->size);
	}

	float training_rate_per_batch = training_rate / batch_size;
	output_layer->prev_connection->prev_layer->calculate_output_delta(label, 0);
	for (int i = connection_count - 2; i >= 0; --i)
	{
		connections[i].back_propagation(0, 0, training_rate_per_batch);
	}

	if ((++training_count_per_batch) >= batch_size)
	{
		for (int i = 0; i < connection_count; ++i)
		{
			if (connections[i].type == neural_network_connection::full_connection)
			{
				update_weight(&connections[i]);
			}
		}
		training_rate = std::max(training_rate * decay_rate, min_training_rate);
		training_count_per_batch = 0;
	}
}

void neural_network::test(std::pair<std::vector<float>, std::vector<float>>& flow)const
{
	assert(!layers.empty() && input_layer && output_layer && !connections.empty() && (int)flow.first.size() >= max_layer_size && (int)flow.second.size() >= max_layer_size);
	preprocess(flow.first);
	for (int i = 0; i < connection_count; ++i)
	{
		const neural_network_connection* conn = &connections[i];
		if (conn->type == neural_network_connection::full_connection)
			conn->forward_propagation(flow);
		else
			conn->activation(flow.first);
	}
}

void neural_network_layer::deserialize(const std::vector<unsigned char>& data, size_t& offset)
{
	memcpy(&size, &data[offset], sizeof(int));
	offset += sizeof(int);
}

void neural_network_connection::deserialize(const std::vector<unsigned char>& data, size_t& offset)
{
	memcpy(&type, &data[offset], sizeof(int));
	offset += sizeof(int);
	memcpy(&size1, &data[offset], sizeof(int));
	offset += sizeof(int);
	memcpy(&size2, &data[offset], sizeof(int));
	offset += sizeof(int);
	if (size1 > 0 && size2 > 0)
	{
		weight2d_root = new float[size1 * size2 + 4];
		weight2d = weight2d_root;
		while (((size_t)weight2d) % 16)
			++weight2d;
		bias.resize((size_t)size2, 0.0f);
		memcpy(&weight2d[0], &data[offset], sizeof(float) * size1 * size2);
		offset += sizeof(float) * size1 * size2;
		memcpy(&bias[0], &data[offset], sizeof(float) * size2);
		offset += sizeof(float) * size2;
	}
}

void neural_network::deserialize(const std::vector<unsigned char>& data)
{
	assert(layers.empty() && !input_layer && !output_layer && connections.empty());
	size_t offset = 0;
	memcpy(&total_layer_count, &data[offset], sizeof(int));
	offset += sizeof(int);
	connection_count = total_layer_count - 1;
	assert(connection_count > 0);
	layers.resize((size_t)total_layer_count);
	connections.resize((size_t)connection_count);
	input_layer = &layers.front();
	output_layer = &layers.back();
	for (int i = 0; i < connection_count; ++i)
	{
		connections[i].prev_layer = &layers[i];
		connections[i].next_layer = &layers[i + 1];
		layers[i].next_connection = &connections[i];
		layers[i + 1].prev_connection = &connections[i];
	}
	for (int i = 0; i < total_layer_count; ++i)
	{
		layers[i].deserialize(data, offset);
		max_layer_size = std::max(max_layer_size, layers[i].size);
	}
	max_layer_size += 4;
	for (int i = 0; i < connection_count; ++i)
	{
		connections[i].deserialize(data, offset);
	}
	average_standarderrors.resize((size_t)input_layer->size * 2);
	memcpy(&average_standarderrors[0], &data[offset], sizeof(float) * average_standarderrors.size());
	offset += sizeof(float) * average_standarderrors.size();
	assert(output_layer->prev_connection->type == neural_network_connection::activation_sigmoid || output_layer->prev_connection->type == neural_network_connection::output_linear || output_layer->prev_connection->type == neural_network_connection::output_softmax);
	assert(offset == data.size() && offset == estimate_byte());
}

size_t neural_network::estimate_byte() const
{
	size_t result = sizeof(int) + sizeof(float) * average_standarderrors.size();
	for (int i = 0; i < total_layer_count; ++i)
	{
		result += layers[i].estimate_byte();
	}
	for (int i = 0; i < connection_count; ++i)
	{
		result += connections[i].estimate_byte();
	}
	return result;
}

void neural_network_layer::serialize(std::vector<unsigned char>& data, size_t& offset) const
{
	memcpy(&data[offset], &size, sizeof(int));
	offset += sizeof(int);
}

void neural_network_connection::serialize(std::vector<unsigned char>& data, size_t& offset) const
{
	memcpy(&data[offset], &type, sizeof(int));
	offset += sizeof(int);
	memcpy(&data[offset], &size1, sizeof(int));
	offset += sizeof(int);
	memcpy(&data[offset], &size2, sizeof(int));
	offset += sizeof(int);
	if (size1 > 0 && size2 > 0)
	{
		memcpy(&data[offset], &weight2d[0], sizeof(float) * size1 * size2);
		offset += sizeof(float) * size1 * size2;
		memcpy(&data[offset], &bias[0], sizeof(float) * size2);
		offset += sizeof(float) * size2;
	}
}

void neural_network::serialize(std::vector<unsigned char>& data) const
{
	data.resize(estimate_byte(), (unsigned char)0);
	size_t offset = 0;
	memcpy(&data[offset], &total_layer_count, sizeof(int));
	offset += sizeof(int);
	for (int i = 0; i < total_layer_count; ++i)
	{
		layers[i].serialize(data, offset);
	}
	for (int i = 0; i < connection_count; ++i)
	{
		connections[i].serialize(data, offset);
	}
	memcpy(&data[offset], &average_standarderrors[0], sizeof(float) * average_standarderrors.size());
	offset += sizeof(float) * average_standarderrors.size();
	assert(offset == data.size());
}

void neural_network::read_from_file(const char* file)
{
	std::vector<unsigned char> data;
	FILE* f = fopen(file, "rb");
	assert(f);
	fseek(f, 0, SEEK_END);
	size_t size = ftell(f);
	assert(size);
	fseek(f, 0, SEEK_SET);
	data.resize(size, (unsigned char)0);
	size_t read_size = fread(&data[0], 1, size, f);
	assert(read_size <= size);
	if (read_size < size)
	{
		data.resize(read_size);
	}
	fclose(f);
	deserialize(data);
}

void neural_network::write_to_file(const char* file) const
{
	std::vector<unsigned char> data;
	serialize(data);
	FILE* f = fopen(file, "wb");
	assert(f);
	fwrite(&data[0], sizeof(unsigned char), data.size(), f);
	fclose(f);
}

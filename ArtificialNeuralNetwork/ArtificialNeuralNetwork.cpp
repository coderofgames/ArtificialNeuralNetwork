// ArtificialNeuralNetwork.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <vector>


using std::cout;
using std::endl;
using std::vector;

class vector2d
{
public:

	float v[2];

	float operator[](unsigned int idx) { return (idx < 2?v[idx]:0.0f); }
	void operator=(vector2d b){ v[0] = b.v[0]; v[1] = b.v[1]; }
};

float sigmoid(float x)
{
	return 1.f / (1.f + exp(-x));
}

int main(int argc, char* argv[])
{

	// TRAINING SET FOR EXCLUSIVE OR GATE
	vector<vector2d > training;
	training.push_back(vector2d{ { 0.f, 0.f } });
	training.push_back(vector2d{ { 0.f, 1.f } });
	training.push_back(vector2d{ { 1.f, 0.f } });
	training.push_back(vector2d{ { 1.f, 1.f } });

	float desired_output[4] = { 0.f, 1.f, 1.f, 0.f };

	// ==========================================

	float input_vector[2];
	float weight_mat_1[2][2];
	float hidden_layer[2];
	float weight_mat_2[2];
	float deltas[3];
	float thetas[3];
	float output_neuron;

	weight_mat_1[0][0] = 0.5f;
	weight_mat_1[0][1] = 0.9f;
	weight_mat_1[1][0] = 0.4f;
	weight_mat_1[1][1] = 1.0f;

	weight_mat_2[0] = -1.2f;
	weight_mat_2[1] = 1.1f;

	thetas[0] = 0.8f;
	thetas[1] = -0.1f;
	thetas[2] = 0.3f;
	

	float output_error = 0.0f;

	float weight_mat_1_delta[3][2];
	float weight_mat_2_delta[3];

	float alpha = 0.3f;

	float sum_squared_errors = 0.0f;


	for (int p = 0; p < 5000; p++)
	{
		sum_squared_errors = 0.0f;
		for (int q = 0; q < 4; q++)
		{
			input_vector[0] = training[q].v[0];
			input_vector[1] = training[q].v[1];

			float sum[3] = { 0.0f, 0.0f, 0.0f };
			 
			for (int i = 0; i < 2; i++)
			{
				for (int j = 0; j < 2; j++)
				{
					sum[i] += input_vector[j] * weight_mat_1[j][i];		
				}
				hidden_layer[i] = sigmoid(sum[i] - thetas[i]);
				sum[2] += hidden_layer[i] * weight_mat_2[i];
			}

			output_neuron = sigmoid(sum[2] - thetas[2]);

#ifdef VERBOSE
			if (p % 250 == 0)
			{
				cout << "Hidden Layer: " << hidden_layer[0] << ", " << hidden_layer[1] << endl;
				cout << "Output: " << output_neuron << endl;
			}
#endif
			output_error = desired_output[q] - output_neuron;

			sum_squared_errors += output_error*output_error;

			// back propogate
			
			deltas[2] = output_neuron * (1 - output_neuron) * output_error;
			deltas[0] = hidden_layer[0] * (1 - hidden_layer[0]) * weight_mat_2[0] * deltas[2];
			deltas[1] = hidden_layer[1] * (1 - hidden_layer[1]) * weight_mat_2[1] * deltas[2];

			// weight deltas

			weight_mat_2_delta[0] = alpha*hidden_layer[0] * deltas[2];
			weight_mat_2_delta[1] = alpha*hidden_layer[1] * deltas[2];
			weight_mat_2_delta[2] = alpha* (-1.0f) * deltas[2];
#ifdef VERBOSE
			if (p % 250 == 0)
				cout << "Delta 5: " << deltas[2] << ", Delta 3: " << deltas[0] << ", Delta 4: " << deltas[1] << endl;
#endif
			weight_mat_1_delta[0][0] = alpha * input_vector[0] * deltas[0];
			weight_mat_1_delta[1][0] = alpha * input_vector[1] * deltas[0];
			weight_mat_1_delta[2][0] = alpha * (-1.0f) * deltas[0];

			weight_mat_1_delta[0][1] = alpha * input_vector[0] * deltas[1];
			weight_mat_1_delta[1][1] = alpha * input_vector[1] * deltas[1];
			weight_mat_1_delta[2][1] = alpha * (-1.0f) * deltas[1];

			weight_mat_2_delta[2] = alpha * (-1) * deltas[2];
#ifdef VERBOSE
			if (p % 250 == 0)
			{
				cout << "Weight Matrix Deltas 00: " << weight_mat_1_delta[0][0] << endl;
				cout << "Weight Matrix Deltas 10: " << weight_mat_1_delta[1][0] << endl;
				cout << "Weight Matrix Deltas 20: " << weight_mat_1_delta[2][0] << endl;

				cout << "Weight Matrix Deltas 01: " << weight_mat_1_delta[0][1] << endl;
				cout << "Weight Matrix Deltas 11: " << weight_mat_1_delta[1][1] << endl;
				cout << "Weight Matrix Deltas 21: " << weight_mat_1_delta[2][1] << endl;
			}
#endif
			// update weights

			for (int i = 0; i < 2; i++)
				for (int j = 0; j < 2; j++)
					weight_mat_1[i][j] = weight_mat_1[i][j] + weight_mat_1_delta[i][j];

			weight_mat_2[0] = weight_mat_2[0] + weight_mat_2_delta[0];
			weight_mat_2[1] = weight_mat_2[1] + weight_mat_2_delta[1];

			thetas[0] = thetas[0] + weight_mat_1_delta[2][0];
			thetas[1] = thetas[1] + weight_mat_1_delta[2][1];

			thetas[2] = thetas[2] + weight_mat_2_delta[2];
#ifdef VERBOSE
			if (p % 250 == 0)
			{
				cout << "Weight Matrix 00: " << weight_mat_1[0][0] << endl;
				cout << "Weight Matrix 10: " << weight_mat_1[1][0] << endl;
				//cout << "Weight Matrix Deltas 20: " << weight_mat_1_delta[2][0] << endl;

				cout << "Weight Matrix 01: " << weight_mat_1[0][1] << endl;
				cout << "Weight Matrix 11: " << weight_mat_1[1][1] << endl;
				//cout << "Weight Matrix Deltas 21: " << weight_mat_1_delta[2][1] << endl;
				cout << "Theta 3: " << thetas[0] << endl;
				cout << "Theta 4: " << thetas[1] << endl;

				cout << "Theta 5: " << thetas[2] << endl;

				cout << "Weight Matrix2 0: " << weight_mat_2[0] << endl;
				cout << "Weight Matrix2 1: " << weight_mat_2[1] << endl;
			}
#endif
		}
		cout << sum_squared_errors << endl;
		if (sum_squared_errors < 0.03)
		{
			cout << "Finished on iteration: " << p << ", with sum squared errors less than 0.03" << endl;
			break;
		}


	}
	
	
	return 0;
}


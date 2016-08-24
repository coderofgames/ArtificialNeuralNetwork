// ArtificialNeuralNetwork.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "Timer.h"
#include <iostream>
#include <vector>


#include "Utils.h"
#include "my_matrix.h"

float sigmoid(float x)
{
	return 1.f / (1.f + exp(-x));
}

matrix& sigmoid(matrix &out, matrix &m)
{
	if (out.NumColumns() != m.NumColumns() || out.NumRows() != m.NumRows())
	{
		out.destroy();
		out.m_sizeX = m.m_sizeX;
		out.m_sizeY = m.m_sizeY;
		out.create();
	}
	for (int i = 0; i < m.NumRows(); i++)
		for (int j = 0; j < m.NumColumns(); j++)
			out(i, j) = 1.f / (1.f + exp(-m(i, j)));

	return out;
}

matrix& anti_sigmoid(matrix& out, matrix &m)
{
	if (out.NumColumns() != m.NumColumns() || out.NumRows() != m.NumRows())
	{
		out.destroy();
		out.m_sizeX = m.m_sizeX;
		out.m_sizeY = m.m_sizeY;
		out.create();
	}
	for (int i = 0; i < m.NumRows(); i++)
		for (int j = 0; j < m.NumColumns(); j++)
			out(i, j) = m(i, j) * (1 - m(i, j));

	return out;
}

void Compute_Simple_XOR_network_version_1()
{
	Timer timer;

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

	timer.Start();
	//	Sleep(2000);


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
		//cout << sum_squared_errors << endl;
		if (sum_squared_errors < 0.03)
		{
			timer.Update();
			timer.Stop();
			cout << "Finished on iteration: " << p << ", with sum squared errors less than 0.03" << endl << "Total calculation performed in " << timer.GetTimeDelta() << " seconds" << endl;
			break;
		}


	}


}

void Compute_Simple_XOR_network_version_2()
{

	const unsigned int num_inputs = 2;
	const unsigned int num_hidden = 3;
	const unsigned int num_output = 1;

	Timer timer;

	// TRAINING SET FOR EXCLUSIVE OR GATE
	vector<vector2d > training;
	training.push_back(vector2d{ { 0.f, 0.f } });
	training.push_back(vector2d{ { 0.f, 1.f } });
	training.push_back(vector2d{ { 1.f, 0.f } });
	training.push_back(vector2d{ { 1.f, 1.f } });

	float desired_output[4] = { 0.f, 1.f, 1.f, 0.f };

	// ==========================================

	float input_vector[2];
	float weight_mat_1[num_inputs][num_hidden];
	float hidden_layer[num_hidden];
	float weight_mat_2[num_hidden];
	float deltas[num_hidden + 1];
	float thetas[num_hidden + 1];
	float output_neuron;

	weight_mat_1[0][0] = 0.5f;
	weight_mat_1[0][1] = 0.9f;
	weight_mat_1[0][2] = 0.2f;
	weight_mat_1[1][0] = 0.4f;
	weight_mat_1[1][1] = 1.0f;
	weight_mat_1[1][2] = 0.2f;

	weight_mat_2[0] = -1.2f;
	weight_mat_2[1] = 1.1f;
	weight_mat_2[2] = 0.6f;

	thetas[0] = 0.8f;
	thetas[1] = -0.1f;
	thetas[2] = 0.3f; 
	thetas[3] = 0.3f;


	float output_error = 0.0f;

	float weight_mat_1_delta[num_inputs+1][num_hidden+1];
	float weight_mat_2_delta[num_hidden + 1];

	float alpha = 0.3f;

	float sum_squared_errors = 0.0f;

	timer.Start();
	//	Sleep(2000);

	
	for (int p = 0; p < 10000; p++)
	{
		sum_squared_errors = 0.0f;
		for (int q = 0; q < 4; q++)
		{
			input_vector[0] = training[q].v[0];
			input_vector[1] = training[q].v[1];

			float sum[num_inputs + 1];
			for (int i = 0; i < num_inputs + 1; i++)sum[i] = 0.0f;

			// Feed Forward

			for (int i = 0; i < num_hidden; i++)
			{
				for (int j = 0; j < num_inputs; j++)
				{
					sum[i] += input_vector[j] * weight_mat_1[j][i];
				}
				hidden_layer[i] = sigmoid(sum[i] - thetas[i]);
				sum[num_inputs] += hidden_layer[i] * weight_mat_2[i];
			}

			output_neuron = sigmoid(sum[num_inputs ] - thetas[num_hidden ]);

			output_error = desired_output[q] - output_neuron;

			sum_squared_errors += output_error*output_error;



			// back propogate


			deltas[num_hidden] = output_neuron * (1 - output_neuron) * output_error;

			int i = 0;
			for (i = 0; i < num_hidden; i++)
			{
				// back propogate
				deltas[i] = hidden_layer[i] * (1 - hidden_layer[i]) * weight_mat_2[i] * deltas[num_hidden];
				weight_mat_2_delta[i] = alpha*hidden_layer[i] * deltas[num_hidden + 1];

				for (int j = 0; j < num_inputs; j++)
				{
					// update weight deltas
					weight_mat_1_delta[j][i] = alpha * input_vector[j] * deltas[i];

					// update weights
					weight_mat_1[j][i] = weight_mat_1[j][i] + weight_mat_1_delta[j][i];

				}
				// update delta theta
				weight_mat_1_delta[2][i] = alpha * (-1.0f) * deltas[i];

				// update output layer weights
				weight_mat_2[i] = weight_mat_2[i] + weight_mat_2_delta[i];

				// update thetas
				thetas[i] = thetas[i] + weight_mat_1_delta[2][i];
			}

			weight_mat_2_delta[num_hidden] = alpha* (-1.0f) * deltas[num_hidden];
			thetas[num_hidden] = thetas[num_hidden] + weight_mat_2_delta[num_hidden];


		}


		//cout << sum_squared_errors << endl;
		if (sum_squared_errors < 0.03)
		{
			timer.Update();
			timer.Stop();
			cout << "Finished on iteration: " << p << ", with sum squared errors less than 0.03" << endl << "Total calculation performed in " << timer.GetTimeDelta() << " seconds" << endl;
			return;
		}


	}
	timer.Update();
	timer.Stop();
	cout << "Finished on iteration: 10000 with total sum squared errors: " << sum_squared_errors <<", " << endl << "Total calculation performed in " << timer.GetTimeDelta() << " seconds" << endl;


}


void Compute_Simple_XOR_network_version_3(int num_iterations)
{
	Timer timer;

	// TRAINING SET FOR EXCLUSIVE OR GATE
	vector<vector2d > training;
	training.push_back(vector2d{ { 0.f, 0.f } });
	training.push_back(vector2d{ { 0.f, 1.f } });
	training.push_back(vector2d{ { 1.f, 0.f } });
	training.push_back(vector2d{ { 1.f, 1.f } });

	float desired_output[4] = { 0.f, 1.f, 1.f, 0.f };

	// ==========================================
	matrix input_matrix(1, 2);
	matrix w_m_1_2_(2, 2);
	matrix hidden_layer_(2, 1);
	matrix w_m_2_3_(2, 1);
	matrix out_(1, 1);

	matrix del_3_2_(1, 1);
	matrix del_2_1_(1, 2);
	matrix theta_1_(1, 2);
	matrix theta_2_(1, 1);
	
	w_m_1_2_(0,0) = 0.5f;
	w_m_1_2_(0, 1) = 0.9f;
	
	w_m_1_2_(1, 0) = 0.4f;
	w_m_1_2_(1, 1) = 1.0f;


	w_m_2_3_(0, 0) = -1.2f;
	w_m_2_3_(1, 0) = 1.1f;

	theta_1_(0, 0) = 0.8f;
	theta_1_(0,1) = -0.1f;
	theta_2_(0,0) = 0.3f;


	float output_error = 0.0f;

	matrix w_m_delta_1_(3, 2);
	matrix w_m_delta_2_(3, 1);
	

	float alpha = 0.3f;

	float sum_squared_errors = 0.0f;

	timer.Start();
	//	Sleep(2000);


	for (int p = 0; p < num_iterations; p++)
	{
		sum_squared_errors = 0.0f;
		for (int q = 0; q < 4; q++)
		{
			input_matrix(0,0)= training[q].v[0];
			input_matrix(0, 1)= training[q].v[1];


			// theta's must be in the weight matrix
			hidden_layer_ = input_matrix * w_m_1_2_ - theta_1_;

			// computes the elementwise sigmoid of the sum 
			sigmoid(hidden_layer_, hidden_layer_);

			out_ = hidden_layer_ * w_m_2_3_ - theta_2_;
			out_(0, 0) = sigmoid(out_(0, 0));
	
//#define VERBOSE
#ifdef VERBOSE
		if (p % 250 == 0)
			{
				cout << "hidden layer: " << endl;
				hidden_layer_.print();
				cout << "output: " << endl;
				out_.print();
				cout << endl;
			}
#endif
			output_error = desired_output[q] - out_(0, 0);

			sum_squared_errors += output_error * output_error;

			// back propogate
			del_3_2_(0, 0) = out_(0, 0) * (1 - out_(0, 0)) * output_error;

			// calculation of the debug values. 
			float correct_val_1 = hidden_layer_(0, 0)*(1 - hidden_layer_(0, 0)) * w_m_2_3_(0, 0) * del_3_2_(0, 0);
			float correct_val_2 = hidden_layer_(0, 1)*(1 - hidden_layer_(0, 1)) * w_m_2_3_(1, 0) * del_3_2_(0, 0);
			
			// computes the elementwise differentiation of the sigmoid function
			anti_sigmoid( del_2_1_, hidden_layer_ );
			
			// the del_2_1_ vector is expanded to inhabit the diagonal of the identity
			// matrix for the next matrix operation
			matrix ident_22(2, 2);
			for (int i = 0; i < 2; i++)
			{
				for (int h = 0; h < 2; h++)
				{
					if (i == h) ident_22(i, h) = del_2_1_(0, i);
					else ident_22(i, h) = 0.0f;
				}
			}

			del_2_1_ = ident_22 * w_m_2_3_ * del_3_2_(0, 0);

			del_2_1_.transpose();

			w_m_delta_2_(0, 0) = alpha * hidden_layer_(0, 0) * del_3_2_(0, 0);
			w_m_delta_2_(1, 0) = alpha * hidden_layer_(0, 1) * del_3_2_(0, 0);
			w_m_delta_2_(2, 0) = alpha * (-1.0f) * del_3_2_(0, 0);
#ifdef VERBOSE
			if (p % 250 == 0)
			{
				cout << "deltas: " << endl;
				del_2_1_.print();
				cout  << endl;
				del_3_2_.print();
				cout << endl;
			}
#endif

#undef VERBOSE
			// this operation could be bunched up into a matrix operation, this
			// shall be left to the next function
			w_m_delta_1_(0, 0) = alpha * input_matrix(0, 0) * del_2_1_(0, 0);
			w_m_delta_1_(1, 0) = alpha * input_matrix(0, 1) * del_2_1_(0, 0);
			w_m_delta_1_(2, 0) = alpha * (-1.0f) * del_2_1_(0, 0);

			w_m_delta_1_(0, 1) = alpha * input_matrix(0, 0) * del_2_1_(0, 1);
			w_m_delta_1_(1, 1) = alpha * input_matrix(0, 1) * del_2_1_(0, 1);
			w_m_delta_1_(2, 1) = alpha * (-1.0f) * del_2_1_(0, 1);

//			weight_mat_2_delta[2] = alpha * (-1) * deltas[2];
#ifdef VERBOSE
			if (p % 250 == 0)
			{
				w_m_delta_1_.print(); // untested
			}
#endif
			// update weights

			for (int i = 0; i < 2; i++)
			{
				for (int j = 0; j < 2; j++)
				{
					w_m_1_2_(i, j) = w_m_1_2_(i, j) + w_m_delta_1_(i, j);// weight_mat_1[i][j] = weight_mat_1[i][j] + weight_mat_1_delta[i][j];
				}
			}
			// it is clear that the operation above is an elementwise matrix addition
			// but the w_m_1_2_ matrix is not the same size as the w_m_delta_1_ 
			// because of the storage of theta bias values ... this is the correct
			// place to store the theta bias values however I have left this stage of
			// development to the next function, where i will attempt to further generalize
			// the matrix operations to facilitate an arbitrary number of inputs, outputs, 
			// hidden layer neurons and number of hidden layers


			w_m_2_3_(0, 0) = w_m_2_3_(0, 0) + w_m_delta_2_(0, 0);
			w_m_2_3_(1, 0) = w_m_2_3_(1, 0) + w_m_delta_2_(1, 0); //weight_mat_2[1] = weight_mat_2[1] + weight_mat_2_delta[1];

			theta_1_(0, 0) = theta_1_(0, 0) + w_m_delta_1_(2, 0);
			theta_1_(0, 1) = theta_1_(0, 1) + w_m_delta_1_(2, 1);

			theta_2_(0, 0) = theta_2_(0, 0) + w_m_delta_2_(2, 0);
#ifdef VERBOSE
			if (p % 250 == 0)
			{
				
			}
#endif


		}
		//cout << sum_squared_errors << endl;
		if (sum_squared_errors < 0.03)
		{
			timer.Update();
			timer.Stop();
			cout << "Finished on iteration: " << p << ", with sum squared errors less than 0.03" << endl << "Total calculation performed in " << timer.GetTimeDelta() << " seconds" << endl;
			break;
		}


	}


}

void Compute_Simple_XOR_network_version_4(int num_iterations)
{
	Timer timer;

	// TRAINING SET FOR EXCLUSIVE OR GATE
	vector<vector2d > training;
	training.push_back(vector2d{ { 0.f, 0.f } });
	training.push_back(vector2d{ { 0.f, 1.f } });
	training.push_back(vector2d{ { 1.f, 0.f } });
	training.push_back(vector2d{ { 1.f, 1.f } });

	float desired_output[4] = { 0.f, 1.f, 1.f, 0.f };

	// ==========================================
	matrix input_matrix(1, 3);
	matrix w_m_1_2_(3, 3);
	matrix hidden_layer_(3, 1);
	matrix w_m_2_3_(3, 1);
	matrix out_(1, 1);

	matrix del_3_2_(1, 1);
	matrix del_2_1_(1, 3);
	matrix theta_1_(1, 2);
	matrix theta_2_(1, 1);

	w_m_1_2_(0, 0) = 0.5f;
	w_m_1_2_(0, 1) = 0.9f;
	w_m_1_2_(0, 2) = 0.0f;

	w_m_1_2_(1, 0) = 0.4f;
	w_m_1_2_(1, 1) = 1.0f;
	w_m_1_2_(1, 2) = 0.0f;

	w_m_1_2_(2, 0) = 0.8f;// theta 1
	w_m_1_2_(2, 1) = -0.1f;//// theta 2
	w_m_1_2_(2, 2) = 1.0f;

	w_m_2_3_(0, 0) = -1.2f;
	w_m_2_3_(1, 0) = 1.1f;
	w_m_2_3_(2, 0) = 0.3f; // theta for output 

	theta_1_(0, 0) = 0.8f;
	theta_1_(0, 1) = -0.1f;
	theta_2_(0, 0) = 0.3f;


	float output_error = 0.0f;

	matrix w_m_delta_1_(3, 3);
	matrix w_m_delta_2_(3, 1);


	float alpha = 0.3;

	float sum_squared_errors = 0.0f;

	timer.Start();
	//	Sleep(2000);

	float last_sum_squared_errors = 0.0f;
	int positive_error_delta_count = 0;
	int negative_error_delta_count = 0;

	for (int p = 0; p < num_iterations; p++)
	{
		sum_squared_errors = 0.0f;
		
		for (int q = 0; q < 4; q++)
		{
			input_matrix(0, 0) = training[q].v[0];
			input_matrix(0, 1) = training[q].v[1];
			input_matrix(0, 2) = -1.0f; // bias is always -1

			float sum[3] = { 0.0f, 0.0f, 0.0f };



			hidden_layer_ = input_matrix * w_m_1_2_;// -theta_1_;


			sigmoid(hidden_layer_, hidden_layer_);

			// OVERWRITE 3rd INPUT
			hidden_layer_(0, 2) = -1.0f;

			out_ = hidden_layer_ * w_m_2_3_;

			out_(0, 0) = sigmoid(out_(0, 0));

#ifdef VERBOSE
			if (p % 250 == 0)
			{
				hidden_layer_.print();
				cout<<endl;
				out_(0, 0).print();
				cout<<endl;
			}
#endif
			output_error = desired_output[q] - out_(0, 0);

			sum_squared_errors += output_error * output_error;


			// back propogate

			anti_sigmoid(del_3_2_, out_);

			del_3_2_ = del_3_2_ * output_error;



			anti_sigmoid(del_2_1_, hidden_layer_);

			// put the vector on the diagonal for next operation ...
			matrix ident_22(3, 3);
			for (int i = 0; i < 3; i++)
			{
				for (int h = 0; h < 3; h++)
				{
					if (i == h) ident_22(i, h) = del_2_1_(0, i);
					else ident_22(i, h) = 0.0f;
				}
			}

			del_2_1_ = ident_22 * w_m_2_3_ * del_3_2_(0, 0);


			// weight deltas

			w_m_delta_2_ = hidden_layer_ * alpha * del_3_2_(0, 0);

			w_m_delta_2_.transpose();


#ifdef VERBOSE
			if (p % 250 == 0)
			{ 
				del_2_1_.print();
				cout<<endl;
				de_3_2_.print();
				cout << endl;
			}
#endif
		
			w_m_delta_1_ = del_2_1_ * input_matrix   * alpha;

			w_m_delta_1_.transpose();

#ifdef VERBOSE
			if (p % 250 == 0)
			{
				w_m_delta_1_.print();
				cout<<endl;
				w_m_delta_2_.print();
				cout << endl;
			}
#endif
			// update weights

			w_m_1_2_ = w_m_1_2_ + w_m_delta_1_;// 


			w_m_2_3_ = w_m_2_3_ + w_m_delta_2_;

#ifdef VERBOSE
			if (p % 250 == 0)
			{
				w_m_1_2_.print();
				cout<<endl;
				w_m_2_3_.print();
				cout << endl;
			}
#endif
		}

		//cout << sum_squared_errors << endl;
		if (sum_squared_errors < 0.03)
		{
			timer.Update();
			timer.Stop();
			cout << "Finished on iteration: " << p << ", with sum squared errors less than 0.03" << endl << "Total calculation performed in " << timer.GetTimeDelta() << " seconds" << endl;
			break;
		}
	}
}

//#define VERBOSE
void Compute_Simple_XOR_network_version_5(int num_iterations)
{
	Timer timer;

	// TRAINING SET FOR EXCLUSIVE OR GATE
	vector<vector2d > training;
	training.push_back(vector2d{ { 0.f, 0.f } });
	training.push_back(vector2d{ { 0.f, 1.f } });
	training.push_back(vector2d{ { 1.f, 0.f } });
	training.push_back(vector2d{ { 1.f, 1.f } });

	float desired_output[4] = { 0.f, 1.f, 1.f, 0.f };

	int input_data_size = 1;
	int num_inputs = 2;
	int num_hidden = 2;
	int num_outputs = 1;

	// ==========================================
	matrix input_matrix( 1, num_inputs + 1 );

	matrix w_m_1_2_(num_inputs + 1, num_hidden + 1);
	
	matrix hidden_layer_(num_hidden+1, 1);
	
	matrix w_m_2_3_(num_hidden + 1, num_outputs);
	
	matrix out_(num_outputs, num_outputs);

	matrix del_3_2_(num_outputs, num_outputs);
	matrix del_2_1_(1, num_hidden + 1);
	
	/*for (int i = 0; i < num_inputs+1; i++)
	{
		for (int j = 0; j < num_hidden + 1; j++)
		{
			w_m_1_2_(i, j) = RandomFloat(-1.2, 1.2);
		}
	}
	*/
	w_m_1_2_(0, 0) = 0.5f;
	w_m_1_2_(0, 1) = 0.9f;
	w_m_1_2_(0, 2) = 0.0f;

	w_m_1_2_(1, 0) = 0.4f;
	w_m_1_2_(1, 1) = 1.0f;
	w_m_1_2_(1, 2) = 0.0f;

	w_m_1_2_(2, 0) = 0.8f;// theta 1
	w_m_1_2_(2, 1) = -0.1f;//// theta 2
	w_m_1_2_(2, 2) = 1.0f;
	
	/*
	for (int i = 0; i < num_hidden + 1; i++)
	{
		w_m_2_3_(i, 0) = RandomFloat(-1.2, 1.2);
	}
	*/
	w_m_2_3_(0, 0) = -1.2f;
	w_m_2_3_(1, 0) = 1.1f;
	w_m_2_3_(2, 0) = 0.3f; // theta for output 
	
	


	float output_error = 0.0f;

	matrix w_m_delta_1_(3, 3);
	matrix w_m_delta_2_(3, 1);


	float alpha = 0.1f;
	float beta = 0.95f;

	float sum_squared_errors = 0.0f;


	timer.Start();
	
	//	Sleep(2000);

	float last_sum_squared_errors = 0.0f;
	int positive_error_delta_count = 0;
	int negative_error_delta_count = 0;
	int alternation_count = 0;

	for (int p = 0; p < num_iterations; p++)
	{
		sum_squared_errors = 0.0f;
		for (int q = 0; q < 4; q++)
		{
			input_matrix(0, 0) = training[q].v[0];
			input_matrix(0, 1) = training[q].v[1];
			input_matrix(0, 2) = -1.0f; // bias is always -1

			float sum[3] = { 0.0f, 0.0f, 0.0f };



			hidden_layer_ = input_matrix * w_m_1_2_;// -theta_1_;


			sigmoid(hidden_layer_, hidden_layer_);

			// OVERWRITE 3rd INPUT
			hidden_layer_(0, 2) = -1.0f;

			out_ = hidden_layer_ * w_m_2_3_;

			sigmoid(out_, out_);

#ifdef VERBOSE
			if (p % 250 == 0)
			{
				hidden_layer_.print();
				cout<<endl;
				out_.print();
				cout<<endl;
			}
#endif
			output_error = desired_output[q] - out_(0, 0);

			sum_squared_errors += output_error * output_error;


			// back propogate

			anti_sigmoid(del_3_2_, out_);

			del_3_2_ = del_3_2_ * output_error;



			anti_sigmoid(del_2_1_, hidden_layer_);

			// put the vector on the diagonal for next operation ...
			matrix ident_22(3, 3);
			for (int i = 0; i < 3; i++)
			{
				for (int h = 0; h < 3; h++)
				{
					if (i == h) ident_22(i, h) = del_2_1_(0, i);
					else ident_22(i, h) = 0.0f;
				}
			}

			
			del_2_1_ = ident_22 * w_m_2_3_ * del_3_2_(0, 0);


			// weight deltas

			w_m_delta_2_.transpose();

			w_m_delta_2_ = w_m_delta_2_ * beta + del_3_2_* hidden_layer_ * alpha ;

			w_m_delta_2_.transpose();


#ifdef VERBOSE
			if (p % 250 == 0)
			{
				del_2_1_.print();
				cout << endl;
				del_3_2_.print();
				cout << endl;
			}
#endif
			w_m_delta_1_.transpose();

			w_m_delta_1_ = w_m_delta_1_ * beta  + del_2_1_ * input_matrix   * alpha;

			w_m_delta_1_.transpose();

#ifdef VERBOSE
			if (p % 250 == 0)
			{
				w_m_delta_1_.print();
				cout << endl;
				w_m_delta_2_.print();
				cout << endl;
			}
#endif
			// update weights

			w_m_1_2_ = w_m_1_2_ + w_m_delta_1_;// 


			w_m_2_3_ = w_m_2_3_ + w_m_delta_2_;

#ifdef VERBOSE
			if (p % 250 == 0)
			{
				w_m_1_2_.print();
				cout << endl;
				w_m_2_3_.print();
				cout << endl;
			}
#endif
		}
		if (sum_squared_errors > last_sum_squared_errors*1.04) alpha *= 0.7;
		if (sum_squared_errors < last_sum_squared_errors) alpha *= 1.05;
		// calculate the change in sum_squared_errors
		float delta_sum_square_errors = sum_squared_errors - last_sum_squared_errors;
		last_sum_squared_errors = sum_squared_errors;
		if (delta_sum_square_errors > 0.0f)
		{
			if (positive_error_delta_count == 0) {
				alternation_count++;
			}
			else{
				alternation_count = 0;
			}
			positive_error_delta_count++;
			negative_error_delta_count = 0;
		}
		else
		{
			if (negative_error_delta_count == 0) {
				alternation_count++;
			}
			else{
				alternation_count = 0;
			}
			negative_error_delta_count++;
			positive_error_delta_count = 0;
		}

		// determine change in learning rate
		if (positive_error_delta_count >= 2 || negative_error_delta_count >= 2)
		{
			alpha += 0.1;
			if (alpha > 1.0f) alpha = 1.0f;
		}
		else if (alternation_count >= 2)
		{
			alpha -= 0.1;
			if (alpha < 0.0f) alpha = 0.01;
		}

		//cout << sum_squared_errors << endl;
		if (sum_squared_errors < 0.001)
		{
			timer.Update();
			timer.Stop();
			cout << "Finished on iteration: " << p << ", with sum squared errors less than 0.001" << endl << "Total calculation performed in " << timer.GetTimeDelta() << " seconds" << endl;
			break;
		}
	}
		
}


void Compute_Simple_XOR_network_version_6(int num_iterations)
{
	Timer timer;

	// TRAINING SET FOR EXCLUSIVE OR GATE
	vector<vector2d > training;
	training.push_back(vector2d{ { 0.f, 0.f } });
	training.push_back(vector2d{ { 0.f, 1.f } });
	training.push_back(vector2d{ { 1.f, 0.f } });
	training.push_back(vector2d{ { 1.f, 1.f } });

	float desired_output[4] = { 0.f, 1.f, 1.f, 0.f };

	int input_data_size = 1;
	int num_inputs = 2;
	int num_hidden = 2;
	int num_outputs = 1;

	// ==========================================
	matrix input_matrix(1, num_inputs );

	matrix w_m_1_2_(num_inputs, num_hidden);

	matrix theta_1_(1, num_hidden);

	matrix hidden_layer_(num_hidden, 1);

	matrix w_m_2_3_(num_hidden, num_outputs);

	matrix theta_2_(1, num_outputs);

	matrix out_(num_outputs, num_outputs);

	matrix del_3_2_(num_outputs, num_outputs);
	matrix del_2_1_(1, num_hidden);

	matrix del_theta_out(num_outputs, num_outputs);
	matrix del_theta_hidden(1, num_hidden);

	/*for (int i = 0; i < num_inputs+1; i++)
	{
	for (int j = 0; j < num_hidden + 1; j++)
	{
	w_m_1_2_(i, j) = RandomFloat(-1.2, 1.2);
	}
	}
	*/
	w_m_1_2_(0, 0) = 0.5f;
	w_m_1_2_(0, 1) = 0.9f;
	//w_m_1_2_(0, 2) = 0.0f;

	w_m_1_2_(1, 0) = 0.4f;
	w_m_1_2_(1, 1) = 1.0f;
//	w_m_1_2_(1, 2) = 0.0f;


	//w_m_1_2_(2, 2) = 1.0f;

	/*
	for (int i = 0; i < num_hidden + 1; i++)
	{
	w_m_2_3_(i, 0) = RandomFloat(-1.2, 1.2);
	}
	*/
	w_m_2_3_(0, 0) = -1.2f;
	w_m_2_3_(1, 0) = 1.1f;
	//w_m_2_3_(2, 0) = 0.3f; // theta for output 


	theta_1_(0, 0) = 0.8f;// theta 1
	theta_1_(0, 1) = -0.1f;//// theta 2

	theta_2_(0, 0) = 0.3f;

	float output_error = 0.0f;

	matrix w_m_delta_1_(2, 2);
	matrix w_m_delta_2_(2, 1);

	matrix w_m_delta_theta_hidden(1, 2);
	matrix w_m_delta_theta_output(1, 1);

	float alpha = 0.1f;
	float beta = 0.95f;

	float sum_squared_errors = 0.0f;


	timer.Start();

	//	Sleep(2000);

	float last_sum_squared_errors = 0.0f;
	int positive_error_delta_count = 0;
	int negative_error_delta_count = 0;
	int alternation_count = 0;

	for (int p = 0; p < num_iterations; p++)
	{
		sum_squared_errors = 0.0f;
		for (int q = 0; q < 4; q++)
		{
			input_matrix(0, 0) = training[q].v[0];
			input_matrix(0, 1) = training[q].v[1];
			//input_matrix(0, 2) = -1.0f; // bias is always -1

			float sum[3] = { 0.0f, 0.0f, 0.0f };



			hidden_layer_ = input_matrix * w_m_1_2_- theta_1_;


			sigmoid(hidden_layer_, hidden_layer_);

			// OVERWRITE 3rd INPUT
			//hidden_layer_(0, 2) = -1.0f;

			out_ = hidden_layer_ * w_m_2_3_ - theta_2_;

			sigmoid(out_, out_);

#ifdef VERBOSE
			if (p % 250 == 0)
			{
				hidden_layer_.print();
				cout << endl;
				out_.print();
				cout << endl;
			}
#endif
			output_error = desired_output[q] - out_(0, 0);

			sum_squared_errors += output_error * output_error;


			// back propogate

			anti_sigmoid(del_3_2_, out_);

			del_3_2_ = del_3_2_ * output_error;



			anti_sigmoid(del_2_1_, hidden_layer_);

			// put the vector on the diagonal for next operation ...
			/*matrix ident_22(3, 3);
			for (int i = 0; i < 3; i++)
			{
				for (int h = 0; h < 3; h++)
				{
					if (i == h) ident_22(i, h) = del_2_1_(0, i);
					else ident_22(i, h) = 0.0f;
				}
			}*/

			w_m_2_3_.transpose();

			del_2_1_ = del_2_1_ | w_m_2_3_ * del_3_2_(0, 0);

			w_m_2_3_.transpose();


			// weight deltas

			w_m_delta_2_.transpose();

			w_m_delta_2_ = w_m_delta_2_ * beta + del_3_2_* hidden_layer_ * alpha;

			w_m_delta_2_.transpose();

			w_m_delta_theta_output = w_m_delta_theta_output * beta + del_3_2_ * (-1.0f) * alpha;
#ifdef VERBOSE
			if (p % 250 == 0)
			{
				del_2_1_.print();
				cout << endl;
				del_3_2_.print();
				cout << endl;
			}
#endif
			w_m_delta_1_.transpose();

			del_2_1_.transpose();

			w_m_delta_1_ = w_m_delta_1_ * beta + del_2_1_*input_matrix   * alpha;

			w_m_delta_1_.transpose();

			del_2_1_.transpose();

			w_m_delta_theta_hidden = w_m_delta_theta_hidden * beta + del_2_1_ * (-1.0f) * alpha;

#ifdef VERBOSE
			if (p % 250 == 0)
			{
				w_m_delta_1_.print();
				cout << endl;
				w_m_delta_2_.print();
				cout << endl;
			}
#endif
			// update weights

			w_m_1_2_ = w_m_1_2_ + w_m_delta_1_;// 

			theta_1_ = theta_1_ + w_m_delta_theta_hidden;

			w_m_2_3_ = w_m_2_3_ + w_m_delta_2_;

			theta_2_ = theta_2_ + w_m_delta_theta_output;

#ifdef VERBOSE
			if (p % 250 == 0)
			{
				w_m_1_2_.print();
				cout << endl;
				w_m_2_3_.print();
				cout << endl;
			}
#endif
		}
		if (sum_squared_errors > last_sum_squared_errors*1.04) alpha *= 0.7;
		if (sum_squared_errors < last_sum_squared_errors) alpha *= 1.05;
		// calculate the change in sum_squared_errors
		float delta_sum_square_errors = sum_squared_errors - last_sum_squared_errors;
		last_sum_squared_errors = sum_squared_errors;
		if (delta_sum_square_errors > 0.0f)
		{
			if (positive_error_delta_count == 0) {
				alternation_count++;
			}
			else{
				alternation_count = 0;
			}
			positive_error_delta_count++;
			negative_error_delta_count = 0;
		}
		else
		{
			if (negative_error_delta_count == 0) {
				alternation_count++;
			}
			else{
				alternation_count = 0;
			}
			negative_error_delta_count++;
			positive_error_delta_count = 0;
		}

		// determine change in learning rate
		if (positive_error_delta_count >= 2 || negative_error_delta_count >= 2)
		{
			alpha += 0.1;
			if (alpha > 1.0f) alpha = 1.0f;
		}
		else if (alternation_count >= 2)
		{
			alpha -= 0.1;
			if (alpha < 0.0f) alpha = 0.01;
		}

		//cout << sum_squared_errors << endl;
		if (sum_squared_errors < 0.001)
		{
			timer.Update();
			timer.Stop();
			cout << "Finished on iteration: " << p << ", with sum squared errors less than 0.001" << endl << "Total calculation performed in " << timer.GetTimeDelta() << " seconds" << endl;
			break;
		}
	}

}

int main(int argc, char* argv[])
{
	cout << endl << endl << "Version 1 ";
	Compute_Simple_XOR_network_version_1();

	cout << endl << endl << "Version 2 ";
	Compute_Simple_XOR_network_version_2();
	
	cout << endl << endl << "Version 3 ";
	Compute_Simple_XOR_network_version_3(5000);

	cout << endl << endl << "Version 4 ";
	Compute_Simple_XOR_network_version_4(5000);

	cout << endl << endl << "Version 5 ";
	Compute_Simple_XOR_network_version_5(5000);

	cout << endl << endl << "Version 6 ";
	Compute_Simple_XOR_network_version_6(5000);
	cout << endl << endl;
	return 0;
}


#ifndef MY_MATRIX_H
#define MY_MATRIX_H

#include <iostream>

using std::cout;
using std::endl;

class vector2d
{
public:

	float v[2];

	float operator[](unsigned int idx) { return (idx < 2 ? v[idx] : 0.0f); }
	void operator=(vector2d b){ v[0] = b.v[0]; v[1] = b.v[1]; }
};

class matrix
{
public:
	matrix(){
		this->m_sizeX = 0; m_sizeY = 0;
		data = 0;
	}


	matrix(matrix *p){
		this->m_sizeX = p->m_sizeX; m_sizeY = p->m_sizeY;
		this->create();
		for (int i = 0; i < m_sizeX; i++)
		{
			for (int j = 0; j < m_sizeY; j++)
				data[i][j] = (*p)[i][j];
			//memcpy((void*)&data[i][j], (void*)&((*p)[i][j]), sizeof(float));
		}
		is_transposed = false;
	}

	matrix(matrix &p){
		this->m_sizeX = p.m_sizeX; m_sizeY = p.m_sizeY;
		this->create();
		for (int i = 0; i < m_sizeX; i++)
		{
			for (int j = 0; j < m_sizeY; j++)
				data[i][j] = p[i][j];
			//memcpy((void*)&data[i][j], (void*)&((p)[i][j]), sizeof(float));
		}
		is_transposed = false;
	}

	matrix(unsigned int n, unsigned int m)
	{
		m_sizeX = n;
		m_sizeY = m;
		create();
	}
	~matrix()
	{
		destroy();
		if (out) delete out;
	}

	void destroy()
	{
		if (data != 0)
		{
			for (int i = 0; i < m_sizeX; i++)
			{
				delete[] data[i];
			}
			delete[] data;
		}
		data = 0;
		is_transposed = false;
	}
	void create()
	{
		if (m_sizeY > 0 && m_sizeX > 0)
		{
			data = new float*[m_sizeX];
			for (int i = 0; i < m_sizeX; i++)
			{
				data[i] = new float[m_sizeY];
				for (int j = 0; j < m_sizeY; j++)
					data[i][j] = 0.0f;
			}
		}
		is_transposed = false;
	}

	void operator=(matrix &b)
	{
		this->destroy();
		this->m_sizeX = b.NumRows();
		this->m_sizeY = b.NumColumns();
		this->create();
		for (int i = 0; i < m_sizeX; i++)
		{
			for (int j = 0; j < m_sizeY; j++)
				data[i][j] = b(i, j);
			//memcpy((void*)&data[i][j], (void*)&b[i][j], sizeof(float));
		}


	}

	void operator=(matrix *b)
	{
		this->destroy();
		this->m_sizeX = b->NumRows();
		this->m_sizeY = b->NumColumns();
		this->create();
		for (int i = 0; i < this->NumRows(); i++)
		{
			for (int j = 0; j < this->NumColumns(); j++)
				data[i][j] = (*b)(i, j);
			//memcpy((void*)&data[i][j], (void*)&((*b)[i][j]), sizeof(float));
		}
	}





	float& operator()(unsigned int i, unsigned int j)
	{
		return get(i, j);
	}

	float& get(unsigned int i, unsigned int j)
	{
		float null_return = 0.0f;
		if (i < NumRows() && j < NumColumns())
			return is_transposed ? data[j][i] : data[i][j];
		else return null_return;
	}
	// Hadamard element wise product
	matrix operator | (matrix &b)
	{
		if (this->NumColumns() == b.NumColumns() && this->NumRows() == b.NumRows())
		{
			if (out) delete out;
			out = new matrix(this->NumRows(), this->NumColumns());

			for (int i = 0; i < this->NumRows(); i++)
			{

				for (int j = 0; j < this->NumColumns(); j++)
				{
					//for (int k = 0; k < this->NumColumns(); k++)
					{
						(*out)(i, j) = get(i, j) * b(i, j);
					}
				}
			}

			return *out;
		}
		return matrix(0, 0);
	}

	matrix operator*(matrix &b)
	{
		if (b.NumColumns() == 1 && b.NumRows() == 1)
		{
			return (*this) * b(0, 0);
		}
		if (this->NumColumns() == b.NumRows())
		{
			if (out) delete out;
			out = new matrix(this->NumRows(), b.NumColumns());

			for (int i = 0; i < this->NumRows(); i++)
			{

				for (int j = 0; j < b.NumColumns(); j++)
				{
					for (int k = 0; k < this->NumColumns(); k++)
					{
						(*out)(i, j) += get(i, k) * b(k, j);
					}
				}
			}

			return *out;
		}
		return matrix(0, 0);
	}

	matrix operator*(float s)
	{
		if (out) delete out;
		out = new matrix(this->NumRows(), this->NumColumns());

		for (int i = 0; i < this->NumRows(); i++)
		{
			for (int j = 0; j < this->NumColumns(); j++)
			{
				(*out)(i, j) = get(i, j) * s;
			}
		}
		return *out;
	}

	matrix operator/(float s)
	{
		if (out) delete out;
		out = new matrix(this->NumRows(), this->NumColumns());

		for (int i = 0; i < this->NumRows(); i++)
		{
			for (int j = 0; j < this->NumColumns(); j++)
			{
				(*out)(i, j) = get(i, j) / s;
			}
		}
		return *out;
	}
	matrix operator+(float s)
	{
		if (out) delete out;
		out = new matrix(this->NumRows(), this->NumColumns());

		for (int i = 0; i < this->NumRows(); i++)
		{
			for (int j = 0; j < this->NumColumns(); j++)
			{
				(*out)(i, j) = get(i, j) + s;
			}
		}
		return *out;
	}
	matrix operator+(matrix &b)
	{
		if (this->NumColumns() != b.NumColumns() || this->NumRows() != b.NumRows())
			return matrix(0, 0);
		else
		{
			if (out) delete out;
			out = new matrix(this->NumRows(), this->NumColumns());

			for (int i = 0; i < this->NumRows(); i++)
			{
				for (int j = 0; j < this->NumColumns(); j++)
				{
					(*out)(i, j) = this->get(i, j) + b.get(i, j);
				}
			}

			return *out;
		}
		return matrix(0, 0);
	}

	matrix operator-(matrix &b)
	{
		if (this->NumColumns() != b.NumColumns() || this->NumRows() != b.NumRows())
			return matrix(0, 0);
		else
		{
			if (out) delete out;
			out = new matrix(this->NumRows(), this->NumColumns());

			for (int i = 0; i < this->NumRows(); i++)
			{

				for (int j = 0; j < this->NumColumns(); j++)
				{
					(*out)(i, j) = this->get(i, j) - b.get(i, j);
				}
			}

			return *out;
		}
		return matrix(0, 0);
	}

	void print()
	{
		if (this->NumRows() == 0 || this->NumColumns() == 0 || this->data == 0)
		{
			cout << "Empty Matrix" << endl;
			return;
		}
		for (int i = 0; i < this->NumRows(); i++)
		{
			for (int j = 0; j < this->NumColumns(); j++)
			{
				cout << get(i, j) << "  ";
			}
			cout << endl;
		}
	}



	float trace()
	{
		float sum = 0.0f;
		if (m_sizeX != m_sizeY) return 0.0f;

		for (int i = 0; i < this->NumRows(); i++)
		{
			for (int j = 0; j < this->NumColumns(); j++)
			{
				if (i == j) sum += get(i, j);
			}

		}
		return sum;
	}

	bool IsSquare() 
	{
		if (m_sizeX == m_sizeY)
			return true;

		return false;
	}

	bool ContainsNAN()
	{
		for (int i = 0; i < m_sizeX; i++)
		{
			for (int j = 0; j < m_sizeY; j++)
			{
				if (data[i][j] != data[i][j])
					return true;
			}
		}
		return false;
	}

	void transpose()
	{
		is_transposed = !is_transposed;
	}



	inline unsigned int NumRows() 
	{ 
		return  (is_transposed ? m_sizeY : m_sizeX); 
	}
	inline unsigned int NumColumns() 
	{ 
		return (is_transposed ? m_sizeX : m_sizeY); 
	}


	bool is_transposed = false;
	unsigned int m_sizeX = 0;
	unsigned int m_sizeY = 0;
	float **data = 0;


private:
	float* operator[](unsigned int a)
	{
		if (a < m_sizeX)
			return data[a];
		else return 0;
	}


	matrix *out = 0;


};
#endif
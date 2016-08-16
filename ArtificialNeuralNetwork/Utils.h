#ifndef UTILS_H
#define UTIlS_H

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>


using std::cout;
using std::endl;
using std::vector;
using std::string;
using std::istringstream;
using std::copy;
using std::istream_iterator;
using std::back_inserter;
using std::ifstream;

inline float RandomFloat(float min, float max)
{
	float r = (float)rand() / (float)RAND_MAX;
	return min + r * (max - min);
}

inline float RandomInt(int min, int max)
{
	float r = (float)rand() / (float)RAND_MAX;
	return (int)((float)min + r * float(max - min));
}


inline int string_to_number(string s)
{
	return atoi(s.c_str());
}

inline float string_to_float(string s)
{
	return atof(s.c_str());
}

inline bool isNumberC(const std::string& s)
{
	char* p;
	strtod(s.c_str(), &p);
	return *p == 0;
}

inline double string_to_double(string s)
{
	char* p;
	double val = strtod(s.c_str(), &p);
	if (*p == 0) return val;

	return 0.0;
}

class CSV
{
public:

	void test()
	{
		vector<vector<string>> output;
		LoadCSV("iris.csv", output);



		for (int i = 0; i < output.size(); i++)
		{
			for (int j = 0; j < output[i].size(); j++)
			{
				cout << output[i][j] << ":";

			}
			cout << endl;
		}

		vector< vector< float >> iris_data;
		vector< string > found_tags;


		RestoreCSV_Iris_Numbers(output, iris_data, found_tags);

		for (int i = 0; i < iris_data.size(); i++)
		{
			for (int j = 0; j < iris_data[i].size(); j++)
			{
				cout << iris_data[i][j] << ":";

			}
			cout << endl;
		}

		cout << "printing found tags" << endl;
		for (int p = 0; p < found_tags.size(); p++)
			cout << found_tags[p] << endl;

	}
	void RestoreCSV_Iris_Numbers(vector< vector< string> > &v1, vector <vector< float> > &vout, vector<string> &found_tags)
	{

		for (int i = 0; i < v1.size(); i++)
		{
			vout.push_back(vector<float>());
			for (int j = 0; j < v1[i].size(); j++)
			{
				if (isNumberC(v1[i][j]))
				{
					vout[i].push_back((float)string_to_double(v1[i][j]));
				}
				else
				{
					bool tag_found = false;
					for (int p = 0; p < found_tags.size(); p++)
					{
						if (v1[i][j] == found_tags[p])
						{
							tag_found = true;
							vout[i].push_back((float)p);
						}
					}
					if (!tag_found)
					{
						found_tags.push_back(v1[i][j]);
						vout[i].push_back(found_tags.size() - 1);
					}
				}
			}
		}
	}

	void LoadCSV(string filename, vector<vector<string>> &output)
	{
		ifstream file_stream;
		file_stream.open(filename, std::ifstream::in);

		// split input by newline
		vector<string> tokenized_string;
		copy(istream_iterator<string>(file_stream),
			istream_iterator<string>(),
			back_inserter<vector<string> >(tokenized_string));

		// split lines by commas and store in output
		for (int j = 0; j < tokenized_string.size(); j++) {
			istringstream ss(tokenized_string[j]);
			vector<string> result;
			while (ss.good()) {
				string substr;
				getline(ss, substr, ',');
				result.push_back(substr);
			}
			output.push_back(result);
		}
		file_stream.close();
	}

};


#endif

#include <iostream>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <algorithm>

using namespace Eigen;

int myrandom (int i) { return std::rand()%i;}

ArrayXf itac (float y)
{
	ArrayXf x (10);
    for (float i = 0 ; i < y ; i++)
    {
        x(i) = 0;
    }
    x (y) =  1;
    for (float i = 9 ; i > y ; i--)
    {
        x (i) = 0;
    }
    return x;
}

std::vector<ArrayXf> data_loader (std::string fileName)
{
    std::ifstream file;
    file.open(fileName);
    std::string line;
    std::string word;
    std::vector <ArrayXf> data;
    while(getline(file , line))
    {
        ArrayXf D (785);
        std::stringstream ss (line);
        int counter = 0;
        while (getline(ss , word ,','))
        {
            
            float y = std::stof(word);
            D(counter) = y;
            counter ++;
        }
		D.tail(784) /= 255;
        data.push_back(D);
    }
    file.close();
	std::cout << "finished reading the file" << std::endl;
	return data;
}


void mnist_viewer (ArrayXf image)
{
	for (int i = 0; i < image.size(); i++)
	{
		if(image[i] > 0.5)
		{
			std::cout << "@ ";
		}
		else
		{
			if (image [i] > 0.01)
			{
				std::cout << "o ";
			}
			else
			{
				std::cout << "  ";
			}
		}
		if ((i+1) % 28 == 0)
		{
			std::cout << std::endl;
		}
	}
	
}

ArrayXXf colwise_expander (ArrayXf arr , int col)
{
	ArrayXXf outputArr (arr.size() , col);
	for (int i = 0 ; i < col ; i++)
	{
		outputArr.col(i) = arr;
	}
	return outputArr;
}

ArrayXXf rowwise_expander (ArrayXf arr , int row)
{
	ArrayXXf outputArr (row , arr.size());
	for (int i = 0 ; i < row ; i++)
	{
		outputArr.row(i) = arr;
	}
	return outputArr;
}

ArrayXf sigmoid (ArrayXf x)
{
	return ArrayXf::Ones(x.size()) / ((-1.0 * x).exp() + 1.0);
}

ArrayXf der_sigmoid (ArrayXf x)
{
	return x * ((-1.0 * x) + 1.0);
}

float conclusion (ArrayXf arr)
{
	ArrayXf::Index maxIndex;
	arr.maxCoeff(&maxIndex);
	return maxIndex;
}


class network
{
private:
	ArrayXf feedforward (ArrayXf inputs , int i)
	{
		ArrayXXf multiplie = l_weights[i] * colwise_expander(inputs , l_weights[i].cols());
		return sigmoid(multiplie.colwise().sum().transpose() + l_biases[i]);
	}
	std::vector <ArrayXXf> delta_w;
	std::vector <ArrayXf> delta_b;
	std::vector<ArrayXXf> l_weights;
	std::vector<ArrayXf> l_biases;
	std::vector<ArrayXf> results;
	ArrayXf input_saver;
	void first_back_prop (ArrayXf rin , ArrayXf inputs)
	{
		results.erase(results.begin(),results.end());
		al_feedforward(inputs);
		ArrayXf delta = der_sigmoid(results[results.size() - 1]) * (results[results.size() - 1] - rin);
		delta_b.push_back(delta);
		delta_w.push_back(colwise_expander(results[results.size()-2] , delta.size()) * rowwise_expander(delta , results[results.size()-2].size()));
		for (int layer = results.size()-2 ; layer > 0 ; layer--)
		{
			delta = ((rowwise_expander(delta , results[layer].size()) * l_weights[layer+1]).rowwise().sum()) * der_sigmoid(results[layer]);
			delta_b.push_back(delta);
			delta_w.push_back(colwise_expander(results[layer-1] , delta.size()) * rowwise_expander(delta , results[layer-1].size()) );
		}
		delta = ((rowwise_expander(delta , results[0].size()) * l_weights[1]).rowwise().sum()) * der_sigmoid(results[0]) ;
		delta_b.push_back(delta);
		delta_w.push_back(colwise_expander(input_saver , delta.size()) * rowwise_expander(delta , input_saver.size()));
	}
	void following_back_prop(ArrayXf rin , ArrayXf inputs)
	{
		results.erase(results.begin(),results.end());
		int lin = delta_b.size()-1;
		al_feedforward(inputs);
		ArrayXf delta = der_sigmoid(results[results.size() - 1]) * (results[results.size() - 1] - rin);
		delta_b[0] = delta_b[0] + delta ;
		delta_w[0] = delta_w[0] + (colwise_expander(results[results.size()-2] , delta.size()) * rowwise_expander(delta , results[results.size()-2].size()));
		for (int layer = results.size()-2 ; layer > 0 ; layer--)
		{
			delta = ((rowwise_expander(delta , results[layer].size()) * l_weights[layer+1]).rowwise().sum()) * der_sigmoid(results[layer]);
			delta_b[lin-layer] = delta_b[lin-layer] + delta;
			delta_w[lin-layer] = delta_w[lin-layer] + (colwise_expander(results[layer-1] , delta.size()) * rowwise_expander(delta , results[layer-1].size()));
		}
		delta = ((rowwise_expander(delta , results[0].size()) * l_weights[1]).rowwise().sum()) * der_sigmoid(results[0]);
		delta_b[lin] = delta_b[lin] + delta;
		delta_w[lin] = delta_w[lin] + (colwise_expander(input_saver , delta.size()) * rowwise_expander(delta , input_saver.size()));
	}
	void descenting (int batchSize)
	{
		float learning_rate = 3.0;
		for (int i = 0 ; i < l_weights.size() ; i++)
		{
			l_weights[i] = l_weights[i] -  (delta_w[l_weights.size()-1-i] * learning_rate / batchSize);
			l_biases[i] = l_biases[i] - (delta_b[l_biases.size()-1-i] * learning_rate /batchSize);
		}
		delta_w.erase(delta_w.begin(),delta_w.end());
		delta_b.erase(delta_b.begin(),delta_b.end());
	}
public:
	int n_i ;
	network(ArrayXi arch , int n_inputs)
	{
		n_i = n_inputs;
		for (int i = 0 ; i < arch.size() ; i++)
		{
			ArrayXXf weight = ArrayXXf::Random(n_inputs , arch(i));
			ArrayXf bias = ArrayXf::Zero(arch(i));
			l_weights.push_back(weight);
			l_biases.push_back(bias);
			n_inputs = arch(i);
		}
	}
	ArrayXf al_feedforward (ArrayXf inputs)
	{
		input_saver = inputs;
		for (int i = 0 ; i < l_weights.size() ; i++)
		{
			inputs = feedforward(inputs , i);
			results.push_back(inputs);
		}
		return inputs;
	}
	void train()
	{
		int batch_size = 10;
		int n_epoch = 3;
		std::vector<ArrayXf> file = data_loader("mnist_train.csv");
		for (int i = 1 ; i <= n_epoch ; i++)
		{
			std::srand ( unsigned ( std::time(0) ) );
			std::random_shuffle(file.begin() , file.end() , myrandom);
			first_back_prop(itac(file[0](0)) , file[0].tail(784));
			for (int k = 1 ; k < file.size(); k++)
			{
				if (k % batch_size == 0)
				{
					descenting(batch_size);
					first_back_prop(itac(file[k](0)) , file[k].tail(784));
				}
				else
				{
					following_back_prop(itac(file[k](0)) , file[k].tail(784));
				}
			}
			std::cout << "epoch over" << std::endl;
		}
	}
};

int main ()
{
	ArrayXi arch (3);
	arch << 30 , 30 , 10 ;
	network network1 = network(arch , 784);
	network1.train();
	std::vector<ArrayXf> file = data_loader("mnist_test.csv");
	int mistakeCounter = 0;
	for (int i = 0 ; i < file.size() ;i++)
	{
		if (conclusion(network1.al_feedforward(file[i].tail(784))) != file[i](0) )
		{
			mnist_viewer(file[i].tail(784));
			std::cout << "\n";
			std::cout << "machine says:" << conclusion(network1.al_feedforward(file[i].tail(784))) << " Correct is :" << file[i](0);
			std::cout << "\n";
			mistakeCounter ++;
		}
	}
	std::cout << "number of mistakes in 10000:" << mistakeCounter << std::endl;
	return 0;
}

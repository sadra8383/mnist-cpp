#include <iostream>
#include <include/eigen3/Eigen/Dense>
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

ArrayXXf myDot (ArrayXf colx , ArrayXf rowx)
{
	ArrayXXf output (colx.size() , rowx.size());
	for (int i = 0 ; i < rowx.size() ; i++)
	{
		output.col(i) = colx * rowx(i);
	}
	return output;
}

ArrayXf multiplieAndColSum (ArrayXXf x , ArrayXf y)
{
	ArrayXf output(x.cols()) ;
	for (int i = 0 ; i < x.cols() ; i++)
	{
		output(i) = (x.col(i) * y).sum();
	}
	return output;
}

ArrayXf multiplieAndRowSum (ArrayXXf x , ArrayXf y)
{
	ArrayXf output(x.rows()) ;
	x.transposeInPlace();
	for (int i = 0 ; i < x.cols() ; i++)
	{
		output(i) = (x.col(i) * y).sum();
	}
	return output;
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
	std::vector <ArrayXXf> delta_w;
	std::vector <ArrayXf> delta_b;
	std::vector<ArrayXXf> l_weights;
	std::vector<ArrayXf> l_biases;
	std::vector<ArrayXf> results;
	ArrayXf feedforward (ArrayXf inputs , int i)
	{
		return sigmoid(multiplieAndColSum(l_weights[i] , inputs) + l_biases[i]);
	}
	void backProp(ArrayXf rin , ArrayXf inputs)
	{
		results.erase(results.begin(),results.end());
		al_feedforward(inputs);
		int rs = results.size() - 1;
		ArrayXf delta = der_sigmoid(results.back()) * (results.back() - rin);
		delta_b.back() += delta ;
		delta_w.back() += myDot(results[rs-1] , delta); 
		for (int layer = rs-1 ; layer > 0 ; layer--)
		{
			delta = multiplieAndRowSum(l_weights[layer] , delta) * der_sigmoid(results[layer]);
			delta_b[layer-1] += delta;
			delta_w[layer-1] += myDot(results[layer-1] , delta);
		}
	}
	void descenting (int batchSize)
	{
		float learning_rate = 3;
		for (int i = 0 ; i < l_weights.size() ; i++)
		{
			l_weights[i] = l_weights[i] -  (delta_w[i] * learning_rate / batchSize);
			delta_w[i] *= 0;
			l_biases[i] = l_biases[i] - (delta_b[i] * learning_rate /batchSize);
			delta_b[i] *= 0;
		}
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
			delta_w.push_back(weight * 0);
			delta_b.push_back(bias);
			n_inputs = arch(i);
		}
	}
	ArrayXf al_feedforward (ArrayXf inputs)
	{
		results.push_back(inputs);
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
		int n_epoch = 5;
		std::vector<ArrayXf> file = data_loader("mnist_train.csv");
		for (int i = 1 ; i <= n_epoch ; i++)
		{
			std::srand ( unsigned ( std::time(0) ) );
			std::random_shuffle(file.begin() , file.end() , myrandom);
			for (int k = 0 ; k < file.size(); k++)
			{
				if ((k+1) % batch_size == 0)
				{
					descenting(batch_size);
					backProp(itac(file[k](0)) , file[k].tail(784));
				}
				else
				{
					backProp(itac(file[k](0)) , file[k].tail(784));
				}
			}
			std::cout << "epoch over" << std::endl;
		}
	}
};

int main ()
{
	ArrayXi arch (2);
	arch << 30 , 10 ;
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
			std::cout << "machine says:" << conclusion(network1.al_feedforward(file[i].tail(784)));
			std::cout << " Correct is :" << file[i](0);
			std::cout << "\n";
			mistakeCounter ++;
		}
	}
	std::cout << "number of mistakes in 10000:" << mistakeCounter << std::endl;
	return 0;
}
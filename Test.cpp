#include <iostream>
#include <string>
#include <vector>
#include "opencv2/core.hpp"
#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>

class NeuralNetwork{
    private:
    std::vector<Eigen::MatrixXf> Layers;
    public:
    NeuralNetwork(std::vector<int> Layer_sizes, std::string Filename="Weights.yml"){
        std::srand((unsigned int) time(0));
        try{
            cv::FileStorage fs(Filename, cv::FileStorage::READ);
            std::vector<cv::Mat> Last_weights;
            fs["Weights"] >> Last_weights;
            fs.release();
            if(Last_weights.size()==0)
                throw 0;
            std::cout<<"Se encontraron pesos guardados"<<std::endl;
            for (int i=0; i<Last_weights.size(); i++){
                std::cout<< Last_weights[i]<<std::endl;
                Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic> Temp_Matrix;
                //Eigen::Map<Eigen::MatrixXf> Temp_Matrix(Last_weights[i].ptr<double>(), Last_weights[i].rows, Last_weights[i].cols);
                cv::cv2eigen(Last_weights[i],Temp_Matrix);
                Layers.push_back(Temp_Matrix);
            }
        }
        catch(...){
            std::cout<<"No se encontraron pesos guardados"<<std::endl;
            for(int i=1; i<Layer_sizes.size(); i++){
                Eigen::MatrixXf Temp = Eigen::MatrixXf::Random(Layer_sizes[i-1], Layer_sizes[i]);//Temporary Matrix with the desired size is created
                Layers.push_back(Temp); //Save the temporary matrix in the vector of Layers
            }
        }
    }
    void WriteFile(std::string Filename="Weights.yml"){
        cv::FileStorage fs(Filename, cv::FileStorage::WRITE);
        std::vector<cv::Mat> LayersMat;
        for(int i=0; i<Layers.size(); i++){
            cv::Mat cvMat(Layers[i].rows(), Layers[i].cols(), CV_64FC1);
            cv::eigen2cv(Layers[i], cvMat);
            std::cout<<cvMat<<std::endl;
            LayersMat.push_back(cvMat);
        }
        fs << "Weights" << LayersMat;
        fs.release();
    }
    void PrintLayers(){
        for(int i=0; i<Layers.size(); i++){
            std::cout<<Layers[i]<<std::endl;
        }
    }
};

int main (int, char** argv){
    //std::srand((unsigned int) time(0));
    std::vector<int> Init_vec={2,3,5};
    NeuralNetwork nueva=NeuralNetwork(Init_vec);
    nueva.PrintLayers();
    nueva.WriteFile();

    return 0;

}

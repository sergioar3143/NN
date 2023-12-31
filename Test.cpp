#include <iostream>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>

class NeuralNetwork{
    private:
    std::vector<Eigen::MatrixXf> Weights;
    std::vector<Eigen::VectorXf> Biases;
    public:
    NeuralNetwork(std::vector<int> Layer_sizes, std::string Filename="Data.yml"){
        std::srand((unsigned int) time(0));
        try{
            cv::FileStorage fs(Filename, cv::FileStorage::READ);
            std::vector<cv::Mat> Last_weights;
            std::vector<cv::Mat> Last_biases;
            fs["Weights"] >> Last_weights;
            fs["Biases"]>> Last_biases;
            fs.release();
            if(Last_weights.size()==0 || Last_biases.size()==0)
                throw 0;
            std::cout<<"Se encontraron pesos guardados"<<std::endl;
            for (int i=0; i<Last_weights.size(); i++){
                //Creating temporary variables with the biases and weights size for each layer
                Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic> Temp_Matrix;
                Eigen::VectorXf Temp_Vec;
                //Changing the biases and weights from OpenCv objet to Eigen
                cv::cv2eigen(Last_biases[i],Temp_Vec);
                cv::cv2eigen(Last_weights[i],Temp_Matrix);
                // Save weights and biases
                Weights.push_back(Temp_Matrix);
                Biases.push_back(Temp_Vec);
            }
        }
        catch(...){
            std::cout<<"No se encontraron pesos guardados"<<std::endl;
            for(int i=1; i<Layer_sizes.size(); i++){
                Eigen::MatrixXf Temp1 = Eigen::MatrixXf::Random(Layer_sizes[i], Layer_sizes[i-1]);//Temporary Matrix with the desired size is created
                Weights.push_back(Temp1); //Save the temporary matrix in the vector of Layers

                Eigen::VectorXf Temp2 = Eigen::VectorXf::Random(Layer_sizes[i]);//Temporary vector with the desired size is created
                Biases.push_back(Temp2); //Save the temporary matrix in the vector of Layers
            }
        }
    }
    void WriteFile(std::string Filename="Data.yml"){
        cv::FileStorage fs(Filename, cv::FileStorage::WRITE);
        std::vector<cv::Mat> LayersMat;
        std::vector<cv::Mat> BiasesMat;
        for(int i=0; i<Weights.size(); i++){
            cv::Mat cvMat1(Weights[i].rows(), Weights[i].cols(), CV_64FC1);
            cv::Mat cvMat2(Biases[i].rows(), 1, CV_64FC1);

            cv::eigen2cv(Weights[i], cvMat1);
            cv::eigen2cv(Biases[i], cvMat2);
            //std::cout<<cvMat1<<std::endl;
            LayersMat.push_back(cvMat1);
            BiasesMat.push_back(cvMat2);
        }
        fs << "Weights" << LayersMat;
        fs << "Biases"  << BiasesMat;
        fs.release();
    }
    void PrintLayers(){
        for(int i=0; i<Weights.size(); i++){
            std::cout<<Weights[i]<<std::endl;
            std::cout<<Biases[i]<<std::endl;
        }
    }
    Eigen::VectorXf Feedforward(Eigen::VectorXf x ){
        for(int i=0; i<Weights.size(); i++){
            Eigen::VectorXf z=Weights[i]*x+Biases[i];
            x=1/(1+(-z).array().exp());
        }
        std::cout<<"Resultado"<<std::endl;
        std::cout<<x<<std::endl;
        return x;
    }
    std::vector<Eigen::VectorXf> Feedforward_verbose(Eigen::VectorXf x ){
        std::vector<Eigen::VectorXf> y;
        for(int i=0; i<Weights.size(); i++){
            Eigen::VectorXf z=Weights[i]*x+Biases[i];
            x=1/(1+(-z).array().exp());
            y.push_back(x);
        }
        return y;
    }
    

};

int main (int, char** argv){
    //std::srand((unsigned int) time(0));
    std::vector<int> Init_vec={5,3,2};
    NeuralNetwork nueva=NeuralNetwork(Init_vec);
    nueva.PrintLayers();
    nueva.WriteFile();
    Eigen::VectorXf x(5);
    x<<1.0,1.0,1.0,1.0,1.0;
    std::cout<<x<<std::endl;
    nueva.Feedforward_verbose(x);
    return 0;

}

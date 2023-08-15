#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

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
        y.push_back(x);
        for(int i=0; i<Weights.size(); i++){
            Eigen::VectorXf z=Weights[i]*x+Biases[i];
            x=1/(1+(-z).array().exp());
            y.push_back(x);
        }
        return y;
    }
    void Backpropagate(Eigen::VectorXf x, Eigen::VectorXf target, std::vector<Eigen::MatrixXf> &nabla_w, std::vector<Eigen::VectorXf> &nabla_b){
        std::vector<Eigen::VectorXf> y= Feedforward_verbose(x);
        int n_layer=Biases.size();
        nabla_w=Weights;
        nabla_b=Biases;
        Eigen::VectorXf delta = (y[n_layer].cwiseProduct(y[n_layer] - target)).cwiseProduct((1 - y[n_layer].array()).matrix());
        nabla_b[n_layer-1]=delta;
        nabla_w[n_layer-1]=delta*y[n_layer-2].transpose();
        for(int i=2; i<=Weights.size(); i++){
            //delta=Weights[n_layer-i+1].transpose()*delta;
            delta=(( Weights[n_layer-i+1].transpose()*delta ).cwiseProduct( y[n_layer-i+1] )).cwiseProduct( (1 - y[n_layer-i+1].array()).matrix() );
            nabla_b[n_layer-i]=delta;
            nabla_w[n_layer-i]=delta*y[n_layer-i].transpose();
        }
        std::cout<<"delta ="<<std::endl;
        std::cout<<delta<<std::endl;
    }

};
//////////////////
//End of the class
//////////////////

//function that take one string path in which the image is located and then return one vector with the image data
Eigen::VectorXf Get_Vec(std::string Image_path, int n_pixels){
    cv::Mat img_original = cv::imread(Image_path);
    cv::Mat img = img_original.clone();
    cv::Mat resized_Img;//Declare resized_Img
    img.convertTo(img, CV_32F);//Change image values to float
    img/=255.0; //Normalize the values
    cv::resize(img, resized_Img, cv::Size(n_pixels, n_pixels), cv::INTER_CUBIC);//resize the image to nxn using the cubic interpolation
    cv::Mat vector=resized_Img.reshape(1,resized_Img.total());//change the RGB matrix to 3 vectors RGB
    vector=resized_Img.reshape(1,vector.total());//change all the values to one vector with all the information
    Eigen::Map<Eigen::VectorXf> vecX(vector.ptr<float>(), vector.rows);//Change Mat vector to Eigen vector
    return vecX;
}

void Get_Data_Matrix(std::string path, std::vector<Eigen::VectorXf> & data, std::vector<Eigen::VectorXf> & target, int n_pixels, int input, int n_img, int n_class){
    //path is the path in which the image are saved
    //input is de size of the vectors that represent each image
    //n_img is the number of image per class that are wanted in the dataMatrix
    //n_class is the number of different classes thta are wanted to clasificate
    Eigen::VectorXf TempVec;
    //std::vector<Eigen::VectorXf> data;
    int class_n=0;//auxiliar for filling data Matrix
    int count_img=0;//the variable counts the number of image in the class, it helps just to train with 10 images
    int class_counter=0;
    for (const auto & entry : fs::directory_iterator(path)){
        std::string path2=entry.path(); //Save the multiple directories in path2
        class_n=0;//begin with zero for each class and count the image number for the class
        for (const auto & entry2 : fs::directory_iterator(path2)){ //then in path2 check the images in each directory
            std::string Image_path=entry2.path(); //get the path for each image
            if(class_n< n_img){ //Save just the first n_img images for each class
                TempVec=Get_Vec(Image_path, n_pixels); //get the image data in one row vector

                Eigen::VectorXf TempTarget(10);
                TempTarget<<0,0,0,0,0,   0,0,0,0,0;
                TempTarget[class_counter]=1;
                target.push_back(TempTarget);
                data.push_back(TempVec); //saving data
                count_img++;//increase the pointer
            }
            class_n++; //increase class number
        }
        class_counter++;
    }
    //return data;
}


int main (int, char** argv){
    //std::srand((unsigned int) time(0));
    std::vector<int> Init_vec={5,3,2};
    NeuralNetwork nueva=NeuralNetwork(Init_vec);
    nueva.PrintLayers();
    nueva.WriteFile();
    Eigen::VectorXf x(5);
    x<<1.0,1.0,1.0,1.0,1.0;
    Eigen::VectorXf target(2);
    target<<0.0, 0.99;
    std::vector<Eigen::MatrixXf> nabla_w;
    std::vector<Eigen::VectorXf> nabla_b;
    nueva.Feedforward(x);
    nueva.Backpropagate(x, target, nabla_w, nabla_b);

    std::vector<Eigen::VectorXf> data;//(150, input_size);//declare the matrix with all the data from the trainning images
    std::vector<Eigen::VectorXf> t;
    int input_size=60*60*3;
    int n_pixels=60; // the images going to have size 100x100
    int n_img=15; //the number of images per class that are wanted in the matrix data, it could change after trainning
    int n_class=10; //the number of classes that are wanted for classificaction, it couldn't be changed

    std::string path = "./Objetos_segmentados";//the path for the trainning images of each class
    Get_Data_Matrix(path,data, t, n_pixels,input_size,n_img, n_class);//all the data is in this Matrix, each row represent an image

    std::cout<<data[0].rows()<<std::endl;
    return 0;

}

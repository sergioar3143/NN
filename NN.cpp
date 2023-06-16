#include "opencv2/opencv.hpp"
#include <string>
#include <iostream>
#include <experimental/filesystem>
#include <eigen3/Eigen/Dense>
//#include <Eigen/Dense>
//namespace fs = std::filesystem;
//namespace fs = std::experimental::filesystem;
//using namespace std;
namespace fs = std::experimental::filesystem;

//function join example:
//RowVectorXd joined(7);
//joined << vec1, vec2; joined is the concadenate result of vec1 and vec2

//function that take one image and return one vector
Eigen::RowVectorXf Get_Vec(std::string Image_path, int n_pixels){
    cv::Mat img_original = cv::imread(Image_path);
    cv::Mat img = img_original.clone();
    cv::Mat resized_Img;//Declare resized_Img
    img.convertTo(img, CV_32F);//Change image values to float
    img/=255.0; //Normalize the values
    cv::resize(img, resized_Img, cv::Size(n_pixels, n_pixels), cv::INTER_CUBIC);//resize the image to nxn using the cubic interpolation
    cv::Mat vector=resized_Img.reshape(1,resized_Img.total());//change the RGB matrix to 3 vectors RGB
    vector=resized_Img.reshape(1,vector.total());//change all the values to one vector with all the information
    //std::cout<< "vector dimension: "<< vector.rows<<" x "<<vector.cols<<std::endl;
    Eigen::Map<Eigen::RowVectorXf> vecX(vector.ptr<float>(), vector.rows);//Change Mat vector to Eigen vector
    //std::cout<< vecX(1000)<< " = " << vector.at<float>(1000,0)<<std::endl;
    return vecX;
}

Eigen::MatrixXf Get_Data_Matrix(std::string path, int n_pixels, int input){
    Eigen::RowVectorXf vecX;
    Eigen::RowVectorXf joined(input+1);
    Eigen::MatrixXf data(150, input+1);
    Eigen::RowVectorXf v_one(1); // this vector represent the signal input for the bias
    v_one << 1; //the value is 1 and going to be concatenated with the input for each layer
    int row_n=0;//auxiliar for filling data Matrix
    int count_img=0;//the variable counts the number of image in the class, it helps to just train with 10 images
    for (const auto & entry : fs::directory_iterator(path)){
        std::string path2=entry.path(); //Save the multiple directories in path2
        //std::cout<< path2 << std::endl;
        row_n=0;
        for (const auto & entry2 : fs::directory_iterator(path2)){ //then in path2 check the images in each directory
            std::string Image_path=entry2.path(); //get the path for each image
            if(row_n<=14){
                vecX=Get_Vec(Image_path, n_pixels);
                joined << v_one, vecX;
                data.row(count_img)=joined;
                count_img++;
            }
            row_n++;
        }
    }
    return data;
}

int main()
{
    cv::Mat resized_Img;
    int input_size=30000;
    int n_pixels=100; // the images going to have size 100x100
    int L1_out=1; //the output size for the first layer
    int L2_out=10; //the output size for the second layer
    std::srand((unsigned int) time(0)); //set the seed for random numbers

    Eigen::MatrixXf data(150, 30001);//declare the matrix with all the data from the trainning images
    Eigen::RowVectorXf aux;

    std::string path = "./Objetos_segmentados";//the path for the trainning images of each class
    data=Get_Data_Matrix(path, n_pixels,input_size);//all the data is in this Matrix, each row represent an image

    Eigen::MatrixXf Layer1 = Eigen::MatrixXf::Random(input_size+1, L1_out);

    for (int i = 0; i <= 149; i++) {
         //aux=data.block(i,1,1,30000);
         aux=data.row(i);
    }
    Eigen::RowVectorXf y1=data.row(3)*Layer1;
    Eigen::RowVectorXf y2=1/(1+y1.array().exp());
    std::cout<< y1 << std::endl;
    std::cout<< y2<<std::endl;
    //print_path("hola");
    return 0;
//cv::resize(inImg, outImg, cv::Size(), 0.75, 0.75);

}

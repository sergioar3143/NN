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

//function that take one string path in which the image is located and then return one vector with the image data
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
    Eigen::MatrixXf data(150, input);
    int class_n=0;//auxiliar for filling data Matrix
    int count_img=0;//the variable counts the number of image in the class, it helps to just train with 10 images
    for (const auto & entry : fs::directory_iterator(path)){
        std::string path2=entry.path(); //Save the multiple directories in path2
        class_n=0;//begin with zero for each class and count the image number for the class
        for (const auto & entry2 : fs::directory_iterator(path2)){ //then in path2 check the images in each directory
            std::string Image_path=entry2.path(); //get the path for each image
            if(class_n<=14){ //Save just the first 15 images for each class
                vecX=Get_Vec(Image_path, n_pixels); //get the image data in one row vector
                data.row(count_img)=vecX; //save the data
                count_img++;//increase the pointer
            }
            class_n++; //increase class number
        }
    }
    return data;
}

Eigen::RowVectorXf LayerOutput(Eigen::MatrixXf& layer, Eigen::RowVectorXf &input_layer){//This function generate the output from current layer
    //input layer is the signal input or the output from the previous layer
    //layer is the matrix with the actual biases and weights from the current layer
    Eigen::RowVectorXf Aux(1+input_layer.size()); //aux=[1 input layer] aux is the concatenated vector with 1 (the input for the bias)
    Aux << 1, input_layer; //adding input for bias
    Eigen::MatrixXf layerT=layer.transpose();//tranpose the matrix
    Eigen::RowVectorXf output=Aux*layerT;
    Eigen::RowVectorXf sigmoid_out=1/(1+output.array().exp());
    return sigmoid_out;
}

Eigen::RowVectorXf Get_delta(Eigen::RowVectorXf& v, Eigen::RowVectorXf& y){
    Eigen::RowVectorXf dif=1-y.array();
    Eigen::RowVectorXf y_new = y.cwiseProduct(dif);
    Eigen::RowVectorXf final_v= v.cwiseProduct(y_new);
    return final_v;
}

Eigen::MatrixXf Get_image_gradient(Eigen::RowVectorXf& delta, Eigen::RowVectorXf& input_layer){
    Eigen::VectorXf deltaT=delta.transpose();
    Eigen::MatrixXf E;
    Eigen::RowVectorXf bias_input(1+input_layer.size());
    bias_input<<1,input_layer;
    E=deltaT*bias_input;
    return E;
}

void Trainning(Eigen::MatrixXf& data, Eigen::MatrixXf& Layer1, Eigen::MatrixXf& Layer2, int epoch, float n, float mu, float epsilon){
    Eigen::MatrixXf L1_ant,nablaL1,Er1,Er2;
    Eigen::MatrixXf L2_ant,nablaL2;
    Eigen::MatrixXf L1_next=Layer1;
    Eigen::MatrixXf L2_next=Layer2;
    Eigen::RowVectorXf t(10),y1,y2,delta,aux,temp1;
    Eigen::VectorXf deltaT;
    int target,l1_in,l2_in,l1_out,l2_out;
    //l1_in=Layer1.rows();
    //Eigen::MatrixXf Layer2 = Eigen::MatrixXf::Random(L2_out, L1_out+1);
    for(int i=0; i< epoch; i++){
        L1_ant=Layer1;
        L2_ant=Layer2;
        Layer1=L1_next;
        Layer2=L2_next;
        nablaL1=Eigen::MatrixXf::Zero(Layer1.rows(),Layer1.cols());
        nablaL2=Eigen::MatrixXf::Zero(Layer2.rows(),Layer2.cols());
        for(int j=0; j<=149; j++){
            target=j/15;//it gives array position for the target
            t<<0,0,0,0,0 ,0,0,0,0,0;
            t(target)=1;//get the target

            aux=data.row(j); //get data from an image
            y1=LayerOutput(Layer1, aux); //get the output for the first layer
            y2=LayerOutput(Layer2, y1); //get the output for the second layer

            temp1=2*(t-y2);
            delta=Get_delta(temp1, y2);

            //deltaT=delta.transpose();
            Eigen::RowVectorXf temp(1+y1.size());
            temp<<1,y1;
            //Er2=deltaT*temp;
            Er2=Get_image_gradient(delta,y1);

            temp1=delta*Layer2;
            delta=Get_delta(temp1, temp);

            deltaT=delta.transpose();
            Eigen::RowVectorXf temp2(1+aux.size());
            temp2<<1,aux;
            Er1=deltaT*temp2;
            //std::cout<<"Error:"<<Er1.rows()<<std::endl;

            //Er1=Er1.bottomRows(Er1.rows()-1);
            Eigen::MatrixXf Er12=Er1.bottomRows(Er1.rows()-1);
            //std::cout<<"Error:"<<std::endl;

            nablaL1=nablaL1+Er12/150;
            nablaL2=nablaL2+Er2/150;

            //std::cout<<"Dimensiones E1:"<<Er1.rows()<<"X"<<Er1.cols()<<std::endl;
            //std::cout<<"Dimensiones L1:"<<Layer1.rows()<<"X"<<Layer1.cols()<<std::endl;

        }
        L2_next=Layer2+n*nablaL2;
        L1_next=Layer1+n*nablaL1;
        std::cout<<"Epoch:"<<i<<std::endl;
    }
}


int main()
{
    cv::Mat resized_Img;
    int input_size=480000;
    int n_pixels=400; // the images going to have size 100x100
    int L1_out=50; //the output size for the first layer
    int L2_out=10; //the output size for the second layer
    std::srand((unsigned int) time(0)); //set the seed for random numbers

    Eigen::MatrixXf data(150, input_size);//declare the matrix with all the data from the trainning images
    Eigen::RowVectorXf y1,y2,aux(input_size);

    std::string path = "./Objetos_segmentados";//the path for the trainning images of each class
    data=Get_Data_Matrix(path, n_pixels,input_size);//all the data is in this Matrix, each row represent an image

    Eigen::MatrixXf Layer1 = Eigen::MatrixXf::Random(L1_out, input_size+1);
    Eigen::MatrixXf Layer2 = Eigen::MatrixXf::Random(L2_out, L1_out+1);

    for (int i = 0; i <= 149; i++) {
         //aux=data.block(i,1,1,30000);
         aux=data.row(i);
    }
    y1=LayerOutput(Layer1, aux);
    std::cout<<"Dimensiones y1:"<<y1.rows()<<"X"<<y1.cols()<<std::endl;
    y2=LayerOutput(Layer2,y1);
    std::cout<<"Dimensiones y2:"<<y2.rows()<<"X"<<y2.cols()<<std::endl;
    std::cout<< y2 << std::endl;
    Trainning(data, Layer1,Layer2, 10, 0.2, 0, 0);
    y1=LayerOutput(Layer1, aux);
    y2=LayerOutput(Layer2,y1);
    std::cout<< y2 << std::endl;
    return 0;
//cv::resize(inImg, outImg, cv::Size(), 0.75, 0.75);
}

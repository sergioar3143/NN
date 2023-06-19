#include "opencv2/opencv.hpp"
#include <string>
#include <iostream>
#include <experimental/filesystem>
#include <eigen3/Eigen/Dense>
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
    Eigen::Map<Eigen::RowVectorXf> vecX(vector.ptr<float>(), vector.rows);//Change Mat vector to Eigen vector
    return vecX;
}

Eigen::MatrixXf Get_Data_Matrix(std::string path, int n_pixels, int input, int n_img, int n_class){
    //path is the path in which the image are saved
    //input is de size of the vectors that represent each image
    //n_img is the number of image per class that are wanted in the dataMatrix
    //n_class is the number of different classes thta are wanted to clasificate
    Eigen::RowVectorXf vecX;
    Eigen::MatrixXf data(n_img*n_class, input);
    int class_n=0;//auxiliar for filling data Matrix
    int count_img=0;//the variable counts the number of image in the class, it helps to just train with 10 images
    for (const auto & entry : fs::directory_iterator(path)){
        std::string path2=entry.path(); //Save the multiple directories in path2
        class_n=0;//begin with zero for each class and count the image number for the class
        for (const auto & entry2 : fs::directory_iterator(path2)){ //then in path2 check the images in each directory
            std::string Image_path=entry2.path(); //get the path for each image
            if(class_n< n_img){ //Save just the first n_img images for each class
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
    //Eigen::RowVectorXf Aux(1+input_layer.size()); //aux=[1 input layer] aux is the concatenated vector with 1 (the input for the bias)
    //Aux << 1, input_layer; //adding input for bias
    //Eigen::MatrixXf layerT=layer.transpose();//tranpose the matrix
    Eigen::RowVectorXf output=-(Eigen::RowVectorXf(1+input_layer.size())<<1,input_layer).finished()*layer.transpose();
    Eigen::RowVectorXf sigmoid_out=1/(1+output.array().exp());
    return sigmoid_out;
}

Eigen::RowVectorXf Get_delta(const Eigen::RowVectorXf& v, Eigen::RowVectorXf& y){//get (1-y).y.v
    Eigen::RowVectorXf dif=1-y.array();
    Eigen::RowVectorXf final_v = v.cwiseProduct(dif.cwiseProduct(y));
    return final_v;
}


void Trainning(Eigen::MatrixXf& data, Eigen::MatrixXf& Layer1, Eigen::MatrixXf& Layer2, int epoch, float n, float mu, float epsilon){
    Eigen::MatrixXf L1_ant,nablaL1,Er1,Er2;
    Eigen::MatrixXf L2_ant,nablaL2;
    Eigen::MatrixXf L1_next=Layer1;
    Eigen::MatrixXf L2_next=Layer2;
    Eigen::RowVectorXf t(10),y1,y2,delta,aux,t2;
    t<<0,0,0,0,0,0,0,0,0,0;
    //Eigen::VectorXf deltaT;
    int target,l1_in,l2_in,l1_out,l2_out;
    //l1_in=Layer1.rows();
    //Eigen::MatrixXf Layer2 = Eigen::MatrixXf::Random(L2_out, L1_out+1);
    for(int i=0; i< epoch; i++){
        L1_ant=Layer1; //variable for saving previous layer1 weights
        L2_ant=Layer2; //variable for saving previous layer2 weights
        Layer1=L1_next;
        Layer2=L2_next;
        nablaL1=Eigen::MatrixXf::Zero(Layer1.rows(),Layer1.cols());
        nablaL2=Eigen::MatrixXf::Zero(Layer2.rows(),Layer2.cols());
        for(int j=0; j<=149; j++){
            target=j/15;//it gives array position of the target
            t2=t;//inicialize with zeros
            t2(target)=1;//get the target

            aux=data.row(j); //get data from an image
            y1=LayerOutput(Layer1, aux); //get the output for the first layer
            y2=LayerOutput(Layer2, y1); //get the output for the second layer

            delta=Get_delta(2*(t2-y2), y2);//Get delta for Layer 2
            Er2= delta.transpose()*(Eigen::RowVectorXf(1+y1.size())<<1,y1).finished();//get gradient for layer 2

            delta=Get_delta(delta*Layer2, (Eigen::RowVectorXf(1+y1.size())<<1,y1).finished() ); //get delta for Layer1
            Er1=(delta.transpose()*(Eigen::RowVectorXf(1+aux.size())<<1,aux).finished()).bottomRows(nablaL1.rows()); //get gradiente for layer 1

            nablaL1=nablaL1+Er1; //add the gradient of layer1 for each image
            nablaL2=nablaL2+Er2; //add the gradient of layer2 for each image

        }
        L2_next=Layer2+n*nablaL2/150+mu*(Layer2-L2_ant)+epsilon*Eigen::MatrixXf::Random(Layer2.rows(), Layer2.cols());//update weights in layer1
        L1_next=Layer1+n*nablaL1/150+mu*(Layer1-L1_ant)+epsilon*Eigen::MatrixXf::Random(Layer1.rows(), Layer1.cols());//update weights in layer2
        std::cout<<"Epoch:"<<i<<std::endl;
    }
}


int main()
{
    cv::Mat resized_Img;//variable for resized images
    int input_size=60*60*3;
    int n_pixels=60; // the images going to have size 100x100
    int L1_out=30; //the output size for the first layer
    int L2_out=10; //the output size for the second layer
    int n_img=15; //the number of images per class that are wanted in the matrix data, it could change after trainning
    int n_class=10; //the number of classes that are wanted for classificaction, it couldn't be changed
    std::srand((unsigned int) time(0)); //set the seed for random numbers

    Eigen::MatrixXf data;//(150, input_size);//declare the matrix with all the data from the trainning images
    Eigen::RowVectorXf y1,y2,aux(input_size);

    std::string path = "./Objetos_segmentados";//the path for the trainning images of each class
    data=Get_Data_Matrix(path, n_pixels,input_size,n_img, n_class);//all the data is in this Matrix, each row represent an image

    Eigen::MatrixXf Layer1 = Eigen::MatrixXf::Random(L1_out, input_size+1);
    Eigen::MatrixXf Layer2 = Eigen::MatrixXf::Random(L2_out, L1_out+1);

    int epoch=20;//epoch number for the trainning
    float n=0.2;//learning rate
    float mu=0.4;
    float epsilon=0.001;//Max value for the noise
    std::cout<<"Start trainning"<<std::endl;
    Trainning(data, Layer1,Layer2, epoch, n, mu, epsilon);//Trainning layer1 and layer2 for epoch with data and parameters n, mu and epsilon

    n_img=20;//change the number of images to 20
    data=Get_Data_Matrix(path, n_pixels,input_size,n_img, n_class);//all the data is in this Matrix, each row represent an image
    Eigen::Index   maxIndex;//variable for saving the index
    float max ; //=mat.colwise().sum().maxCoeff(&maxIndex);
    //for (int i = 0; i < data.rows(); i++) {
      //   aux=data.row(i);
      //   y1=LayerOutput(Layer1, aux);//get the first layer output
      //   y2=LayerOutput(Layer2, y1);//get the second layer output
      //   max=y2.maxCoeff(&maxIndex);//check for the class with the maximum value
      //   std::cout<<"The class from "<<i<< "image is from class "<< maxIndex << " with the output "<< y2 <<std::endl;//print information
    //}
    int i=0;
    int target=0;
    std::string result;
    for (const auto & entry : fs::directory_iterator(path)){
        std::string path2=entry.path(); //Save the multiple directories in path2
        for (const auto & entry2 : fs::directory_iterator(path2)){ //then in path2 check the images in each directory
            std::string Image_path=entry2.path(); //get the path for each image
            cv::Mat img_original = cv::imread(Image_path);
            cv::Mat img = img_original.clone();
            cv::Mat resized_Img;//Declare resized_Img
            cv::resize(img, resized_Img, cv::Size(3*n_pixels, 3*n_pixels), cv::INTER_CUBIC);//resize the image to nxn using the cubic interpolation
            cv::imshow("Image",resized_Img);
            aux=data.row(i);
            y1=LayerOutput(Layer1, aux);//get the first layer output
            y2=LayerOutput(Layer2, y1);//get the second layer output
            max=y2.maxCoeff(&maxIndex);//check for the class with the maximum value
            switch(maxIndex) {//the
                case 0: //Bloque de instrucciones 1;
                    result="Chocolate cookies";
                break;
                case 1: //Bloque de instrucciones 2;
                    result="Red Mug";
                break;
                case 2: //Bloque de instrucciones 3;
                    result="Apple Juice";
                break;
                case 3: //Bloque de instrucciones 1;
                    result="Blue Spoon";
                break;
                case 4: //Bloque de instrucciones 2;
                    result="Blue Bowl";
                break;
                case 5: //Bloque de instrucciones 3;
                    result="Orange Juice";
                break;
                case 6: //Bloque de instrucciones 1;
                    result="Red Lego";
                break;
                case 7: //Bloque de instrucciones 2;
                    result="Orange Knife";
                break;
                case 8: //Bloque de instrucciones 3;
                    result="Blue Mug";
                break;
                case 9: //Bloque de instrucciones 1;
                    result="Blue Lego";
                break;
                default: //Bloque de instrucciones por defecto;
                    result="Error";
            }
            std::cout<<"The class from image "<<i<< " is "<< result <<std::endl;//print information
            std::cout<<"The output from the Neural Network is "<<y2<<std::endl;
            i++; //increase class number
            cv::waitKey(0);
            std::cout << "Press Enter to continue..." << std::endl;
        }
        target++;
    }
    return 0;
}

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

int main()
{
    cv::Mat resized_Img;
    int n_pixels=100;
      Eigen::Matrix2d a;
      a << 1, 2,
           3, 4;
     std::cout<< a << std::endl;
    //cv::imshow("Lines", img);
    std::string path = "./Objetos_segmentados";
    for (const auto & entry : fs::directory_iterator(path)){
        std::string path2=entry.path(); //Save the multiple directories in path2
        std::cout<< path2 << std::endl;
        for (const auto & entry2 : fs::directory_iterator(path2)){ //then in path2 check the images in each directory
            std::string Image_path=entry2.path(); //get the path for each image
            //std::cout << Image_path <<std::endl;
            cv::Mat img_original = cv::imread(Image_path);
            cv::Mat img = img_original.clone();
            cv::resize(img, resized_Img, cv::Size(n_pixels, n_pixels), cv::INTER_CUBIC);
            //cv::resize(img, resized_Img , cv::Size(), 0.5, 1.5, cv::INTER_CUBIC);
            cv::imshow("Resized", resized_Img);
            cv::waitKey(100);
        }
    }


    return 0;
//cv::resize(inImg, outImg, cv::Size(), 0.75, 0.75);

}

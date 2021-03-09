///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2020, STEREOLABS.
//
// All rights reserved.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////

/***********************************************************************************************
 ** This sample demonstrates how to use the ZED SDK with OpenCV. 					  	      **
 ** Depth and images are captured with the ZED SDK, converted to OpenCV format and displayed. **
 ***********************************************************************************************/

 // ZED includes
#include <sl/Camera.hpp>

// OpenCV includes
#include <opencv2/opencv.hpp>
// OpenCV dep
#include <opencv2/cvconfig.h>

// Sample includes
#include <SaveDepth.hpp>

#include <algorithm>
#include "CornerDetAC.h"
#include "ChessboradStruct.h"

#include <fstream>
#include <stdio.h>
#include <iostream>
#include <time.h>
#include <math.h>
#include "MatrixReaderWriter.h"

using namespace sl;
typedef struct Trafo {
    cv::Mat rot;
    float scale;
    cv::Point3f offset1, offset2;
} Trafo;

std::vector<cv::Point2i> points;
cv::Mat slMat2cvMat(Mat& input);
#ifdef HAVE_CUDA
cv::cuda::GpuMat slMat2cvMatGPU(Mat& input);
#endif // HAVE_CUDA

Trafo registration(vector<cv::Point3f>& pts1, std::vector<cv::Point3f>& pts2);
void printHelp();
Corners CornerDetection(cv::Mat image_ocv);
void pointcloud_registration(std::string reference_pcl, std::string detected_pcl);

int main(int argc, char **argv) {

    // Create a ZED camera object
    Camera zed;

    // Set configuration parameters
    InitParameters init_params;
    init_params.camera_resolution = RESOLUTION::HD1080;
    init_params.depth_mode = DEPTH_MODE::ULTRA;
    init_params.coordinate_units = UNIT::METER;
    if (argc > 1) init_params.input.setFromSVOFile(argv[1]);
        
    // Open the camera
    ERROR_CODE err = zed.open(init_params);
    if (err != ERROR_CODE::SUCCESS) {
        printf("%s\n", toString(err).c_str());
        zed.close();
        return 1; // Quit if an error occurred
    }

    // Display help in console
    printHelp();

    // Set runtime parameters after opening the camera
    RuntimeParameters runtime_parameters;
    runtime_parameters.sensing_mode = SENSING_MODE::STANDARD;

    // Prepare new image size to retrieve half-resolution images
    Resolution image_size = zed.getCameraInformation().camera_resolution;
    int new_width = image_size.width / 2;
    int new_height = image_size.height / 2;

    Resolution new_image_size(new_width, new_height);

    // To share data between sl::Mat and cv::Mat, use slMat2cvMat()
    // Only the headers and pointer to the sl::Mat are copied, not the data itself
    Mat image_zed(new_width, new_height, MAT_TYPE::U8_C4);
    cv::Mat image_ocv = slMat2cvMat(image_zed);

#ifndef HAVE_CUDA // If no cuda, use CPU memory
    Mat depth_image_zed(new_width, new_height, MAT_TYPE::U8_C4);
    cv::Mat depth_image_ocv = slMat2cvMat(depth_image_zed);
#else
    Mat depth_image_zed_gpu(new_width, new_height, MAT_TYPE::U8_C4, sl::MEM::GPU); // alloc sl::Mat to store GPU depth image
    cv::cuda::GpuMat depth_image_ocv_gpu = slMat2cvMatGPU(depth_image_zed_gpu); // create an opencv GPU reference of the sl::Mat
    cv::Mat depth_image_ocv; // cpu opencv mat for display purposes
#endif
    Mat point_cloud;
    cv::Mat std_dev_depth;
    // Loop until 'q' is pressed
    char key = ' ';
    int i = 0;
    while (/*key != 'q' ||*/ i < 50) {
        if (zed.grab(runtime_parameters) == ERROR_CODE::SUCCESS) {

            // Retrieve the left image, depth image in half-resolution
            zed.retrieveImage(image_zed, VIEW::LEFT, MEM::CPU, new_image_size);

#ifndef HAVE_CUDA 
            // retrieve CPU -> the ocv reference is therefore updated
            zed.retrieveImage(depth_image_zed, VIEW::DEPTH, MEM::CPU, new_image_size);
#else
            // retrieve GPU -> the ocv reference is therefore updated
            zed.retrieveImage(depth_image_zed_gpu, VIEW::DEPTH, MEM::GPU, new_image_size);
#endif
            // Retrieve the RGBA point cloud in half-resolution
            // To learn how to manipulate and display point clouds, see Depth Sensing sample
            zed.retrieveMeasure(point_cloud, MEASURE::XYZRGBA, MEM::CPU, new_image_size);

            Corners corners = CornerDetection(image_ocv); // std::vector<cv::Point2f> p
            // std::cout << "2D pixel points " << endl << endl;
            // std::cout << corners.p << endl;
            if (!corners.p.empty()) {
                std::cout << "Number of found points: " << corners.p.size() << std::endl;
                std::ofstream detected_pointcloud;
                detected_pointcloud.open("detected_pointcloud.xyz");
                std::cout << "3D points from 2D pixels" << endl << endl;
                for (auto pixel : corners.p) {
                    sl::float4 point3D;
                    point_cloud.getValue(pixel.x, pixel.y, &point3D);
                    std::cout << point3D.x << " " << point3D.y << " " << point3D.z << std::endl;
                    if (!std::isnan(point3D.x) && !std::isnan(point3D.y) && !std::isnan(point3D.z)) {
                        detected_pointcloud << point3D.x << " " << point3D.y << " " << point3D.z << std::endl;
                    }
                }
                detected_pointcloud.close();
                //pointcloud_registration("proba.xyz", "proba_Cloud.xyz"); // reference pcl and detected pcl
            }

            // Display image and depth using cv:Mat which share sl:Mat data
            //cv::imshow("Image", image_ocv);
#ifdef HAVE_CUDA
            // download the Ocv GPU data from Device to Host to be displayed
            depth_image_ocv_gpu.download(depth_image_ocv);
#endif
            // cv::imshow("Depth", depth_image_ocv);
            // Handle key event
            key = cv::waitKey(10);
            processKeyEvent(zed, key);
            cv::waitKey(0);
            i++;
        }
    }


#ifdef HAVE_CUDA
    // sl::Mat GPU memory needs to be free before the zed
    depth_image_zed_gpu.free();
#endif
    zed.close();
    return 0;
}

// Mapping between MAT_TYPE and CV_TYPE
int getOCVtype(sl::MAT_TYPE type) {
    int cv_type = -1;
    switch (type) {
        case MAT_TYPE::F32_C1: cv_type = CV_32FC1; break;
        case MAT_TYPE::F32_C2: cv_type = CV_32FC2; break;
        case MAT_TYPE::F32_C3: cv_type = CV_32FC3; break;
        case MAT_TYPE::F32_C4: cv_type = CV_32FC4; break;
        case MAT_TYPE::U8_C1: cv_type = CV_8UC1; break;
        case MAT_TYPE::U8_C2: cv_type = CV_8UC2; break;
        case MAT_TYPE::U8_C3: cv_type = CV_8UC3; break;
        case MAT_TYPE::U8_C4: cv_type = CV_8UC4; break;
        default: break;
    }
    return cv_type;
}

/**
* Conversion function between sl::Mat and cv::Mat
**/
cv::Mat slMat2cvMat(Mat& input) {
    // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
    // cv::Mat and sl::Mat will share a single memory structure
    return cv::Mat(input.getHeight(), input.getWidth(), getOCVtype(input.getDataType()), input.getPtr<sl::uchar1>(MEM::CPU), input.getStepBytes(sl::MEM::CPU));
}

#ifdef HAVE_CUDA
/**
* Conversion function between sl::Mat and cv::Mat
**/
cv::cuda::GpuMat slMat2cvMatGPU(Mat& input) {
    // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
    // cv::Mat and sl::Mat will share a single memory structure
    return cv::cuda::GpuMat(input.getHeight(), input.getWidth(), getOCVtype(input.getDataType()), input.getPtr<sl::uchar1>(MEM::GPU), input.getStepBytes(sl::MEM::GPU));
}
#endif

/**
* This function displays help in console
**/
void printHelp() {
    std::cout << " Press 's' to save Side by side images" << std::endl;
    std::cout << " Press 'p' to save Point Cloud" << std::endl;
    std::cout << " Press 'd' to save Depth image" << std::endl;
    std::cout << " Press 'm' to switch Point Cloud format" << std::endl;
    std::cout << " Press 'n' to switch Depth format" << std::endl;
}

Corners CornerDetection(cv::Mat image_ocv) {
    cv::Mat src1;
    cv::Mat src;
    printf("read image...\n");

    src1 = image_ocv;//cv::imread(simage.c_str(), -1);
    if (src1.channels() == 1)
    {
        src = src1.clone();
    }
    else
    {
        if (src1.channels() == 3)
        {
            cv::cvtColor(src1, src, CV_BGR2GRAY);
        }
        else
        {
            if (src1.channels() == 4)
            {
                cv::cvtColor(src1, src, CV_BGRA2GRAY);
            }
        }
    }

    std::vector<cv::Point> corners_p;// Store the found corners

    double t = (double)cv::getTickCount();
    std::vector<cv::Mat> chessboards;
    CornerDetAC corner_detector(src);
    ChessboradStruct chessboardstruct;

    Corners corners_s;
    corner_detector.detectCorners(src, corners_p, corners_s, 0.01);
    // std::cout << corners_s.p << std::endl;

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "time cost :" << t << std::endl;

    ImageChessesStruct ics;
    chessboardstruct.chessboardsFromCorners(corners_s, chessboards, 0.6);
    chessboardstruct.drawchessboard(src1, corners_s, chessboards, "cb", 0);
    return corners_s;
}

Trafo registration(vector<cv::Point3f>& pts1, std::vector<cv::Point3f>& pts2) {

    Trafo ret;

    int NUM = pts1.size();

    //Subtract offsets

    cv::Point3d offset1(0.0, 0.0, 0.0);
    cv::Point3d offset2(0.0, 0.0, 0.0);

    for (int i = 0; i < NUM; i++) {
        cv::Point3f v1 = pts1[i];
        cv::Point3f v2 = pts2[i];

        offset1.x += v1.x;
        offset1.y += v1.y;
        offset1.z += v1.z;

        offset2.x += v2.x;
        offset2.y += v2.y;
        offset2.z += v2.z;
    }

    offset1.x /= NUM;
    offset1.y /= NUM;
    offset1.z /= NUM;

    offset2.x /= NUM;
    offset2.y /= NUM;
    offset2.z /= NUM;

    ret.offset1.x = offset1.x;
    ret.offset1.y = offset1.y;
    ret.offset1.z = offset1.z;

    ret.offset2.x = offset2.x;
    ret.offset2.y = offset2.y;
    ret.offset2.z = offset2.z;

    cv::Mat H = cv::Mat::zeros(3, 3, CV_32F);
    for (int i = 0; i < NUM; i++) {
        cv::Point3f v1 = pts1[i];
        cv::Point3f v2 = pts2[i];

        float x1 = v1.x - offset1.x;
        float y1 = v1.y - offset1.y;
        float z1 = v1.z - offset1.z;

        float x2 = v2.x - offset2.x;
        float y2 = v2.y - offset2.y;
        float z2 = v2.z - offset2.z;

        H.at<float>(0, 0) += x2 * x1;
        H.at<float>(0, 1) += x2 * y1;
        H.at<float>(0, 2) += x2 * z1;

        H.at<float>(1, 0) += y2 * x1;
        H.at<float>(1, 1) += y2 * y1;
        H.at<float>(1, 2) += y2 * z1;

        H.at<float>(2, 0) += z2 * x1;
        H.at<float>(2, 1) += z2 * y1;
        H.at<float>(2, 2) += z2 * z1;
    }


    cv::Mat w(3, 3, CV_32F);
    cv::Mat u(3, 3, CV_32F);
    cv::Mat vt(3, 3, CV_32F);

    cv::SVD::compute(H, w, u, vt);

    cv::Mat rot = vt.t() * u.t();
    ret.rot = rot;

    float numerator = 0.0;
    float denominator = 0.0;

    for (int i = 0; i < NUM; i++) {
        cv::Point3f v1 = pts1[i];
        cv::Point3f v2 = pts2[i];

        float x1 = v1.x - offset1.x;
        float y1 = v1.y - offset1.y;
        float z1 = v1.z - offset1.z;


        cv::Mat p2(3, 1, CV_32F);

        p2.at<float>(0, 0) = v2.x - offset2.x;
        p2.at<float>(1, 0) = v2.y - offset2.y;
        p2.at<float>(2, 0) = v2.z - offset2.z;

        p2 = rot * p2;

        float x2 = p2.at<float>(0, 0);
        float y2 = p2.at<float>(1, 0);
        float z2 = p2.at<float>(2, 0);


        numerator += x1 * x2 + y1 * y2 + z1 * z2;
        denominator += x2 * x2 + y2 * y2 + z2 * z2;

    }

    ret.scale = numerator / denominator;


    return ret;
}

void pointcloud_registration(std::string reference_pcl, std::string detected_pcl) {
    MatrixReaderWriter mrw1(reference_pcl.c_str());
    MatrixReaderWriter mrw2(detected_pcl.c_str());

    int commonNum = mrw2.rowNum; //atoi("144");

    printf("%d %d\n", mrw1.rowNum, mrw1.columnNum);
    printf("%d %d\n", mrw2.rowNum, mrw2.columnNum);

    vector<cv::Point3f> pts1, pts2;
    for (int i = 0; i < commonNum; i++) {
        float x1 = mrw1.data[3 * i];
        float y1 = mrw1.data[3 * i + 1];
        float z1 = mrw1.data[3 * i + 2];

        float x2 = mrw2.data[3 * i];
        float y2 = mrw2.data[3 * i + 1];
        float z2 = mrw2.data[3 * i + 2];

        cv::Point3f pt1(x1, y1, z1), pt2(x2, y2, z2);
        pts1.push_back(pt1);
        pts2.push_back(pt2);
    }


    Trafo trafo = registration(pts1, pts2);

    //Write result
    float error = 0.0;
    int num = commonNum;
    double* data = new double[2 * num * 3];
    for (int i = 0; i < num; i++) {
        float x1 = mrw1.data[3 * i];
        float y1 = mrw1.data[3 * i + 1];
        float z1 = mrw1.data[3 * i + 2];

        data[6 * i] = x1 - trafo.offset1.x;
        data[6 * i + 1] = y1 - trafo.offset1.y;
        data[6 * i + 2] = z1 - trafo.offset1.z;

        cv::Mat pont1(3, 1, CV_32F);
        pont1.at<float>(0, 0) = x1 - trafo.offset1.x;
        pont1.at<float>(1, 0) = y1 - trafo.offset1.y;
        pont1.at<float>(2, 0) = z1 - trafo.offset1.z;


        float x2 = mrw2.data[3 * i];
        float y2 = mrw2.data[3 * i + 1];
        float z2 = mrw2.data[3 * i + 2];

        cv::Mat p2(3, 1, CV_32F);
        p2.at<float>(0, 0) = x2 - trafo.offset2.x;
        p2.at<float>(1, 0) = y2 - trafo.offset2.y;
        p2.at<float>(2, 0) = z2 - trafo.offset2.z;

        p2 = (trafo.rot * p2) * trafo.scale;

        data[6 * i + 3] = p2.at<float>(0, 0);
        data[6 * i + 4] = p2.at<float>(1, 0);
        data[6 * i + 5] = p2.at<float>(2, 0);

        error += cv::norm(pont1, p2);

    }
    error /= num;
    error = sqrt(error);

    cout << "RMSE:" << error << endl;
    MatrixReaderWriter mrw3(data, 2 * num, 3);
    mrw3.save("res.xyz");
}
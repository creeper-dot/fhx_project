// tasks/kalman_filter.hpp
#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>

namespace auto_aim {

class ArmorKalmanFilter {
private:
    cv::KalmanFilter kf;
    const float dt = 0.02f;
    bool initialized = false;

public:
    ArmorKalmanFilter() : kf(8, 4, 0) {  // 8状态，4测量 [x,y,z,yaw]
        // 状态转移矩阵
        kf.transitionMatrix = (cv::Mat_<float>(8, 8) << 
            1, 0, 0, dt, 0, 0,  0,  0,    // x  = x  + vx·dt
            0, 1, 0,  0, dt, 0,  0,  0,    // y  = y  + vy·dt
            0, 0, 1,  0,  0, dt, 0,  0,    // z  = z  + vz·dt
            0, 0, 0,  1,  0,  0, 0,  0,    // vx = vx
            0, 0, 0,  0,  1,  0, 0,  0,    // vy = vy
            0, 0, 0,  0,  0,  1, 0,  0,    // vz = vz
            0, 0, 0,  0,  0,  0, 1, dt,    // yaw  = yaw + ω·dt
            0, 0, 0,  0,  0,  0, 0,  1);   // ω    = ω   (角速度)

        // 测量矩阵：只测量位置和yaw
        kf.measurementMatrix = cv::Mat::zeros(4, 8, CV_32F);
        kf.measurementMatrix.at<float>(0,0) = 1.0f; // x
        kf.measurementMatrix.at<float>(1,1) = 1.0f; // y  
        kf.measurementMatrix.at<float>(2,2) = 1.0f; // z
        kf.measurementMatrix.at<float>(3,6) = 1.0f; // yaw

        cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-4));
        kf.processNoiseCov.at<float>(3,3) = 1e-2; 
        kf.processNoiseCov.at<float>(4,4) = 1e-2; 
        kf.processNoiseCov.at<float>(7,7) = 1e-1;
        
        cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-2));
        kf.measurementNoiseCov.at<float>(3,3) = 0.1f; // yaw测量噪声
        
        cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1));
    }

    void predict() { 
        kf.predict(); 
        if (initialized) {
            std::cout << "[KF] 预测后状态: ";
            for (int i = 0; i < 8; i++) {
                std::cout << kf.statePost.at<float>(i) << " ";
            }
            std::cout << std::endl;
        }
    }
    
    void correct(const cv::Mat& tvec, double yaw_deg) {
        cv::Mat meas(4, 1, CV_32F);
        meas.at<float>(0) = static_cast<float>(tvec.at<double>(0));
        meas.at<float>(1) = static_cast<float>(tvec.at<double>(1));
        meas.at<float>(2) = static_cast<float>(tvec.at<double>(2)); 
        meas.at<float>(3) = static_cast<float>(yaw_deg * CV_PI / 180.0);

        std::cout << "[KF] 测量值: " << meas.t() << std::endl;

        if (!initialized) {
            // 第一次测量时初始化状态
            kf.statePost.at<float>(0) = meas.at<float>(0);
            kf.statePost.at<float>(1) = meas.at<float>(1); 
            kf.statePost.at<float>(2) = meas.at<float>(2);
            kf.statePost.at<float>(6) = meas.at<float>(3);
            initialized = true;
            std::cout << "[KF] 初始化完成" << std::endl;
            return;
        }
        
        kf.correct(meas);
        std::cout << "[KF] 修正后状态: ";
        for (int i = 0; i < 8; i++) {
            std::cout << kf.statePost.at<float>(i) << " ";
        }
        std::cout << std::endl;
    }
    
    cv::Point3d getPredictedPosition(double dt_ahead = 0.1) const {
        if (!initialized) return cv::Point3d(0, 0, 0);
        
        cv::Mat state = kf.statePost;
        double x = state.at<float>(0) + state.at<float>(3) * dt_ahead;
        double y = state.at<float>(1) + state.at<float>(4) * dt_ahead;
        double z = state.at<float>(2) + state.at<float>(5) * dt_ahead;
        
        std::cout << "[KF] 预测位置: (" << x << ", " << y << ", " << z << ")" << std::endl;
        return cv::Point3d(x, y, z);
    }
    
    double getPredictedYaw() const { 
        if (!initialized) return 0.0;
        double yaw = kf.statePost.at<float>(6) * 180.0 / CV_PI;
        std::cout << "[KF] 预测Yaw: " << yaw << " 度" << std::endl;
        return yaw;
    }
    
    bool isInitialized() const { return initialized; }
};

}  // namespace auto_aim
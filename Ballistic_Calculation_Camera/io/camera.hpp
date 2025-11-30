// io/camera.hpp
#ifndef IO_CAMERA__HPP
#define IO_CAMERA__HPP

#include <opencv2/opencv.hpp>
#include "MvCameraControl.h"   

namespace io
{
    class Camera
    {
    public:
        explicit Camera(int exposure_us = 10000);
        ~Camera();

        bool read(cv::Mat& img);

        void print_info() const;

    private:
        void* handle_ = nullptr;                    
        unsigned char* pData_ = nullptr;            
        MV_FRAME_OUT_INFO_EX stFrameInfo_{};        

        int exposure_us_;                           
        bool is_color_ = true;                      

        static constexpr double fx = 3483.396240;
        static constexpr double fy = 3486.235840;
        static constexpr double cx = 1522.190508;
        static constexpr double cy = 1023.456894;

        static constexpr double k1 = -0.091955;
        static constexpr double k2 =  0.395148;
        static constexpr double p1 =  0.003058;
        static constexpr double p2 = -0.000456;
        static constexpr double k3 =  0.000000;

        cv::Mat map1_, map2_;

        bool init_camera();
        void close_camera();
        void init_undistort_map();
        void set_exposure(int exposure_us);
    };

} // namespace io

#endif 
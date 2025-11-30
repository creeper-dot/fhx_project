#include "camera.hpp"
#include <iostream>
#include <cstring>

namespace io
{
    Camera::Camera(int exposure_us) : exposure_us_(exposure_us)
    {
        if (!init_camera())
            throw std::runtime_error("海康MV-CS060-10UC-PRO 初始化失败！请检查USB线");

        set_exposure(exposure_us_);
        init_undistort_map();
        std::cout << "[Camera] MV-CS060-10UC-PRO 初始化成功！\n";
        print_info();
    }

    Camera::~Camera()
    {
        close_camera();
        if (pData_) {
            delete[] pData_;
            pData_ = nullptr;
        }
    }

    bool Camera::read(cv::Mat& img)
    {
        MV_FRAME_OUT_INFO_EX stFrameInfo{};
        memset(&stFrameInfo, 0, sizeof(stFrameInfo));

        unsigned int nDataSize = 3072 * 2048 * 3 + 2048;

        int nRet = MV_CC_GetOneFrameTimeout(handle_, pData_, nDataSize, &stFrameInfo, 1000);
        if (nRet != MV_OK) {
            return false;
        }

        cv::Mat raw(2048, 3072, CV_8UC3, pData_);
        cv::remap(raw, img, map1_, map2_, cv::INTER_LINEAR);
        return true;
    }

    void Camera::init_undistort_map()
    {
        cv::Mat camera_matrix = (cv::Mat_<double>(3,3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
        cv::Mat dist_coeffs   = (cv::Mat_<double>(1,5) << k1, k2, p1, p2, k3);
        cv::initUndistortRectifyMap(camera_matrix, dist_coeffs, cv::Mat(),
                                    camera_matrix, cv::Size(3072, 2048), CV_16SC2, map1_, map2_);
        std::cout << "[Camera] 去畸变映射表初始化完成\n";
    }

    bool Camera::init_camera()
    {
        MV_CC_DEVICE_INFO_LIST stDeviceList{};
        int nRet = MV_CC_EnumDevices(MV_USB_DEVICE, &stDeviceList);
        if (nRet != MV_OK || stDeviceList.nDeviceNum == 0) {
            std::cerr << "未发现海康相机！\n";
            return false;
        }

        nRet = MV_CC_CreateHandle(&handle_, stDeviceList.pDeviceInfo[0]);
        if (nRet != MV_OK) return false;

        nRet = MV_CC_OpenDevice(handle_);
        if (nRet != MV_OK) {
            MV_CC_DestroyHandle(handle_);
            return false;
        }

        nRet = MV_CC_SetEnumValueByString(handle_, "PixelFormat", "BayerRG8");
        if (nRet != MV_OK) {
            MV_CC_SetEnumValue(handle_, "PixelFormat", 0x01080009U);
        }

        MV_CC_SetBoolValue(handle_, "IspEnable", true);

        pData_ = new unsigned char[3072 * 2048 * 3 + 2048];
        MV_CC_StartGrabbing(handle_);
        return true;
    }

    void Camera::set_exposure(int exposure_us)
    {
        MV_CC_SetEnumValue(handle_, "ExposureAuto", 0);
        MV_CC_SetEnumValue(handle_, "GainAuto", 0);
        MV_CC_SetFloatValue(handle_, "ExposureTime", static_cast<float>(exposure_us));
    }

    void Camera::close_camera()
    {
        if (handle_) {
            MV_CC_StopGrabbing(handle_);
            MV_CC_CloseDevice(handle_);
            MV_CC_DestroyHandle(handle_);
            handle_ = nullptr;
        }
    }

    void Camera::print_info() const
    {
        if (!handle_) return;

        MV_CC_DEVICE_INFO stDeviceInfo{};
        if (MV_CC_GetDeviceInfo(handle_, &stDeviceInfo) == MV_OK &&
            stDeviceInfo.nTLayerType == MV_USB_DEVICE)
        {
            std::cout << "[Camera] 型号: " 
                      << stDeviceInfo.SpecialInfo.stUsb3VInfo.chModelName << "\n";
            std::cout << "[Camera] 序列号: " 
                      << stDeviceInfo.SpecialInfo.stUsb3VInfo.chSerialNumber << "\n";
        }
    }

} // namespace io
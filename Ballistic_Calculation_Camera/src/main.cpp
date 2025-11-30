#include "io/camera.hpp"
#include "tasks/detector.hpp"
#include "tasks/ballistic_solver.hpp"
#include "tools/img_tools.hpp"
#include <opencv2/opencv.hpp>
#include <fmt/core.h>
#include <iostream>
#include <signal.h>

using namespace cv;
using namespace std;

// 全局变量，用于信号处理
bool stop_signal = false;

// 信号处理函数
void signalHandler(int signum) {
    cout << "接收到中断信号，正在退出..." << endl;
    stop_signal = true;
}

static const Mat camera_matrix = (Mat_<double>(3,3) <<
    3483.396240, 0.000000,    1522.190508,
    0.000000,    3486.235840, 1023.456894,
    0.000000,    0.000000,    1.000000
);
static const Mat dist_coeffs = (Mat_<double>(1,5) <<
    -0.091955, 0.395148, 0.003058, -0.000456, 0.000000
);

static const double W = 0.135 / 2, H = 0.056;
static const vector<Point3f> object_points = {
    {-W, H, 0},   // 左上
    {W, H, 0},    // 右上  
    {W, -H, 0},   // 右下
    {-W, -H, 0}   // 左下
};

void draw_data_block(Mat& img, const vector<vector<string>>& data, Point start_pos, 
                    double font_scale = 1.2, int thickness = 3, int line_spacing = 40) {
    int font = FONT_HERSHEY_SIMPLEX;
    Scalar color(255, 255, 255); 
    
    for (size_t i = 0; i < data.size(); i++) {
        string line;
        for (size_t j = 0; j < data[i].size(); j++) {
            line += data[i][j];
            if (j < data[i].size() - 1) {
                line += "  "; 
            }
        }
        
        Point text_pos(start_pos.x, start_pos.y + i * line_spacing);
        putText(img, line, text_pos, font, font_scale, color, thickness, LINE_AA);
    }
}

// 绘制装甲板角点和序号
void draw_armor_points(Mat& img, const vector<Point2f>& points, const Scalar& color = Scalar(0, 255, 255)) {
    for (size_t i = 0; i < points.size(); i++) {
        circle(img, points[i], 8, color, -1);
        putText(img, to_string(i), points[i] + Point2f(5, -5), 
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 0), 2);
    }
}

// 调整图像大小以适应显示
Mat resize_for_display(const Mat& img, double scale = 0.5) {
    Mat resized;
    cv::resize(img, resized, Size(), scale, scale);
    return resized;
}

int main() {
    // 注册信号处理
    signal(SIGINT, signalHandler);
    
    cout << "=== RM自动瞄准系统 - 相机版本 ===" << endl;
    cout << "初始化相机..." << endl;

    // 初始化相机
    io::Camera* camera = nullptr;
    try {
        camera = new io::Camera(1000000); // 曝光时间1000ms
        cout << "相机初始化成功！" << endl;
    } catch (const exception& e) {
        cerr << "相机初始化失败: " << e.what() << endl;
        cerr << "请检查:" << endl;
        cerr << "1. 相机是否正确连接" << endl;
        cerr << "2. USB线是否完好" << endl;
        cerr << "3. 相机驱动是否安装" << endl;
        return -1;
    }

    auto_aim::Detector detector;
    auto_aim::BallisticSolver solver;

    Mat frame;
    bool pause = false;
    int frame_count = 0;
    double fps = 0;
    int64 start_time = getTickCount();

    cout << "开始主循环..." << endl;
    cout << "控制说明:" << endl;
    cout << "  SPACE: 暂停/继续" << endl;
    cout << "  S: 单步模式" << endl;
    cout << "  Q: 退出" << endl;

    namedWindow("RM Vision - 相机实时检测", WINDOW_NORMAL);
    resizeWindow("RM Vision - 相机实时检测", 1200, 800);

    while (!stop_signal) {
        if (!pause) {
            // 从相机读取帧
            if (!camera->read(frame)) {
                cerr << "从相机读取帧失败！" << endl;
                break;
            }

            if (frame.empty()) {
                cerr << "读取到空帧！" << endl;
                continue;
            }

            frame_count++;

            // 计算FPS
            if (frame_count % 30 == 0) {
                int64 end_time = getTickCount();
                fps = 30.0 / ((end_time - start_time) / getTickFrequency());
                start_time = end_time;
            }

            auto armors = detector.detect(frame);

            if (!armors.empty()) {
                auto& a = armors.front();
                
                vector<Point2f> pts = {a.left.top, a.right.top, a.right.bottom, a.left.bottom};
                
                // 绘制装甲板轮廓和角点
                tools::draw_points(frame, pts, Scalar(0,255,0), 6);
                draw_armor_points(frame, pts, Scalar(0, 0, 255));
                
                // 使用EPnP方法
                Mat rvec_epnp, tvec_epnp;
                bool pnp_success = solvePnP(object_points, pts, camera_matrix, dist_coeffs, 
                                           rvec_epnp, tvec_epnp, false, SOLVEPNP_EPNP);
                
                if (pnp_success) {
                    // 计算距离
                    double distance = norm(tvec_epnp);
                    
                    // 计算旋转矩阵和yaw角
                    Mat R; 
                    Rodrigues(rvec_epnp, R);
                    double yaw = atan2(R.at<double>(0,2), R.at<double>(2,2)) * 180 / CV_PI;

                    // 直接使用EPnP结果
                    Point3d pred(tvec_epnp.at<double>(0), tvec_epnp.at<double>(1), tvec_epnp.at<double>(2));
                    double dist = distance;
                    
                    // 弹道解算 
                    auto sol = solver.solve(dist, pred.y, 28.0);
                    double yaw_comp = atan2(pred.x, pred.z) * 180 / CV_PI;
                    double pitch_comp = sol.pitch_angle_deg;

                    // 在图像上绘制坐标系
                    vector<Point2f> image_points;
                    vector<Point3f> axis_points = {
                        {0, 0, 0},  // 原点
                        {0.1, 0, 0}, // X轴
                        {0, 0.1, 0}, // Y轴  
                        {0, 0, 0.1}  // Z轴
                    };
                    projectPoints(axis_points, rvec_epnp, tvec_epnp, camera_matrix, dist_coeffs, image_points);
                    
                    // 绘制坐标系
                    line(frame, image_points[0], image_points[1], Scalar(0, 0, 255), 3); // X轴 - 红色
                    line(frame, image_points[0], image_points[2], Scalar(0, 255, 0), 3); // Y轴 - 绿色
                    line(frame, image_points[0], image_points[3], Scalar(255, 0, 0), 3); // Z轴 - 蓝色
                    
                    // 在坐标系末端添加标签
                    putText(frame, "X", image_points[1], FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 2);
                    putText(frame, "Y", image_points[2], FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
                    putText(frame, "Z", image_points[3], FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 0, 0), 2);

                    vector<vector<string>> display_data = {
                        {fmt::format("YAW: {:.2f}", yaw_comp), 
                         fmt::format("PITCH: {:.2f}", pitch_comp)},
                        {fmt::format("DIST: {:.2f}m", dist),
                         fmt::format("RAW_YAW: {:.2f}", yaw)},
                        {fmt::format("FPS: {:.1f}", fps),
                         fmt::format("V0: 28.0 m/s")}
                    };
                    
                    draw_data_block(frame, display_data, Point(50, 100));

                    // 控制台输出
                    printf("Frame %d - YAW: %.2f, PITCH: %.2f, DIST: %.2fm\n", 
                           frame_count, yaw_comp, pitch_comp, dist);
                    fflush(stdout);
                }
            } else {
                // 没识别到目标时显示提示
                vector<vector<string>> no_target_data = {
                    {"NO TARGET"},
                    {fmt::format("FPS: {:.1f}", fps)}
                };
                draw_data_block(frame, no_target_data, Point(200, 300), 2.0, 4);
            }

            // 调整显示大小并显示
            Mat display_frame = resize_for_display(frame, 0.5);
            imshow("RM Vision - 相机实时检测", display_frame);
        }

        // 键盘输入处理
        char k = (char)waitKey(1);
        if (k == 'q' || k == 'Q') break;
        if (k == ' ') pause = !pause;
        if (k == 's' || k == 'S') { 
            pause = true; 
            // 单步模式下，按任意键继续一帧
            if (pause) {
                waitKey(0);
            }
        }
    }

    // 清理资源
    cout << "正在清理资源..." << endl;
    if (camera) {
        delete camera;
        camera = nullptr;
    }
    destroyAllWindows();
    
    cout << "程序正常退出" << endl;
    return 0;
}
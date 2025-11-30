#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <utility>  // 添加这个头文件用于 std::pair

namespace auto_aim {

struct Solution {
    double pitch_angle_deg;  // 俯仰角度
    double time_of_flight;   // 飞行时间
};

struct SimulationResult {
    double x, y, t;  // 水平距离, 垂直高度, 时间
};

class BallisticSolver {
public:
    Solution solve(double distance_m, double height_offset_m = 0.0, double v0 = 28.0) const {
        double g = 9.79, k = 0.06;
        double low = 0.0, high = 45.0, best_pitch = 20.0, min_error = 1e9;

        for (int i = 0; i < 60; ++i) {
            double mid = (low + high) / 2;
            SimulationResult res = simulate(v0, mid * CV_PI / 180.0, k, g);
            double error = std::fabs(res.x - distance_m) + std::fabs(res.y - height_offset_m) * 5.0;
            if (res.x < distance_m) low = mid; 
            else high = mid;
            if (error < min_error) { 
                min_error = error; 
                best_pitch = mid; 
            }
        }

        SimulationResult final_res = simulate(v0, best_pitch * CV_PI / 180.0, k, g);
        return {best_pitch, final_res.t};
    }

private:
    SimulationResult simulate(double v0, double angle, double k, double g) const {
        double vx = v0 * std::cos(angle), vy = v0 * std::sin(angle), x = 0, y = 0, t = 0, dt = 0.001;
        
        // 修复的 Runge-Kutta 实现
        while (y >= -0.1 && x < 10.0) {
            // k1 阶段
            auto k1 = drag(vx, vy, k);
            double k1vx = k1.first;
            double k1vy = k1.second;
            
            // k2 阶段
            auto k2 = drag(vx + k1vx * dt / 2, vy + k1vy * dt / 2, k);
            double k2vx = k2.first;
            double k2vy = k2.second;
            
            // k3 阶段
            auto k3 = drag(vx + k2vx * dt / 2, vy + k2vy * dt / 2, k);
            double k3vx = k3.first;
            double k3vy = k3.second;
            
            // k4 阶段
            auto k4 = drag(vx + k3vx * dt, vy + k3vy * dt, k);
            double k4vx = k4.first;
            double k4vy = k4.second;

            // 更新速度和位置
            vx += (k1vx + 2*k2vx + 2*k3vx + k4vx) * dt / 6;
            vy += (k1vy + 2*k2vy + 2*k3vy + k4vy) * dt / 6 - g * dt;
            x += vx * dt; 
            y += vy * dt; 
            t += dt;
            
            if (y < 0) break;
        }
        return {x, y, t};
    }

    // 使用 std::pair
    std::pair<double, double> drag(double vx, double vy, double k) const {
        double v = std::sqrt(vx*vx + vy*vy);
        if (v < 1e-6) return {0, 0};
        double F = k * v * v;
        return {-F * vx / v, -F * vy / v};
    }
};

}  // namespace auto_aim
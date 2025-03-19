#ifndef LANE_DETECTION_HPP
#define LANE_DETECTION_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <chrono>
#include <sstream>
#include <stdexcept>
#include <memory>

// --------------------------------------------------
// 1) 모드 설정 enum class (카메라 vs 동영상)
// --------------------------------------------------
enum class DetectionMode {
    Camera,
    Video
};

// --------------------------------------------------
// 2) 파이프라인 클래스 선언
// --------------------------------------------------
class Pipeline {
public:
    // 생성자
    Pipeline();

    // 차선 검출 시작 함수
    void startDetection(DetectionMode mode, bool showWindows = true);

private:
    // --- 상수 정의 ---
    static constexpr int kWidth = 1280;
    static constexpr int kHeight = 720;
    static constexpr int kWindowMargin = 480; // 슬라이딩 윈도우 탐색용 margin

    // ROI 꼭짓점 (좌하, 좌상, 우상, 우하)
    std::vector<cv::Point> vertices_;

    // 투시변환 행렬
    cv::Mat transform_matrix_;
    cv::Mat inv_transform_matrix_;

    // 슬라이딩 윈도우 검색용
    int leftx_mid_;
    int rightx_mid_;
    int leftx_base_;
    int rightx_base_;

    // 곡선 피팅 계수 보관
    std::vector<double> left_a_;
    std::vector<double> left_b_;
    std::vector<double> left_c_;
    std::vector<double> right_a_;
    std::vector<double> right_b_;
    std::vector<double> right_c_;

    // y좌표 (0 ~ height-1)
    std::vector<double> ploty_;

    // 조향각
    double prev_angle_;
    double steering_angle_;

    // 시각화 여부, 종료 플래그
    bool visible_;
    bool exit_flag_;

private:
    // -----------------------------
    // 내부에서 사용될 함수들
    // -----------------------------
    cv::Mat toGray(const cv::Mat& img) const;
    cv::Mat noiseRemoval(const cv::Mat& img) const;
    cv::Mat edgeDetection(const cv::Mat& img) const;
    cv::Mat morphologyClose(const cv::Mat& img) const;
    cv::Mat applyROI(const cv::Mat& img) const;
    cv::Mat perspectiveTransform(const cv::Mat& img) const;
    cv::Mat invPerspectiveTransform(const cv::Mat& img) const;

    bool slidingWindow(const cv::Mat& binary_img, 
                       int nwindows, int margin, int minpix,
                       bool draw_windows,
                       cv::Mat& out_img,
                       std::vector<double>& left_fitx,
                       std::vector<double>& right_fitx,
                       bool& left_lane_detected,
                       bool& right_lane_detected);

    double getAngleOnLane(const std::vector<double>& left_fitx,
                          const std::vector<double>& right_fitx,
                          bool left_lane_detected,
                          bool right_lane_detected);

    cv::Mat displayHeadingLine(const cv::Mat& base_img,
                               const cv::Mat& overlay_img,
                               double steering_angle) const;

    // 실제 매 프레임에 대한 파이프라인 처리
    cv::Mat processFrame(const cv::Mat& frame);
};

#endif // LANE_DETECTION_HPP

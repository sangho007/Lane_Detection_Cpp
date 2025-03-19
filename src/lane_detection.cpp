#include "Lane_detection.hpp"

// 편의상, 최소제곱법/평균 계산 등을 위한 유틸리티 함수들은
// 별도 namespace로 내부 구현. (hpp에 공개 X)
namespace utils {

inline bool polyfit1D(const std::vector<double>& xs,
                      const std::vector<double>& ys,
                      double& a, double& b) 
{
    // y = a*x + b
    if (xs.size() < 2 || xs.size() != ys.size()) {
        return false;
    }

    const double n = static_cast<double>(xs.size());
    double sum_x = 0.0, sum_y = 0.0, sum_x2 = 0.0, sum_xy = 0.0;
    for (std::size_t i = 0; i < xs.size(); ++i) {
        sum_x  += xs[i];
        sum_y  += ys[i];
        sum_x2 += xs[i] * xs[i];
        sum_xy += xs[i] * ys[i];
    }

    const double denom = n * sum_x2 - sum_x * sum_x;
    if (std::fabs(denom) < 1e-12) {
        return false;
    }

    a = (n * sum_xy - sum_x * sum_y) / denom;
    b = (sum_y - a * sum_x) / n;
    return true;
}

inline bool polyfit2D(const std::vector<double>& xs,
                      const std::vector<double>& ys,
                      double& a, double& b, double& c) 
{
    // y = a*x^2 + b*x + c
    if (xs.size() < 3 || xs.size() != ys.size()) {
        return false;
    }

    const std::size_t n = xs.size();
    double x_sum   = 0.0, x2_sum  = 0.0, x3_sum  = 0.0, x4_sum  = 0.0;
    double y_sum   = 0.0, xy_sum  = 0.0, x2y_sum = 0.0;
    double one_sum = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        const double x = xs[i];
        const double y = ys[i];
        one_sum    += 1.0;
        x_sum      += x;
        x2_sum     += x*x;
        x3_sum     += x*x*x;
        x4_sum     += x*x*x*x;
        y_sum      += y;
        xy_sum     += x*y;
        x2y_sum    += x*x*y;
    }

    // Normal equation 항들
    const double a11 = x4_sum;    const double a12 = x3_sum;    const double a13 = x2_sum;
    const double a21 = x3_sum;    const double a22 = x2_sum;    const double a23 = x_sum;
    const double a31 = x2_sum;    const double a32 = x_sum;     const double a33 = one_sum;

    const double b1 = x2y_sum;    const double b2 = xy_sum;     const double b3 = y_sum;

    // 3x3 행렬식
    auto det3x3 = [](double m11, double m12, double m13,
                     double m21, double m22, double m23,
                     double m31, double m32, double m33) {
        return m11*(m22*m33 - m23*m32)
             - m12*(m21*m33 - m23*m31)
             + m13*(m21*m32 - m22*m31);
    };

    const double det = det3x3(a11, a12, a13,
                              a21, a22, a23,
                              a31, a32, a33);
    if (std::fabs(det) < 1e-12) {
        return false;
    }

    // Cramer's rule
    const double det_a = det3x3(b1, a12, a13,
                                b2, a22, a23,
                                b3, a32, a33);
    const double det_b = det3x3(a11, b1, a13,
                                a21, b2, a23,
                                a31, b3, a33);
    const double det_c = det3x3(a11, a12, b1,
                                a21, a22, b2,
                                a31, a32, b3);

    a = det_a / det;
    b = det_b / det;
    c = det_c / det;

    return true;
}

inline double meanOfLast10(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    const int len = static_cast<int>(v.size());
    const int start = (len > 10) ? (len - 10) : 0;
    double sum = 0.0;
    for (int i = start; i < len; ++i) {
        sum += v[i];
    }
    return sum / (len - start);
}

inline std::vector<std::vector<int>> getNonzeroPointsByRow(const cv::Mat& binary_img) {
    std::vector<std::vector<int>> result(binary_img.rows);

    if (binary_img.type() != CV_8UC1) {
        std::cerr << "[getNonzeroPointsByRow] Expected a single-channel (CV_8UC1) image.\n";
        return result;
    }

    const int rows = binary_img.rows;
    const int cols = binary_img.cols;
    // row stride
    const std::size_t step = binary_img.step;
    const uchar* data = binary_img.ptr<uchar>(0);

    for (int y = 0; y < rows; ++y) {
        const uchar* row_ptr = data + y * step;
        for (int x = 0; x < cols; ++x) {
            if (row_ptr[x] != 0) {
                result[y].push_back(x);
            }
        }
    }
    return result;
}

// 간단한 hconcat2 예시(직접 구현)
inline void hconcat2(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& dst) {
    std::vector<cv::Mat> srcs { img1, img2 };
    cv::hconcat(srcs, dst);
}

} // namespace utils


// --------------------------------------------------
// 3) Pipeline 클래스 구현부
// --------------------------------------------------

// 생성자
Pipeline::Pipeline()
    : prev_angle_(0.0),
      steering_angle_(0.0),
      visible_(true),
      exit_flag_(false)
{
    // ROI 꼭짓점 (좌하, 좌상, 우상, 우하)
    vertices_.push_back(cv::Point(200, kHeight - 100));
    vertices_.push_back(cv::Point(kWidth / 2 - 100, kHeight / 2 + 120));
    vertices_.push_back(cv::Point(kWidth / 2 + 100, kHeight / 2 + 120));
    vertices_.push_back(cv::Point(kWidth - 200, kHeight - 100));

    // 투시 변환 좌표
    std::vector<cv::Point2f> points_src {
        {200.0f,        static_cast<float>(kHeight - 100)},
        {static_cast<float>(kWidth/2 - 100), static_cast<float>(kHeight/2 + 120)},
        {static_cast<float>(kWidth/2 + 100), static_cast<float>(kHeight/2 + 120)},
        {static_cast<float>(kWidth - 200),   static_cast<float>(kHeight - 100)}
    };
    // Bird's-eye view에서의 목표 좌표
    std::vector<cv::Point2f> points_dst {
        {200.0f,        static_cast<float>(kHeight)},
        {300.0f,        0.0f},
        {static_cast<float>(kWidth - 300), 0.0f},
        {static_cast<float>(kWidth - 200), static_cast<float>(kHeight)}
    };

    transform_matrix_ = cv::getPerspectiveTransform(points_src, points_dst);
    inv_transform_matrix_ = cv::getPerspectiveTransform(points_dst, points_src);

    // 기본 중앙값
    leftx_mid_  = kWidth / 4;
    rightx_mid_ = kWidth * 3 / 4;
    leftx_base_ = leftx_mid_;
    rightx_base_ = rightx_mid_;

    // ploty 세팅
    ploty_.reserve(kHeight);
    for (int i = 0; i < kHeight; ++i) {
        ploty_.push_back(static_cast<double>(i));
    }

    // 초기값
    left_a_.push_back(0.0);
    left_b_.push_back(0.0);
    left_c_.push_back(static_cast<double>(leftx_mid_));

    right_a_.push_back(0.0);
    right_b_.push_back(0.0);
    right_c_.push_back(static_cast<double>(rightx_mid_));
}

// --------------------------------------------------
// 전처리 함수들
// --------------------------------------------------
cv::Mat Pipeline::toGray(const cv::Mat& img) const {
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

cv::Mat Pipeline::noiseRemoval(const cv::Mat& img) const {
    cv::Mat dst;
    cv::GaussianBlur(img, dst, cv::Size(5, 5), 0.0);
    return dst;
}

cv::Mat Pipeline::edgeDetection(const cv::Mat& img) const {
    cv::Mat edges;
    cv::Canny(img, edges, 200.0, 350.0, 3, false);
    return edges;
}

cv::Mat Pipeline::morphologyClose(const cv::Mat& img) const {
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
    cv::Mat dst;
    cv::morphologyEx(img, dst, cv::MORPH_CLOSE, kernel);
    return dst;
}

cv::Mat Pipeline::applyROI(const cv::Mat& img) const {
    cv::Mat mask = cv::Mat::zeros(img.size(), img.type());
    std::vector<std::vector<cv::Point>> contours { vertices_ };
    cv::fillPoly(mask, contours, cv::Scalar(255,255,255));

    cv::Mat masked_img;
    cv::bitwise_and(img, mask, masked_img);
    return masked_img;
}

cv::Mat Pipeline::perspectiveTransform(const cv::Mat& img) const {
    cv::Mat result;
    cv::warpPerspective(img, result, transform_matrix_,
                        cv::Size(kWidth, kHeight), cv::INTER_LINEAR);
    return result;
}

cv::Mat Pipeline::invPerspectiveTransform(const cv::Mat& img) const {
    cv::Mat result;
    cv::warpPerspective(img, result, inv_transform_matrix_,
                        cv::Size(kWidth, kHeight), cv::INTER_LINEAR);
    return result;
}

// --------------------------------------------------
// 슬라이딩 윈도우 기반 차선 검출
// --------------------------------------------------
bool Pipeline::slidingWindow(const cv::Mat& binary_img, 
                             int nwindows, int margin, int minpix,
                             bool draw_windows,
                             cv::Mat& out_img,
                             std::vector<double>& left_fitx,
                             std::vector<double>& right_fitx,
                             bool& left_lane_detected,
                             bool& right_lane_detected)
{
    // row별 nonzero x좌표
    auto nonzero_points_by_row = utils::getNonzeroPointsByRow(binary_img);

    // 시각화용 3채널 이미지
    cv::Mat color_channels[3] = { binary_img, binary_img, binary_img };
    cv::merge(color_channels, 3, out_img);

    // window 이미지를 더 크게 잡아 표시
    cv::Mat window_img = cv::Mat::zeros(binary_img.rows, 
                                        binary_img.cols + 2*margin, 
                                        CV_8UC3);

    {
        cv::Rect roi(margin, 0, binary_img.cols, binary_img.rows);
        cv::Mat sub = window_img(roi);
        out_img.copyTo(sub);
    }

    const int midpoint = kWidth / 2;
    const int window_height = kHeight / nwindows;

    int leftx_current  = leftx_base_;
    int rightx_current = rightx_base_;
    int leftx_past     = leftx_current;
    int rightx_past    = rightx_current;

    std::vector<std::pair<int,int>> left_lane_points;
    std::vector<std::pair<int,int>> right_lane_points;
    left_lane_points.reserve(7000);  // 대략 여유치
    right_lane_points.reserve(7000); // 대략 여유치

    // 아래서 위로 탐색
    for (int window_i = 0; window_i < nwindows; ++window_i) {
        const int win_y_low = kHeight - (window_i + 1)*window_height;
        const int win_y_high = kHeight - window_i*window_height;

        const int win_xleft_low  = leftx_current - margin;
        const int win_xleft_high = leftx_current + margin;
        const int win_xright_low  = rightx_current - margin;
        const int win_xright_high = rightx_current + margin;

        if (draw_windows) {
            // out_img 위 사각형(왼/오)
            cv::rectangle(out_img,
                          cv::Point(win_xleft_low,  win_y_low),
                          cv::Point(win_xleft_high, win_y_high),
                          cv::Scalar(0,255,0), 2);
            cv::rectangle(out_img,
                          cv::Point(win_xright_low,  win_y_low),
                          cv::Point(win_xright_high, win_y_high),
                          cv::Scalar(0,255,0), 2);

            // window_img에도 표시(색만 다름)
            cv::rectangle(window_img,
                          cv::Point(win_xleft_low + margin,  win_y_low),
                          cv::Point(win_xleft_high + margin, win_y_high),
                          cv::Scalar(255,100,100), 1);
            cv::rectangle(window_img,
                          cv::Point(win_xright_low + margin,  win_y_low),
                          cv::Point(win_xright_high + margin, win_y_high),
                          cv::Scalar(255,100,100), 1);
        }

        int good_left_count  = 0;
        int good_right_count = 0;

        // 윈도우 내부 픽셀 찾기
        for (int y = win_y_low; y < win_y_high; ++y) {
            if (y < 0 || y >= kHeight) continue;
            const auto& row_vec = nonzero_points_by_row[y];
            for (auto x : row_vec) {
                // 왼쪽
                if (x >= win_xleft_low && x < win_xleft_high) {
                    left_lane_points.emplace_back(y, x);
                    ++good_left_count;
                }
                // 오른쪽
                if (x >= win_xright_low && x < win_xright_high) {
                    right_lane_points.emplace_back(y, x);
                    ++good_right_count;
                }
            }
        }

        // 픽셀이 일정 이상이면 평균 x로 갱신
        if (good_left_count > minpix) {
            double sum_x = 0.0;
            for (int i = 0; i < good_left_count; ++i) {
                sum_x += left_lane_points[left_lane_points.size() - 1 - i].second;
            }
            leftx_current = static_cast<int>(sum_x / good_left_count);
        }
        if (good_right_count > minpix) {
            double sum_x = 0.0;
            for (int i = 0; i < good_right_count; ++i) {
                sum_x += right_lane_points[right_lane_points.size() - 1 - i].second;
            }
            rightx_current = static_cast<int>(sum_x / good_right_count);
        }

        // 한쪽 픽셀이 부족하면 반대쪽 이동량 참고
        if (good_left_count < minpix) {
            leftx_current += (rightx_current - rightx_past);
        }
        if (good_right_count < minpix) {
            rightx_current += (leftx_current - leftx_past);
        }

        // 첫 윈도우에서 기준점 보정
        if (window_i == 0) {
            if (leftx_current > midpoint + 40) {
                leftx_current = midpoint + 40;
            }
            if (leftx_current < 0) {
                leftx_current = 0;
            }
            if (rightx_current < midpoint - 40) {
                rightx_current = midpoint - 40;
            }
            if (rightx_current > kWidth) {
                rightx_current = kWidth;
            }
            leftx_base_  = leftx_current;
            rightx_base_ = rightx_current;
        }

        leftx_past  = leftx_current;
        rightx_past = rightx_current;
    }

    // 좌우 픽셀 분리
    std::vector<double> leftx_vals, lefty_vals;
    std::vector<double> rightx_vals, righty_vals;
    leftx_vals.reserve(left_lane_points.size());
    lefty_vals.reserve(left_lane_points.size());
    rightx_vals.reserve(right_lane_points.size());
    righty_vals.reserve(right_lane_points.size());

    for (auto& p : left_lane_points) {
        lefty_vals.push_back(static_cast<double>(p.first));
        leftx_vals.push_back(static_cast<double>(p.second));
    }
    for (auto& p : right_lane_points) {
        righty_vals.push_back(static_cast<double>(p.first));
        rightx_vals.push_back(static_cast<double>(p.second));
    }

    left_lane_detected  = (leftx_vals.size() >= 5000);
    right_lane_detected = (rightx_vals.size() >= 5000);

    // 2차 곡선 피팅
    if (left_lane_detected) {
        double a_, b_, c_;
        if (utils::polyfit2D(lefty_vals, leftx_vals, a_, b_, c_)) {
            left_a_.push_back(a_);
            left_b_.push_back(b_);
            left_c_.push_back(c_);
        }
    }
    if (right_lane_detected) {
        double a_, b_, c_;
        if (utils::polyfit2D(righty_vals, rightx_vals, a_, b_, c_)) {
            right_a_.push_back(a_);
            right_b_.push_back(b_);
            right_c_.push_back(c_);
        }
    }

    // 시각화용 색칠
    cv::Mat left_mask  = cv::Mat::zeros(binary_img.size(), CV_8UC1);
    cv::Mat right_mask = cv::Mat::zeros(binary_img.size(), CV_8UC1);
    for (auto& p : left_lane_points) {
        if (p.first >= 0 && p.first < kHeight &&
            p.second >= 0 && p.second < kWidth)
        {
            left_mask.at<uchar>(p.first, p.second) = 255;
        }
    }
    for (auto& p : right_lane_points) {
        if (p.first >= 0 && p.first < kHeight &&
            p.second >= 0 && p.second < kWidth)
        {
            right_mask.at<uchar>(p.first, p.second) = 255;
        }
    }
    // BGR (왼=파랑, 오른=빨강)
    out_img.setTo(cv::Scalar(255,0,0), left_mask);
    out_img.setTo(cv::Scalar(0,0,255), right_mask);

    // 최근 10개 계수 평균
    double la = utils::meanOfLast10(left_a_);
    double lb = utils::meanOfLast10(left_b_);
    double lc = utils::meanOfLast10(left_c_);
    double ra = utils::meanOfLast10(right_a_);
    double rb = utils::meanOfLast10(right_b_);
    double rc = utils::meanOfLast10(right_c_);

    left_fitx.clear();
    right_fitx.clear();
    left_fitx.reserve(ploty_.size());
    right_fitx.reserve(ploty_.size());

    for (auto yv : ploty_) {
        const double lx = la*yv*yv + lb*yv + lc;
        const double rx = ra*yv*yv + rb*yv + rc;
        left_fitx.push_back(lx);
        right_fitx.push_back(rx);
    }

    // 양쪽 다 인식 안 된 경우
    if (!left_lane_detected && !right_lane_detected) {
        leftx_base_  = leftx_mid_ - 30;
        rightx_base_ = rightx_mid_ + 30;
    }

    return true;
}

// --------------------------------------------------
// 조향각 계산
// --------------------------------------------------
double Pipeline::getAngleOnLane(const std::vector<double>& left_fitx,
                                const std::vector<double>& right_fitx,
                                bool left_lane_detected,
                                bool right_lane_detected)
{
    using namespace utils;

    if (!left_lane_detected && !right_lane_detected) {
        return prev_angle_;
    }

    double slope = 0.0;

    // 한쪽만 인식되는 경우
    if ((left_lane_detected && !right_lane_detected) ||
        (!left_lane_detected && right_lane_detected))
    {
        std::vector<double> X, Y;
        if (left_lane_detected) {
            X = left_fitx;
            Y = ploty_;
        } else {
            X = right_fitx;
            Y = ploty_;
        }
        double a, b;
        if (polyfit1D(X, Y, a, b)) {
            slope = a;
        }
    }
    else {
        // 양쪽 모두 인식
        double la, lb;
        double ra, rb;
        // 두 선 각각 (x -> y)로 1차회귀
        polyfit1D(left_fitx,  ploty_, la, lb);  
        polyfit1D(right_fitx, ploty_, ra, rb);  

        // 거의 평행한 경우
        if (std::fabs(la - ra) < 1e-12) {
            const double inter_x = -(lb + rb) / (2.0 * la);
            const double inter_y = 0.0;
            slope = (static_cast<double>(kHeight) - inter_y) 
                  / (static_cast<double>(kWidth)/2.0 - inter_x);
        } else {
            const double inter_x = (rb - lb) / (la - ra);
            const double inter_y = la * inter_x + lb;
            slope = (static_cast<double>(kHeight) - inter_y) 
                  / (static_cast<double>(kWidth)/2.0 - inter_x);
        }
    }

    // 라디안→도(deg)
    double angle = std::atan(slope) * 180.0 / CV_PI;
    // 0도를 기준으로 +90 or -90
    if (angle > 0.0) {
        angle -= 90.0;
        if (angle <= -20.0) {
            angle = -20.0;
        }
    } else {
        angle += 90.0;
        if (angle >= 20.0) {
            angle = 20.0;
        }
    }
    prev_angle_ = angle;
    return angle;
}

// --------------------------------------------------
// 시각화: 조향각 그리기
// --------------------------------------------------
cv::Mat Pipeline::displayHeadingLine(const cv::Mat& base_img,
                                     const cv::Mat& overlay_img,
                                     double steering_angle) const
{
    cv::Mat img_clone = base_img.clone();
    const int w = img_clone.cols;
    const int h = img_clone.rows;
    if (w <= 0 || h <= 0) {
        return img_clone;
    }

    double angle = steering_angle;
    if (angle > 0.0) {
        angle += 90.0;
    } else if (angle < 0.0) {
        angle -= 90.0;
    } else {
        angle = 90.0;
    }

    const double rad = angle * CV_PI / 180.0;
    const int x1 = w / 2;
    const int y1 = h;
    // h/2 정도 되는 길이로 조향선 표현
    const int x2 = static_cast<int>(x1 - (kHeight/2.0)/std::tan(rad));
    const int y2 = static_cast<int>(h * 3.5 / 5.0);

    // 초록색 조향선
    cv::line(img_clone, cv::Point(x1, y1), cv::Point(x2, y2),
             cv::Scalar(0,255,0), 5);

    // 가중합성
    cv::Mat heading_image;
    cv::addWeighted(img_clone, 1.0, overlay_img, 1.0, 0.0, heading_image);
    return heading_image;
}

// --------------------------------------------------
// 한 프레임 처리
// --------------------------------------------------
cv::Mat Pipeline::processFrame(const cv::Mat& frame) {
    const auto start_time = std::chrono::high_resolution_clock::now();

    // 원본 복사
    cv::Mat img = frame.clone();

    // 전처리
    cv::Mat gray   = toGray(img);
    cv::Mat blur   = noiseRemoval(gray);
    cv::Mat edges  = edgeDetection(blur);
    cv::Mat closed = morphologyClose(edges);
    cv::Mat roi_img = applyROI(closed);
    cv::Mat birds_eye = perspectiveTransform(roi_img);

    // 슬라이딩 윈도우
    cv::Mat sliding_window_img;
    std::vector<double> left_fitx, right_fitx;
    bool left_detected = false;
    bool right_detected = false;

    slidingWindow(birds_eye, 15, 100, 50, true,
                  sliding_window_img, left_fitx, right_fitx,
                  left_detected, right_detected);

    // 조향각
    steering_angle_ = getAngleOnLane(left_fitx, right_fitx, left_detected, right_detected);

    // 시각화 여부
    if (visible_) {
        // 슬라이딩 윈도우 결과 + 조향선
        cv::Mat sliding_with_line = displayHeadingLine(sliding_window_img, 
                                                       sliding_window_img, 
                                                       steering_angle_);
        // 역투시 변환 후 원본에 합성
        cv::Mat inv_trans = invPerspectiveTransform(sliding_with_line);

        cv::Mat total_processed;
        cv::addWeighted(img, 1.0, inv_trans, 1.0, 0.0, total_processed);

        // 텍스트 표시(조향각, FPS)
        std::ostringstream angle_oss, fps_oss;
        angle_oss << "Angle: " << static_cast<int>(steering_angle_);

        const auto end_time = std::chrono::high_resolution_clock::now();
        float elapsed = std::chrono::duration<float>(end_time - start_time).count();
        float fps     = 1.0f / elapsed;
        fps_oss << "FPS: " << fps;

        cv::putText(total_processed, 
                    angle_oss.str(), 
                    cv::Point(20, 100),
                    cv::FONT_HERSHEY_SIMPLEX, 
                    2.0, 
                    cv::Scalar(255,255,255), 
                    2);
        cv::putText(total_processed, 
                    fps_oss.str(),
                    cv::Point(total_processed.cols - 400, 50),
                    cv::FONT_HERSHEY_SIMPLEX, 
                    2.0, 
                    cv::Scalar(255,255,255), 
                    2);

        // 좌: 슬라이딩윈도우 결과, 우: 최종 병합
        cv::Mat merged;
        cv::Mat sliding_texted = sliding_with_line.clone();
        // 동일한 Angle 표시(왼쪽에도)
        cv::putText(sliding_texted, 
                    angle_oss.str(), 
                    cv::Point(20, 100),
                    cv::FONT_HERSHEY_SIMPLEX, 
                    2.0, 
                    cv::Scalar(255,255,255), 
                    2);

        utils::hconcat2(sliding_texted, total_processed, merged);
        return merged;
    } 
    else {
        // 비표시 모드면 빈 Mat 반환
        return cv::Mat();
    }
}

// --------------------------------------------------
// 차선 검출 시작
// --------------------------------------------------
void Pipeline::startDetection(DetectionMode mode, bool showWindows) {
    visible_ = showWindows;

    // VideoCapture 생성
    cv::VideoCapture cap;
    try {
        if (mode == DetectionMode::Camera) {
            if (!cap.open(0)) {  // 웹캠
                throw std::runtime_error("Failed to open camera (index 0).");
            }
        } else {
            // 필요에 따라 파일 경로 수정
            if (!cap.open("./video/challenge.mp4")) {
                throw std::runtime_error("Failed to open video file: ./video/challenge.mp4");
            }
        }
    } 
    catch (const std::exception& e) {
        std::cerr << "[startDetection] Exception in opening capture: " << e.what() << std::endl;
        return;
    }

    if (visible_) {
        cv::namedWindow("CAM View", cv::WINDOW_AUTOSIZE);
    }

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "[startDetection] No more frames or failed to read.\n";
            break;
        }

        // 해상도 1280x720 리사이즈
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(kWidth, kHeight));

        const auto loop_start = std::chrono::high_resolution_clock::now();
        cv::Mat merged = processFrame(resized);

        if (visible_) {
            cv::imshow("CAM View", merged);
            const int key = cv::waitKey(1);
            if (key == 'q' || key == 27) { // ESC
                exit_flag_ = true;
            }
            if (exit_flag_) {
                break;
            }
        }

        const auto loop_end = std::chrono::high_resolution_clock::now();
        const float elapsed = std::chrono::duration<float>(loop_end - loop_start).count();
        const float fps = 1.0f / elapsed;
        std::cout << "Loop FPS: " << fps << std::endl;
    }
}

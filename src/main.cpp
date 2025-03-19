#include <iostream>
#include <memory>
#include "Lane_detection.hpp"

int main() 
{
    // RAII 예시: 스마트 포인터로 Pipeline 관리
    std::unique_ptr<Pipeline> pipeline = std::make_unique<Pipeline>();

    try {
        // 원하는 모드(Camera / Video) 설정
        //  - DetectionMode::Camera → 웹캠
        //  - DetectionMode::Video  → 동영상
        pipeline->startDetection(DetectionMode::Video, true);
    }
    catch (const std::exception& e) {
        std::cerr << "[main] Exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

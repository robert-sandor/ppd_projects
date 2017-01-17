//
// Created by robert on 1/17/17.
//
#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>

#include "cuda.h"

static const char *const ORIGINAL_IMAGE_PATH = "/home/robert/Pictures/Wallpapers/the_last_of_us_2_screenshot_2.png";

using namespace std::chrono;

int main() {
    milliseconds start = duration_cast<milliseconds>(system_clock::now().time_since_epoch());

    cv::Mat image = cv::imread(ORIGINAL_IMAGE_PATH, 1);

    if (!image.data) {
        std::cout << "image_cuda : No image data!\n";
        return -1;
    }

    uint8_t *imageData = image.data;

    if (image.isContinuous()) {
        image_cuda((char *) imageData, (image.total() * image.elemSize()), image.rows, image.cols, image.channels());
    } else {
        std::cout << "image_cuda : Image is not continous!\n";
    }

    cv::imwrite("./cuda_image.png", image);
    milliseconds end = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
    std::cout << "Cuda took : " << (end - start).count() << " ms";
}
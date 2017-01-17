#include <iostream>
#include <opencv2/opencv.hpp>
#include <thread>

static const int NUM_THREADS = 20;
static const char *const ORIGINAL_IMAGE_PATH = "/home/robert/Pictures/Wallpapers/the_last_of_us_2_screenshot_2.png";
static const int COLOR_CHANGE_AMOUNT = 25;

void red_filter_threads(uint8_t *imageData, int rows, int columns, int channels, int thread_id) {
    int from = (int) (ceil(rows / NUM_THREADS) * thread_id);
    int to = (int) (ceil(rows / NUM_THREADS) * (thread_id + 1));

    for (int x = from; x < to; x++) {
        for (int y = 0; y < columns; y++) {
            imageData[x * columns * channels + y * channels] = (uint8_t) (
                    imageData[x * columns * channels + y * channels] - COLOR_CHANGE_AMOUNT < 0 ? 0 :
                    imageData[x * columns * channels + y * channels] - COLOR_CHANGE_AMOUNT);
            imageData[x * columns * channels + y * channels + 1] = (uint8_t) (
                    imageData[x * columns * channels + y * channels + 1] - COLOR_CHANGE_AMOUNT < 0 ? 0 :
                    imageData[x * columns * channels + y * channels + 1] - COLOR_CHANGE_AMOUNT);
            imageData[x * columns * channels + y * channels + 2] = (uint8_t) (
                    imageData[x * columns * channels + y * channels + 2] + COLOR_CHANGE_AMOUNT > 255 ? 0 :
                    imageData[x * columns * channels + y * channels + 2] + COLOR_CHANGE_AMOUNT);
        }
    }
}

using namespace std::chrono;

int main() {
    milliseconds start = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
    cv::Mat image = cv::imread(ORIGINAL_IMAGE_PATH, 1);

    if (!image.data) {
        std::cout << "image_threads : No image data!\n";
        return -1;
    }

    uint8_t *imageData = image.data;

    std::thread threads[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i] = std::thread(red_filter_threads, imageData, image.rows, image.cols, image.channels(), i);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }

    cv::imwrite("./threads_image.png", image);
    milliseconds end = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
    std::cout << "Cuda took : " << (end - start).count() << " ms";
    return 0;
}
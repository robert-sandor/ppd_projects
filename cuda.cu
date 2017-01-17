#include <iostream>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "cuda.h"

static const int NUM_THREADS = 20;
static const int COLOR_CHANGE_AMOUNT = 25;

__global__ void red_filter(char* imageData, int rows, int columns, int channels) {
    int thread_id = threadIdx.x;
    int from = rows / NUM_THREADS * thread_id;
    int to = rows / NUM_THREADS * (thread_id + 1);

    for (int x = from; x < to; x++) {
        for (int y = 0; y < columns; y++) {
            imageData[x * columns * channels + y * channels] =
                    (char) (imageData[x * columns * channels + y * channels] - COLOR_CHANGE_AMOUNT < 0 ? 0 :
                                        imageData[x * columns * channels + y * channels] - COLOR_CHANGE_AMOUNT);
            imageData[x * columns * channels + y * channels + 1] =
                    (char) (imageData[x * columns * channels + y * channels + 1] - COLOR_CHANGE_AMOUNT < 0 ? 0 :
                                        imageData[x * columns * channels + y * channels + 1] - COLOR_CHANGE_AMOUNT);
            imageData[x * columns * channels + y * channels + 2] =
                    (char) (imageData[x * columns * channels + y * channels + 2] + COLOR_CHANGE_AMOUNT > 255 ? 0 :
                                        imageData[x * columns * channels + y * channels + 2] + COLOR_CHANGE_AMOUNT);
        }
    }
}

void image_cuda(char *imageData, size_t size, int rows, int cols, int channels) {
    char *dev_image;

    cudaMalloc((void **) &dev_image, size);
    cudaMemcpy(dev_image, imageData, size, cudaMemcpyHostToDevice);

    red_filter << < 1, NUM_THREADS >> > (dev_image, rows, cols, channels);

    cudaMemcpy(imageData, dev_image, size, cudaMemcpyDeviceToHost);
    cudaFree(dev_image);
}
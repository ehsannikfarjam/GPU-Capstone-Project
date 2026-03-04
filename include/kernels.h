#ifndef KERNELS_H
#define KERNELS_H

#include <cuda_runtime.h>

// Filter configurations
#define TILE_SIZE 16
#define RADIUS 1
#define BLUR_SIZE 3
#define SOBEL_SIZE 3

// Function prototypes
void launch_grayscale(unsigned char *d_input, unsigned char *d_output,
                      int width, int height);
void launch_gaussian_blur(unsigned char *d_input, unsigned char *d_output,
                          int width, int height);
void launch_sobel(unsigned char *d_input, unsigned char *d_output, int width,
                  int height);

#endif // KERNELS_H

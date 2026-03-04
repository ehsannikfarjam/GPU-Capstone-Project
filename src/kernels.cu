#include "../include/kernels.h"
#include <device_launch_parameters.h>

// ----------------------------------------------------------------------------
// Grayscale Kernel
// ----------------------------------------------------------------------------
__global__ void grayscale_kernel(unsigned char *input, unsigned char *output,
                                 int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int idx = (y * width + x) * 3;
    unsigned char r = input[idx];
    unsigned char g = input[idx + 1];
    unsigned char b = input[idx + 2];

    // Scientific Grayscale (Luma coding)
    output[y * width + x] =
        (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
  }
}

void launch_grayscale(unsigned char *d_input, unsigned char *d_output,
                      int width, int height) {
  dim3 block(TILE_SIZE, TILE_SIZE);
  dim3 grid((width + TILE_SIZE - 1) / TILE_SIZE,
            (height + TILE_SIZE - 1) / TILE_SIZE);
  grayscale_kernel<<<grid, block>>>(d_input, d_output, width, height);
  cudaDeviceSynchronize();
}

// ----------------------------------------------------------------------------
// Gaussian Blur Kernel (Shared Memory Tiling)
// ----------------------------------------------------------------------------
__global__ void gaussian_blur_kernel(unsigned char *input,
                                     unsigned char *output, int width,
                                     int height) {
  __shared__ unsigned char tile[TILE_SIZE + 2 * RADIUS][TILE_SIZE + 2 * RADIUS];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x = blockIdx.x * blockDim.x + tx;
  int y = blockIdx.y * blockDim.y + ty;

  // Load tile with halos
  for (int i = ty; i < TILE_SIZE + 2 * RADIUS; i += TILE_SIZE) {
    for (int j = tx; j < TILE_SIZE + 2 * RADIUS; j += TILE_SIZE) {
      int gx = blockIdx.x * TILE_SIZE + j - RADIUS;
      int gy = blockIdx.y * TILE_SIZE + i - RADIUS;

      if (gx >= 0 && gx < width && gy >= 0 && gy < height) {
        tile[i][j] = input[gy * width + gx];
      } else {
        tile[i][j] = 0;
      }
    }
  }
  __syncthreads();

  if (x < width && y < height) {
    float sum = 0.0f;
    // 3x3 Gaussian Weight Matrix
    const float kernel[3][3] = {{1 / 16.0f, 2 / 16.0f, 1 / 16.0f},
                                {2 / 16.0f, 4 / 16.0f, 2 / 16.0f},
                                {1 / 16.0f, 2 / 16.0f, 1 / 16.0f}};

    for (int i = -1; i <= 1; ++i) {
      for (int j = -1; j <= 1; ++j) {
        sum += tile[ty + RADIUS + i][tx + RADIUS + j] * kernel[i + 1][j + 1];
      }
    }
    output[y * width + x] = (unsigned char)sum;
  }
}

void launch_gaussian_blur(unsigned char *d_input, unsigned char *d_output,
                          int width, int height) {
  dim3 block(TILE_SIZE, TILE_SIZE);
  dim3 grid((width + TILE_SIZE - 1) / TILE_SIZE,
            (height + TILE_SIZE - 1) / TILE_SIZE);
  gaussian_blur_kernel<<<grid, block>>>(d_input, d_output, width, height);
  cudaDeviceSynchronize();
}

// ----------------------------------------------------------------------------
// Sobel Edge Detection Kernel (Shared Memory Tiling)
// ----------------------------------------------------------------------------
__global__ void sobel_kernel(unsigned char *input, unsigned char *output,
                             int width, int height) {
  __shared__ unsigned char tile[TILE_SIZE + 2 * RADIUS][TILE_SIZE + 2 * RADIUS];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x = blockIdx.x * blockDim.x + tx;
  int y = blockIdx.y * blockDim.y + ty;

  // Load tile with halos
  for (int i = ty; i < TILE_SIZE + 2 * RADIUS; i += TILE_SIZE) {
    for (int j = tx; j < TILE_SIZE + 2 * RADIUS; j += TILE_SIZE) {
      int gx = blockIdx.x * TILE_SIZE + j - RADIUS;
      int gy = blockIdx.y * TILE_SIZE + i - RADIUS;

      if (gx >= 0 && gx < width && gy >= 0 && gy < height) {
        tile[i][j] = input[gy * width + gx];
      } else {
        tile[i][j] = 0;
      }
    }
  }
  __syncthreads();

  if (x < width && y < height) {
    int Gx = 0, Gy = 0;

    // Sobel Operators
    const int Sx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    const int Sy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    for (int i = -1; i <= 1; ++i) {
      for (int j = -1; j <= 1; ++j) {
        unsigned char val = tile[ty + RADIUS + i][tx + RADIUS + j];
        Gx += val * Sx[i + 1][j + 1];
        Gy += val * Sy[i + 1][j + 1];
      }
    }

    int magnitude = sqrtf((float)(Gx * Gx + Gy * Gy));
    output[y * width + x] = (unsigned char)(magnitude > 255 ? 255 : magnitude);
  }
}

void launch_sobel(unsigned char *d_input, unsigned char *d_output, int width,
                  int height) {
  dim3 block(TILE_SIZE, TILE_SIZE);
  dim3 grid((width + TILE_SIZE - 1) / TILE_SIZE,
            (height + TILE_SIZE - 1) / TILE_SIZE);
  sobel_kernel<<<grid, block>>>(d_input, d_output, width, height);
  cudaDeviceSynchronize();
}

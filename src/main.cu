#include "../include/kernels.h"
#include "../include/pnm_image.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>


void print_usage(const char *prog_name) {
  std::cout << "Usage: " << prog_name
            << " <input.ppm> <output_prefix> <filter_type>\n"
            << "Filters:\n"
            << "  grayscale - Convert to grayscale\n"
            << "  blur      - Gaussian blur\n"
            << "  sobel     - Sobel edge detection\n"
            << "  all       - Run all filters sequentially\n";
}

int main(int argc, char **argv) {
  if (argc < 4) {
    print_usage(argv[0]);
    return 1;
  }

  std::string input_file = argv[1];
  std::string output_prefix = argv[2];
  std::string filter = argv[3];

  PNMImage img;
  try {
    img.load(input_file);
    std::cout << "[INFO] Loaded image: " << input_file << " (" << img.width
              << "x" << img.height << ")\n";
  } catch (const std::exception &e) {
    std::cerr << "[ERROR] " << e.what() << "\n";
    return 1;
  }

  // Allocate Device Memory
  unsigned char *d_input, *d_gray, *d_blur, *d_sobel;
  size_t color_size = img.width * img.height * 3;
  size_t gray_size = img.width * img.height;

  cudaMalloc(&d_input, color_size);
  cudaMalloc(&d_gray, gray_size);
  cudaMalloc(&d_blur, gray_size);
  cudaMalloc(&d_sobel, gray_size);

  cudaMemcpy(d_input, img.pixels.data(), color_size, cudaMemcpyHostToDevice);

  auto start = std::chrono::high_resolution_clock::now();

  // Pipeline Orchestration
  if (filter == "grayscale" || filter == "all") {
    std::cout << "[INFO] Running Grayscale filter...\n";
    launch_grayscale(d_input, d_gray, img.width, img.height);
  }

  if (filter == "blur" || filter == "all") {
    std::cout << "[INFO] Running Gaussian Blur filter...\n";
    // If running only blur, we need grayscale first as input
    if (filter == "blur")
      launch_grayscale(d_input, d_gray, img.width, img.height);
    launch_gaussian_blur(d_gray, d_blur, img.width, img.height);
  }

  if (filter == "sobel" || filter == "all") {
    std::cout << "[INFO] Running Sobel Edge Detection...\n";
    // Ensure we have blurred input for sobel for better results
    if (filter == "sobel") {
      launch_grayscale(d_input, d_gray, img.width, img.height);
      launch_gaussian_blur(d_gray, d_blur, img.width, img.height);
    }
    launch_sobel(d_blur, d_sobel, img.width, img.height);
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;

  std::cout << "[RESULT] GPU Processing took: " << std::fixed
            << std::setprecision(3) << duration.count() << " ms\n";

  // Copy results back and save
  PNMImage out_img;
  out_img.width = img.width;
  out_img.height = img.height;
  out_img.isColor = false;
  out_img.data.resize(gray_size);

  if (filter == "grayscale" || filter == "all") {
    cudaMemcpy(out_img.data.data(), d_gray, gray_size, cudaMemcpyDeviceToHost);
    out_img.save(output_prefix + "_gray.pgm");
  }
  if (filter == "blur" || filter == "all") {
    cudaMemcpy(out_img.data.data(), d_blur, gray_size, cudaMemcpyDeviceToHost);
    out_img.save(output_prefix + "_blur.pgm");
  }
  if (filter == "sobel" || filter == "all") {
    cudaMemcpy(out_img.data.data(), d_sobel, gray_size, cudaMemcpyDeviceToHost);
    out_img.save(output_prefix + "_sobel.pgm");
  }

  // Cleanup
  cudaFree(d_input);
  cudaFree(d_gray);
  cudaFree(d_blur);
  cudaFree(d_sobel);

  std::cout << "[INFO] Cleanup complete. Results saved with prefix: "
            << output_prefix << "\n";

  return 0;
}

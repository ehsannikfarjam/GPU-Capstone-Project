# GPU Image Processing Suite (CUDA)

## Overview
This project is a high-performance image processing suite implemented in C++/CUDA. It provides a pipeline for transforming images through grayscale conversion, Gaussian blurring, and Sobel edge detection. The implementation focuses on scientific accuracy and utilizes advanced GPU programming techniques such as **Shared Memory Tiling** to optimize performance.

## Features
- **Grayscale Conversion**: Uses the Luma formula (0.299R + 0.587G + 0.114B) for scientifically accurate luminance transformation.
- **Gaussian Blur**: Implements a 3x3 Gaussian kernel using shared memory tiling to reduce global memory bandwidth bottlenecks.
- **Sobel Edge Detection**: Utilizes shared memory to calculate horizontal and vertical gradients, producing a high-contrast edge map.
- **Performance Profiling**: Built-in microsecond-accurate timing for GPU operations.
- **CLI Interface**: Flexible command-line tool to process images with specific filters or the entire pipeline.

## Project Structure
- `src/`: CUDA source files (`main.cu`, `kernels.cu`).
- `include/`: Header files (`kernels.h`, `pnm_image.h`).
- `data/`: Input and output images (PPM/PGM format).
- `bin/`: Compiled binaries.
- `Makefile`: Build script for the project.
- `run.ps1`: Automation script for building and execution on Windows.

## Installation & Building
### Prerequisites
- NVIDIA GPU (Compute Capability 5.0 or higher recommended).
- CUDA Toolkit installed.
- C++ Compiler (MSVC on Windows).

### Build Instructions
Run the following in PowerShell:
```powershell
./run.ps1
```
Or use the Makefile:
```cmd
make all
```

## Usage
The binary accepts three arguments:
```cmd
./bin/image_processor.exe <input.ppm> <output_prefix> <filter_type>
```
**Filter Types:**
- `grayscale`: Only grayscale conversion.
- `blur`: Grayscale + Gaussian Blur.
- `sobel`: Grayscale + Blur + Sobel Edge Detection.
- `all`: Runs all filters and saves intermediate results.

## Design Decisions
1. **Shared Memory Tiling**: To efficiently handle local filter operations (Blur, Sobel), threads load a tile of pixels plus a halo region into shared memory. This significantly reduces redundant global memory reads for overlapping filter windows.
2. **PNM Format**: Used the Netpbm format (P5/P6) for its simplicity and transparency, ensuring the project remains self-contained without needing heavy external image libraries like OpenCV.
3. **Google C++ Style**: The code follows the Google C++ Style Guide for readability and maintainability.

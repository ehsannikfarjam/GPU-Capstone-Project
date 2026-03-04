#!/bin/bash

# GPU Image Processor - Linux Execution Script (Coursera IDE)
set -e

echo "--- GPU Capstone Project: High-Performance Image Processing Suite ---"

# 1. Build the project
echo "[1/3] Building project for Linux..."
make clean
make build

# 2. Run the pipeline
echo "[2/3] Running GPU Image Processing Pipeline..."
INPUT_IMG="data/test_pattern.ppm"
OUTPUT_PREFIX="data/result"

# Ensure output directory exists
mkdir -p data

./bin/image_processor $INPUT_IMG $OUTPUT_PREFIX all | tee execution_log_linux.txt

# 3. Final Verification
echo "[3/3] Verification Summary"
if [ -f "data/result_sobel.pgm" ]; then
    echo "SUCCESS: Pipeline completed successfully. Output images generated in 'data/' folder."
else
    echo "FAILURE: Output files were not generated."
    exit 1
fi

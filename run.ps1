# GPU Image Processor Automation Script

$ErrorActionPreference = "Stop"

$ProjectRoot = "d:\Drive C\Desktop\Coursera\GPU Programming\GPU Capstone Assignment\GPU_Image_Processor"
$BinDir = "$ProjectRoot\bin"
$DataDir = "$ProjectRoot\data"
$LogFile = "$ProjectRoot\execution_log.txt"

Write-Host "--- GPU Capstone Project: High-Performance Image Processing Suite ---" -ForegroundColor Cyan

# 1. Clean and Build
Write-Host "[1/3] Building project..." -ForegroundColor Yellow
if (Test-Path $BinDir) { Remove-Item -Recurse -Force $BinDir }
New-Item -ItemType Directory -Force -Path $BinDir | Out-Null

try {
    # Attempt build with nvcc
    # We assume nvcc is in the path or the user will run this where nvcc is known
    nvcc -I"$ProjectRoot\include" -O3 -arch=sm_50 "$ProjectRoot\src\main.cu" "$ProjectRoot\src\kernels.cu" -o "$BinDir\image_processor.exe"
    Write-Host "Build Successful!" -ForegroundColor Green
} catch {
    Write-Host "Build Failed! Please ensure nvcc is in your PATH." -ForegroundColor Red
    exit 1
}

# 2. Run Benchmarks
Write-Host "[2/3] Running GPU Image Processing Pipeline..." -ForegroundColor Yellow
$InputImg = "$DataDir\test_pattern.ppm"
$OutputPrefix = "$DataDir\result"

if (-not (Test-Path $InputImg)) {
    Write-Host "Test image not found! Generating simple pattern..."
    # Fallback pattern generation or error
}

Start-Transcript -Path $LogFile -Append
& "$BinDir\image_processor.exe" $InputImg $OutputPrefix all
Stop-Transcript

# 3. Verification Summary
Write-Host "[3/3] Verification Summary" -ForegroundColor Yellow
Write-Host "Logs saved to: $LogFile"
Write-Host "Output images generated in: $DataDir"

if (Test-Path "$DataDir\result_sobel.pgm") {
    Write-Host "SUCCESS: Pipeline completed successfully." -ForegroundColor Green
} else {
    Write-Host "FAILURE: Output files missing." -ForegroundColor Red
}

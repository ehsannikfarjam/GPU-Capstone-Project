NVCC = nvcc
NVCC_FLAGS = -I./include -O3 -arch=sm_50 -Wno-deprecated-gpu-targets
SRC_DIR = src
BIN_DIR = bin
TARGET = $(BIN_DIR)/image_processor

all: build

build:
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) src/main.cu src/kernels.cu -o $(TARGET)

clean:
	rm -rf $(BIN_DIR) *.o *.obj

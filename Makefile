CC=cc
NVCC=nvcc
CUDA_HOME = $(shell dirname $(shell dirname $(shell which nvcc)))
CUDA_LIB = $(CUDA_HOME)/lib64
# INC_PATH = $(CUDA_HOME)/include

# $(info $(CUDA_HOME) $(INC_PATH))
INC_DIR = $(abspath ./include)
INC_PATH += $(INC_DIR) 
SRCS = $(shell find $(abspath ./) -name "*.cu")
BUILD_DIR = ./build
$(shell mkdir -p $(BUILD_DIR))

BINARY = $(BUILD_DIR)/sgemm
default: $(BINARY)

all: default

$(BINARY): $(SRCS)
	$(NVCC) -I $(INC_PATH) $(SRCS) -o $(abspath $@)

run:$(BINARY)
	@$^

clean:
	rm -rf $(BUILD_DIR)

.PHONY:run clean all default

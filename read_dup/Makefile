PROGRAM := readdup
SOURCES := read_dup.cu

# Another typical location is: /usr/local/cuda/bin/nvcc
NVCC    ?= /opt/cuda/bin/nvcc
CCBIN   =

# You may want this, if CUDA complains that your gcc version is too new.
# Otherwise, it may be commented out. For example, Fedora will probably
# require this CCBIN override, while RHEL/CentOS will probably not.
CCBIN = -ccbin /usr/local/gcc-5/bin/g++

NVCC_FLAGS := -G -Xcompiler -Wall -g -O1

$(PROGRAM): $(SOURCES)
	$(NVCC) $(CCBIN) $(NVCC_FLAGS) -I .  $^ -o $@

clean:
	-rm $(PROGRAM)

run: $(PROGRAM)
	./$< 0 1

print-%:
	@echo "$*=$($*)"

.PHONY: clean run

# ---------
CXX := g++
STD := c++17

# ---------
TARGET := matmat
SRCS   := main.cpp matmat.cpp misc.cpp bench.cpp autotune.cpp
OBJS   := $(SRCS:.cpp=.o)
DEPS   := $(OBJS:.o=.d)

# ---------
WARN := -Wall -Wextra

# SIMD ISA required by the hand-written kernels:
#   simd256 kernel (4x4)  -> AVX2 + FMA   (__m256d, _mm256_fmadd_pd)
#   simd512 kernel (8x16) -> AVX-512F     (__m512d, _mm512_fmadd_pd)
SIMD := -mavx2 -mfma -mavx512f

OPT := -O3 -march=native
ifdef DEBUG
	OPT := -O0 -g
endif

CXXFLAGS := -std=$(STD) $(WARN) $(OPT) $(SIMD) -fopenmp
LDFALGS  := -fopenmp
LDLIBS   :=

# Optional: BLAS baseline
ifdef BLAS
	CXXFLAGS += -DUSE_BLAS
	LDLIBS   += -lopenblas
endif

# ------------- rules ------------------
.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(LDFALGS) $(OBJS) -o $@ $(LDLIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(OBJS) $(DEPS) $(TARGET)

-include $(DEPS)
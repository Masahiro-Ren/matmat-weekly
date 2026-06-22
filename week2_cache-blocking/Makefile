# ---------
CXX := g++
STD := c++17

# ---------
TARGET := matmat
SRCS   := main.cpp matmat.cpp misc.cpp
OBJS   := $(SRCS:.cpp=.o)
DEPS   := $(OBJS:.o=.d)

# ---------
WARN := -Wall -Wextra

OPT := -O3 -march=native
ifdef DEBUG
	OPT := -O0 -g
endif

CXXFLAGS := -std=$(STD) $(WARN) $(OPT) -fopenmp
LDFALGS  := -fopenmp
LDLIBS   :=

# Optional: BLAS baseline
ifdef BLAS
	CXXFLAGS += -DUSE_BLAS
	LDLIBS   += -lopenblas
endif

#Optional: disable vectorization
ifdef NOVEC
	CXXFLAGS += -fno-tree-vectorize -fno-tree-slp-vectorize
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
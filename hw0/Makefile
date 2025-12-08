PYTHON ?= python

# 提取 include 与版本
PY_INCLUDE := $(shell $(PYTHON) -c "import sysconfig;print(sysconfig.get_paths()['include'])")
PY_PYBIND_INCLUDE := $(shell $(PYTHON) -m pybind11 --includes)

# 尝试获取 LIBDIR 和 LIBRARY，空则 fallback
PY_LIBDIR_RAW := $(shell $(PYTHON) -c "import sysconfig;print(sysconfig.get_config_var('LIBDIR'))")
PY_LIBRARY_RAW := $(shell $(PYTHON) -c "import sysconfig;print(sysconfig.get_config_var('LIBRARY'))")

# Fallback logic
ifeq ($(PY_LIBDIR_RAW),None)
  PY_LIBDIR := $(shell $(PYTHON) -c "import sys,os;print(os.path.join(sys.prefix,'libs'))")
else
  PY_LIBDIR := $(PY_LIBDIR_RAW)
endif

ifeq ($(PY_LIBRARY_RAW),None)
  PY_LIBRARY := $(shell $(PYTHON) -c "import sys;print(f'python{sys.version_info.major}{sys.version_info.minor}.lib')")
else
  PY_LIBRARY := $(PY_LIBRARY_RAW)
endif

# 提取不带扩展的名字用于 -l
PY_LIB_BARE := $(basename $(PY_LIBRARY))

CXX ?= g++
CXXFLAGS := -O3 -Wall -std=c++17 -fPIC -I$(PY_INCLUDE) $(PY_PYBIND_INCLUDE)
LDFLAGS := -shared -L$(PY_LIBDIR) -l$(PY_LIB_BARE)

SRC := src/simple_ml_ext.cpp
TARGET := src/simple_ml_ext.pyd

default: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	$(RM) $(TARGET)
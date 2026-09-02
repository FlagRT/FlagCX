# makefiles/platforms/ascend.mk
# Huawei Ascend platform configuration.

DEVICE_HOME  ?= /usr/local/Ascend/ascend-toolkit/latest
# DEVICE_LIB 保持单目录（Makefile 链接规则 `-L$(DEVICE_LIB)` 只支持单目录，
# 多目录展开会把后续目录当输入文件）；额外 -L 放 DEVICE_LINK。
DEVICE_LIB   := $(DEVICE_HOME)/lib64
DEVICE_INCLUDE := $(DEVICE_HOME)/include \
                  $(DEVICE_HOME)/opp/vendors/customize/op_api/include
# -lopapi/-lcust_opapi: #7.2 DAG 持久 kernel 的 aclnn 接口（依赖已安装的
# custom_opp_*.run 包）。未安装时编译链接失败——#8 运行时探测/回退再处理。
DEVICE_LINK  := -lascendcl -L$(DEVICE_HOME)/aarch64-linux/lib64 -lopapi -lnnopbase \
                -L$(DEVICE_HOME)/opp/vendors/customize/op_api/lib -lcust_opapi
DEVICE_PLATFORM := CANN
DEVICE_COMPILER :=
DEVICE_COMPILE_FLAG :=
DEVICE_LINK_FLAG :=
DEVICE_FILE_EXTENSION :=

CCL_HOME    ?= /usr/local/Ascend/ascend-toolkit/latest
CCL_LIB     := $(CCL_HOME)/lib64
CCL_INCLUDE := $(CCL_HOME)/include
CCL_LINK    := -lhccl
# FLAGCX_USE_ASCEND_DAG: #7.2 昇腾 DAG 持久 kernel launch 开关（uniRunner 侧
# 经 deviceAdaptor->launchCollectiveKernel 走 aclnn；昇腾侧不定义
# COMPILE_KERNEL_HOST，故必须用独立宏）。
ADAPTOR_FLAG := -DUSE_ASCEND_ADAPTOR -DFLAGCX_COMM_TRAITS_DEFAULT -march=armv8.2-a -std=c++17 \
                -DFLAGCX_USE_ASCEND_DAG

PLATFORM_KERNEL_DIR  :=
PLATFORM_KERNEL_SRCS :=
PLATFORM_EXTRA_SRCS  := flagcx/adaptor/device_api/default_dev_api_backend.cc

# makefiles/platforms/ascend.mk
# Huawei Ascend platform configuration.

DEVICE_HOME  ?= /usr/local/Ascend/ascend-toolkit/latest

# FLAGCX_ASCEND_DAG: 昇腾 DAG 持久 kernel（#7.2）的构建开关，**默认关闭**。
# 关闭时：不定义 -DFLAGCX_USE_ASCEND_DAG，也不链接该路径独有的自定义算子包
# （-L…/opp/vendors/customize/op_api/lib -lcust_opapi，来自已安装的
#  custom_opp_*.run）。c10d 与完成语义路径都不经 DAG，不受影响。
# ⚠️ -lopapi/-lnnopbase 是【标准 CANN】库，必须始终链接：设备侧 reduce
# （cannAdaptorReduceSum）所用的 aclnnInplaceAddGetWorkspaceSize 就在
# libopapi.so 里。只砍 -lcust_opapi 而保留这两个；砍多了会在 import 期得到
# `undefined symbol: aclnnInplaceAddGetWorkspaceSize`。
# 打开方式：`make FLAGCX_ASCEND_DAG=1 …`，或构建前 `export FLAGCX_ASCEND_DAG=1`
# （pip install 场景同样按环境变量生效）。
FLAGCX_ASCEND_DAG ?= 0

# DEVICE_LIB 保持单目录（Makefile 链接规则 `-L$(DEVICE_LIB)` 只支持单目录，
# 多目录展开会把后续目录当输入文件）；额外 -L 放 DEVICE_LINK。
DEVICE_LIB   := $(DEVICE_HOME)/lib64
DEVICE_INCLUDE := $(DEVICE_HOME)/include
DEVICE_LINK  := -lascendcl -L$(DEVICE_HOME)/aarch64-linux/lib64 -lopapi -lnnopbase
ifeq ($(FLAGCX_ASCEND_DAG),1)
DEVICE_INCLUDE += $(DEVICE_HOME)/opp/vendors/customize/op_api/include
DEVICE_LINK    += -L$(DEVICE_HOME)/opp/vendors/customize/op_api/lib -lcust_opapi
endif
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
# COMPILE_KERNEL_HOST，故必须用独立宏）。默认关闭，见文件头 FLAGCX_ASCEND_DAG。
ADAPTOR_FLAG := -DUSE_ASCEND_ADAPTOR -DFLAGCX_COMM_TRAITS_DEFAULT -march=armv8.2-a -std=c++17
ifeq ($(FLAGCX_ASCEND_DAG),1)
ADAPTOR_FLAG += -DFLAGCX_USE_ASCEND_DAG
endif

PLATFORM_KERNEL_DIR  :=
PLATFORM_KERNEL_SRCS :=
PLATFORM_EXTRA_SRCS  := flagcx/adaptor/device_api/default_dev_api_backend.cc

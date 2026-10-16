#ifdef USE_ASCEND_ADAPTOR
#include "acl/acl.h"
#include "flagcx.h"
#include "flagcx_errors.h"
#include "hccl/hccl.h"
#include <map>
struct flagcxInnerDevComm {};

struct flagcxInnerComm {
  HcclComm base;
};

struct flagcxStream {
  aclrtStream base;
};

struct flagcxEvent {
  aclrtEvent base;
};

struct flagcxIpcMemHandle {
  char *base; // to be implemented
};

#define DEVCHECK(func)                                                         \
  {                                                                            \
    int ret = func;                                                            \
    if (ret != ACL_SUCCESS) {                                                  \
      const char *errmsg = aclGetRecentErrMsg();                               \
      flagcx::setLastError(flagcxUnhandledDeviceError, (int)ret, __FILE__,     \
                           __LINE__,                                           \
                           "device call failed: %s (aclError %d: %s)", #func,  \
                           (int)ret, errmsg ? errmsg : "");                    \
      return flagcxUnhandledDeviceError;                                       \
    }                                                                          \
  }
#endif // USE_ASCEND_ADAPTOR

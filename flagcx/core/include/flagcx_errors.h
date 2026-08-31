#pragma once
#include "flagcx.h"
#include <cstdarg>
#include <cstdio>
#include <cstring>

// O3: 线程局部最近一次错误记录（flagcxGetLastError 的数据源）。
// header-only（C++17 inline thread_local，多 TU 共享单实例），无需新 .cc。
// DEVCHECK / FLAGCXCHECK 在失败时调用 flagcx::setLastError 写入。

#define FLAGCX_LAST_ERROR_MSG_LEN 512

namespace flagcx {

struct LastErrorRecord {
  int code = flagcxSuccess;
  int vendorCode = 0;
  char msg[FLAGCX_LAST_ERROR_MSG_LEN];
  LastErrorRecord() { msg[0] = '\0'; }
};

inline thread_local LastErrorRecord tlsLastError;
inline thread_local bool tlsErrorSet = false;

inline flagcxResult_t setLastError(int code, int vendorCode,
                                   const char *file, int line,
                                   const char *fmt, ...) {
  tlsLastError.code = code;
  tlsLastError.vendorCode = vendorCode;
  snprintf(tlsLastError.msg, sizeof(tlsLastError.msg), "%s:%d ", file, line);
  size_t off = strlen(tlsLastError.msg);
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(tlsLastError.msg + off, sizeof(tlsLastError.msg) - off, fmt, ap);
  va_end(ap);
  tlsErrorSet = true;
  return (flagcxResult_t)code;
}

inline const char *getLastErrorMessage() {
  return tlsErrorSet ? tlsLastError.msg : nullptr;
}

inline int getLastErrorCode() {
  return tlsErrorSet ? tlsLastError.code : flagcxSuccess;
}

inline int getLastVendorCode() {
  return tlsErrorSet ? tlsLastError.vendorCode : 0;
}

}  // namespace flagcx

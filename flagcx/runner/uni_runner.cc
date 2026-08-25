/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 ************************************************************************/

#include "flagcx_hetero.h"
#include "proxy.h"
#include "runner.h"
#include "uni_runner_impl.h"

FLAGCX_PARAM(UniRunnerUseLocRed, "UNIRUNNER_USE_LOCRED", 0);
FLAGCX_PARAM(UniRunnerUseRingAG, "UNIRUNNER_USE_RINGAG", 0);
FLAGCX_PARAM(UniRunnerUseSlicedAR, "UNIRUNNER_USE_SLICEDAR", 0);

flagcxResult_t uniRunnerReduce(const void *sendbuff, void *recvbuff,
                               size_t count, flagcxDataType_t datatype,
                               flagcxRedOp_t op, int root, flagcxComm_t comm,
                               flagcxStream_t stream) {
  flagcxResult_t res = flagcxSuccess;
  flagcxHeteroComm_t hcomm = comm->heteroComm;
  flagcxUniRunnerState *runnerState = &hcomm->proxyState->uniRunnerState;
  void *scratchbuff = nullptr;
  FLAGCXCHECK(deviceAdaptor->deviceMalloc(
      &scratchbuff, 2 * count * getFlagcxDataTypeSize(datatype),
      flagcxMemDevice, stream));
  FLAGCXCHECKGOTO(initUniRunner(comm, stream), res, out);
  FLAGCXCHECKGOTO(initUniRunnerStateTreeRed(runnerState, sendbuff, recvbuff,
                                            scratchbuff, count, datatype, op,
                                            root, comm),
                  res, out);
  FLAGCXCHECKGOTO(runUniRunner(comm), res, out);
out:
  FLAGCXCHECK(deviceAdaptor->deviceFree(scratchbuff, flagcxMemDevice, stream));
  FLAGCXCHECK(cleanupUniRunner(comm));
  return res;
}

flagcxResult_t uniRunnerGather(const void *sendbuff, void *recvbuff,
                               size_t count, flagcxDataType_t datatype,
                               int root, flagcxComm_t comm,
                               flagcxStream_t stream) {
  size_t size = count * getFlagcxDataTypeSize(datatype);
  char *buffer = static_cast<char *>(recvbuff);

  FLAGCXCHECK(flagcxHeteroGroupStart());
  if (comm->rank == root) {
    for (int r = 0; r < comm->nranks; r++) {
      FLAGCXCHECK(flagcxHeteroRecv(static_cast<void *>(buffer + r * size),
                                   count, datatype, r, comm->heteroComm,
                                   stream));
    }
  }
  FLAGCXCHECK(flagcxHeteroSend(sendbuff, count, datatype, root,
                               comm->heteroComm, stream));
  FLAGCXCHECK(flagcxHeteroGroupEnd());
  return flagcxSuccess;
}

flagcxResult_t uniRunnerScatter(const void *sendbuff, void *recvbuff,
                                size_t count, flagcxDataType_t datatype,
                                int root, flagcxComm_t comm,
                                flagcxStream_t stream) {
  size_t size = count * getFlagcxDataTypeSize(datatype);
  const char *buffer = static_cast<const char *>(sendbuff);

  FLAGCXCHECK(flagcxHeteroGroupStart());
  if (comm->rank == root) {
    for (int r = 0; r < comm->nranks; r++) {
      FLAGCXCHECK(flagcxHeteroSend(static_cast<const void *>(buffer + r * size),
                                   count, datatype, r, comm->heteroComm,
                                   stream));
    }
  }
  FLAGCXCHECK(flagcxHeteroRecv(recvbuff, count, datatype, root,
                               comm->heteroComm, stream));
  FLAGCXCHECK(flagcxHeteroGroupEnd());
  return flagcxSuccess;
}

flagcxResult_t uniRunnerBroadcast(const void *sendbuff, void *recvbuff,
                                  size_t count, flagcxDataType_t datatype,
                                  int root, flagcxComm_t comm,
                                  flagcxStream_t stream) {
  FLAGCXCHECK(flagcxHeteroGroupStart());
  if (comm->rank == root) {
    for (int r = 0; r < comm->nranks; r++) {
      FLAGCXCHECK(flagcxHeteroSend(sendbuff, count, datatype, r,
                                   comm->heteroComm, stream));
    }
  }
  FLAGCXCHECK(flagcxHeteroRecv(recvbuff, count, datatype, root,
                               comm->heteroComm, stream));
  FLAGCXCHECK(flagcxHeteroGroupEnd());
  return flagcxSuccess;
}

flagcxResult_t uniRunnerAllGather(const void *sendbuff, void *recvbuff,
                                  size_t sendcount, flagcxDataType_t datatype,
                                  flagcxComm_t comm, flagcxStream_t stream);

flagcxResult_t uniRunnerAllReduce(const void *sendbuff, void *recvbuff,
                                  size_t count, flagcxDataType_t datatype,
                                  flagcxRedOp_t op, flagcxComm_t comm,
                                  flagcxStream_t stream) {
  // Kistich(hetero-fallback): the DAG engine (initUniRunner) requires UVA
  // (device access to host fifo via hostGetDevicePointer, NULL on CANN —
  // Ascend has no UVA). Degrade to the plain path: allgather (Send/Recv,
  // verified working) + host reduce + H2D copy. Correctness over speed.
  size_t esize = getFlagcxDataTypeSize(datatype);
  size_t bytes = count * esize;
  int nranks = comm->nranks;

  if (nranks == 1 || count == 0) {
    if (sendbuff != recvbuff && count > 0) {
      FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
          recvbuff, const_cast<void *>(sendbuff), bytes,
          flagcxMemcpyDeviceToDevice, stream, nullptr));
    }
    return flagcxSuccess;
  }

  // 1) gather all ranks into a temporary device buffer (nranks slices)
  void *tmpDev = nullptr;
  FLAGCXCHECK(deviceAdaptor->deviceMalloc(&tmpDev, bytes * nranks,
                                          flagcxMemDevice, stream));
  FLAGCXCHECK(
      uniRunnerAllGather(sendbuff, tmpDev, count, datatype, comm, stream));

  // 2) D2H the gathered buffer, then reduce on host
  char *hostBuf = (char *)malloc(bytes * nranks);
  if (hostBuf == nullptr)
    return flagcxSystemError;
  FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
      hostBuf, tmpDev, bytes * nranks, flagcxMemcpyDeviceToHost, stream,
      nullptr));
  FLAGCXCHECK(deviceAdaptor->streamSynchronize(stream));

#define FLAGCX_HOST_REDUCE(T, OPNAME, EXPR)                                   \
  do {                                                                        \
    const T *src = (const T *)(hostBuf + bytes);                              \
    T *dst = (T *)hostBuf;                                                    \
    for (size_t i = 0; i < count; i++) {                                      \
      T a = dst[i], b = src[i];                                               \
      (void)a;                                                                \
      (void)b;                                                                \
      dst[i] = (EXPR);                                                        \
    }                                                                         \
  } while (0)

  for (int r = 1; r < nranks; r++) {
    char *slice = hostBuf + r * bytes;
    switch (datatype) {
    case flagcxFloat32: {
      float *dst = (float *)hostBuf;
      const float *src = (const float *)slice;
      for (size_t i = 0; i < count; i++) {
        if (op == flagcxSum) dst[i] += src[i];
        else if (op == flagcxProd) dst[i] *= src[i];
        else if (op == flagcxMax) dst[i] = dst[i] > src[i] ? dst[i] : src[i];
        else if (op == flagcxMin) dst[i] = dst[i] < src[i] ? dst[i] : src[i];
      }
      break;
    }
    case flagcxFloat64: {
      double *dst = (double *)hostBuf;
      const double *src = (const double *)slice;
      for (size_t i = 0; i < count; i++) {
        if (op == flagcxSum) dst[i] += src[i];
        else if (op == flagcxProd) dst[i] *= src[i];
        else if (op == flagcxMax) dst[i] = dst[i] > src[i] ? dst[i] : src[i];
        else if (op == flagcxMin) dst[i] = dst[i] < src[i] ? dst[i] : src[i];
      }
      break;
    }
    case flagcxBfloat16: {
      uint16_t *dst = (uint16_t *)hostBuf;
      const uint16_t *src = (const uint16_t *)slice;
      auto bf2f = [](uint16_t v) -> float {
        uint32_t u = ((uint32_t)(v & 0x8000)) << 16 |
                     ((uint32_t)(v & 0x7fff)) << 16;
        return *((float *)&u);
      };
      auto f2bf = [](float f) -> uint16_t {
        uint32_t u = *((uint32_t *)&f);
        uint32_t lsb = (u >> 16) & 1;
        uint16_t r = (uint16_t)((u >> 16) + lsb);
        return r;
      };
      for (size_t i = 0; i < count; i++) {
        float a = bf2f(dst[i]), b = bf2f(src[i]);
        float v;
        if (op == flagcxSum) v = a + b;
        else if (op == flagcxProd) v = a * b;
        else if (op == flagcxMax) v = a > b ? a : b;
        else if (op == flagcxMin) v = a < b ? a : b;
        else v = a;
        dst[i] = f2bf(v);
      }
      break;
    }
    case flagcxFloat16: {
      uint16_t *dst = (uint16_t *)hostBuf;
      const uint16_t *src = (const uint16_t *)slice;
      auto h2f = [](uint16_t h) -> float {
        uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 0x1f, frac = h & 0x3ff;
        uint32_t u;
        if (exp == 0)
          u = (sign << 31) | (frac << 13);
        else if (exp == 31)
          u = (sign << 31) | 0x7f800000 | (frac << 13);
        else
          u = (sign << 31) | ((exp - 15 + 127) << 23) | (frac << 13);
        return *((float *)&u);
      };
      auto f2h = [](float f) -> uint16_t {
        uint32_t u = *((uint32_t *)&f);
        uint32_t sign = (u >> 31) & 1, exp = (u >> 23) & 0xff, frac = u & 0x7fffff;
        uint16_t h;
        if (exp == 0xff)
          h = (uint16_t)((sign << 15) | 0x7c00 | (frac ? 1 : 0));
        else {
          int32_t e = (int32_t)exp - 127 + 15;
          if (e <= 0)
            h = (uint16_t)(sign << 15);
          else if (e >= 31)
            h = (uint16_t)((sign << 15) | 0x7c00);
          else
            h = (uint16_t)((sign << 15) | (e << 10) | (frac >> 13));
        }
        return h;
      };
      for (size_t i = 0; i < count; i++) {
        float a = h2f(dst[i]), b = h2f(src[i]);
        float v;
        if (op == flagcxSum) v = a + b;
        else if (op == flagcxProd) v = a * b;
        else if (op == flagcxMax) v = a > b ? a : b;
        else if (op == flagcxMin) v = a < b ? a : b;
        else v = a;
        dst[i] = f2h(v);
      }
      break;
    }
    case flagcxInt64: {
      int64_t *dst = (int64_t *)hostBuf;
      const int64_t *src = (const int64_t *)slice;
      for (size_t i = 0; i < count; i++) {
        if (op == flagcxSum) dst[i] += src[i];
        else if (op == flagcxProd) dst[i] *= src[i];
        else if (op == flagcxMax) dst[i] = dst[i] > src[i] ? dst[i] : src[i];
        else if (op == flagcxMin) dst[i] = dst[i] < src[i] ? dst[i] : src[i];
      }
      break;
    }
    case flagcxUint64: {
      uint64_t *dst = (uint64_t *)hostBuf;
      const uint64_t *src = (const uint64_t *)slice;
      for (size_t i = 0; i < count; i++) {
        if (op == flagcxSum) dst[i] += src[i];
        else if (op == flagcxProd) dst[i] *= src[i];
        else if (op == flagcxMax) dst[i] = dst[i] > src[i] ? dst[i] : src[i];
        else if (op == flagcxMin) dst[i] = dst[i] < src[i] ? dst[i] : src[i];
      }
      break;
    }
    case flagcxInt32: {
      int32_t *dst = (int32_t *)hostBuf;
      const int32_t *src = (const int32_t *)slice;
      for (size_t i = 0; i < count; i++) {
        if (op == flagcxSum) dst[i] += src[i];
        else if (op == flagcxProd) dst[i] *= src[i];
        else if (op == flagcxMax) dst[i] = dst[i] > src[i] ? dst[i] : src[i];
        else if (op == flagcxMin) dst[i] = dst[i] < src[i] ? dst[i] : src[i];
      }
      break;
    }
    case flagcxUint32: {
      uint32_t *dst = (uint32_t *)hostBuf;
      const uint32_t *src = (const uint32_t *)slice;
      for (size_t i = 0; i < count; i++) {
        if (op == flagcxSum) dst[i] += src[i];
        else if (op == flagcxProd) dst[i] *= src[i];
        else if (op == flagcxMax) dst[i] = dst[i] > src[i] ? dst[i] : src[i];
        else if (op == flagcxMin) dst[i] = dst[i] < src[i] ? dst[i] : src[i];
      }
      break;
    }
    case flagcxInt8: {
      int8_t *dst = (int8_t *)hostBuf;
      const int8_t *src = (const int8_t *)slice;
      for (size_t i = 0; i < count; i++) {
        if (op == flagcxSum) dst[i] += src[i];
        else if (op == flagcxMax) dst[i] = dst[i] > src[i] ? dst[i] : src[i];
        else if (op == flagcxMin) dst[i] = dst[i] < src[i] ? dst[i] : src[i];
      }
      break;
    }
    case flagcxUint8: {
      uint8_t *dst = (uint8_t *)hostBuf;
      const uint8_t *src = (const uint8_t *)slice;
      for (size_t i = 0; i < count; i++) {
        if (op == flagcxSum) dst[i] += src[i];
        else if (op == flagcxMax) dst[i] = dst[i] > src[i] ? dst[i] : src[i];
        else if (op == flagcxMin) dst[i] = dst[i] < src[i] ? dst[i] : src[i];
      }
      break;
    }
    default:
      free(hostBuf);
      return flagcxInvalidArgument;
    }
  }
#undef FLAGCX_HOST_REDUCE

  // 3) H2D the reduced slice
  FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
      recvbuff, hostBuf, bytes, flagcxMemcpyHostToDevice, stream, nullptr));
  free(hostBuf);
  FLAGCXCHECK(deviceAdaptor->deviceFree(tmpDev, flagcxMemDevice, stream));
  return flagcxSuccess;
}

flagcxResult_t uniRunnerReduceScatter(const void *sendbuff, void *recvbuff,
                                      size_t recvcount,
                                      flagcxDataType_t datatype,
                                      flagcxRedOp_t op, flagcxComm_t comm,
                                      flagcxStream_t stream) {
  flagcxResult_t res = flagcxSuccess;
  flagcxHeteroComm_t hcomm = comm->heteroComm;
  flagcxUniRunnerState *runnerState = &hcomm->proxyState->uniRunnerState;
  void *scratchbuff = nullptr;
  FLAGCXCHECK(deviceAdaptor->deviceMalloc(
      &scratchbuff, recvcount * comm->nranks * getFlagcxDataTypeSize(datatype),
      flagcxMemDevice, stream));
  FLAGCXCHECKGOTO(initUniRunner(comm, stream), res, out);
  FLAGCXCHECKGOTO(initUniRunnerStateRingRS(runnerState, sendbuff, recvbuff,
                                           scratchbuff, recvcount, datatype, op,
                                           comm),
                  res, out);
  FLAGCXCHECKGOTO(runUniRunner(comm), res, out);
out:
  FLAGCXCHECK(deviceAdaptor->deviceFree(scratchbuff, flagcxMemDevice, stream));
  FLAGCXCHECK(cleanupUniRunner(comm));
  return res;
}

flagcxResult_t uniRunnerAllGather(const void *sendbuff, void *recvbuff,
                                  size_t sendcount, flagcxDataType_t datatype,
                                  flagcxComm_t comm, flagcxStream_t stream) {
  size_t size = sendcount * getFlagcxDataTypeSize(datatype);
  char *bufferOut = static_cast<char *>(recvbuff);
  FLAGCXCHECK(flagcxHeteroGroupStart());
  for (int r = 0; r < comm->nranks; r++) {
    FLAGCXCHECK(flagcxHeteroSend(sendbuff, sendcount, datatype, r,
                                 comm->heteroComm, stream));
    FLAGCXCHECK(flagcxHeteroRecv(static_cast<void *>(bufferOut + r * size),
                                 sendcount, datatype, r, comm->heteroComm,
                                 stream));
  }
  FLAGCXCHECK(flagcxHeteroGroupEnd());
  return flagcxSuccess;
}

flagcxResult_t uniRunnerAlltoAll(const void *sendbuff, void *recvbuff,
                                 size_t count, flagcxDataType_t datatype,
                                 flagcxComm_t comm, flagcxStream_t stream) {
  size_t size = count * getFlagcxDataTypeSize(datatype);
  const char *bufferIn = static_cast<const char *>(sendbuff);
  char *bufferOut = static_cast<char *>(recvbuff);
  FLAGCXCHECK(flagcxHeteroGroupStart());
  for (int r = 0; r < comm->nranks; r++) {
    FLAGCXCHECK(flagcxHeteroSend(static_cast<const void *>(bufferIn + r * size),
                                 count, datatype, r, comm->heteroComm, stream));
    FLAGCXCHECK(flagcxHeteroRecv(static_cast<void *>(bufferOut + r * size),
                                 count, datatype, r, comm->heteroComm, stream));
  }
  FLAGCXCHECK(flagcxHeteroGroupEnd());
  return flagcxSuccess;
}

flagcxResult_t uniRunnerAlltoAllv(const void *sendbuff, size_t *sendcounts,
                                  size_t *sdispls, void *recvbuff,
                                  size_t *recvcounts, size_t *rdispls,
                                  flagcxDataType_t datatype, flagcxComm_t comm,
                                  flagcxStream_t stream) {
  size_t size = getFlagcxDataTypeSize(datatype);
  const char *bufferIn = static_cast<const char *>(sendbuff);
  char *bufferOut = static_cast<char *>(recvbuff);
  FLAGCXCHECK(flagcxHeteroGroupStart());
  for (int r = 0; r < comm->nranks; r++) {
    if (flagcxCCLAdaptorNeedSendrecv(sendcounts[r])) {
      FLAGCXCHECK(flagcxHeteroSend(
          static_cast<const void *>(bufferIn + sdispls[r] * size),
          sendcounts[r], datatype, r, comm->heteroComm, stream));
    }
    if (flagcxCCLAdaptorNeedSendrecv(recvcounts[r])) {
      FLAGCXCHECK(flagcxHeteroRecv(
          static_cast<void *>(bufferOut + rdispls[r] * size), recvcounts[r],
          datatype, r, comm->heteroComm, stream));
    }
  }
  FLAGCXCHECK(flagcxHeteroGroupEnd());
  return flagcxSuccess;
}

flagcxResult_t uniRunnerSend(const void *sendbuff, size_t count,
                             flagcxDataType_t datatype, int peer,
                             flagcxComm_t comm, flagcxStream_t stream) {
  FLAGCXCHECK(flagcxHeteroSend(sendbuff, count, datatype, peer,
                               comm->heteroComm, stream));
  return flagcxSuccess;
}

flagcxResult_t uniRunnerRecv(void *recvbuff, size_t count,
                             flagcxDataType_t datatype, int peer,
                             flagcxComm_t comm, flagcxStream_t stream) {
  FLAGCXCHECK(flagcxHeteroRecv(recvbuff, count, datatype, peer,
                               comm->heteroComm, stream));
  return flagcxSuccess;
}

flagcxResult_t uniRunnerGroupStart() {
  FLAGCXCHECK(flagcxHeteroGroupStart());
  return flagcxSuccess;
}

flagcxResult_t uniRunnerGroupEnd() {
  FLAGCXCHECK(flagcxHeteroGroupEnd());
  return flagcxSuccess;
}

struct flagcxRunner uniRunner = {
    // Communication functions
    uniRunnerReduce, uniRunnerGather, uniRunnerScatter, uniRunnerBroadcast,
    uniRunnerAllReduce, uniRunnerReduceScatter, uniRunnerAllGather,
    uniRunnerAlltoAll, uniRunnerAlltoAllv, uniRunnerSend, uniRunnerRecv,
    // Group semantics
    uniRunnerGroupStart, uniRunnerGroupEnd};
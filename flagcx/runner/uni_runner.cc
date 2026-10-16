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

// Forward declaration: the P8 plain-path fallback below calls uniRunnerAllGather
// whose definition appears later in this file (Send/Recv naive allgather).
flagcxResult_t uniRunnerAllGather(const void *sendbuff, void *recvbuff,
                                  size_t sendcount, flagcxDataType_t datatype,
                                  flagcxComm_t comm, flagcxStream_t stream);

#if defined(FLAGCX_USE_ASCEND_DAG)
// #8 Ascend DAG capability probe cache (once per process):
//   -1 = not probed; 1 = DAG available (.run installed + launch ok);
//   0 = unavailable (degrade to P8 plain path).
// Probe = a real first DAG run: capability loss (.run not installed /
// GetWorkspaceSize failure) surfaces at launch time, leaves no persistent
// kernel behind, so cleanupUniRunner is safe.
static int ascendDagProbe = -1;
#endif

flagcxResult_t uniRunnerAllReduce(const void *sendbuff, void *recvbuff,
                                  size_t count, flagcxDataType_t datatype,
                                  flagcxRedOp_t op, flagcxComm_t comm,
                                  flagcxStream_t stream) {
#if defined(FLAGCX_USE_ASCEND_DAG)
  // ---------- Ascend: #8 DAG first + runtime probe, auto degrade to P8 ----------
  if (ascendDagProbe != 0) {
    flagcxResult_t res = flagcxSuccess;
    flagcxHeteroComm_t hcomm = comm->heteroComm;
    flagcxUniRunnerState *runnerState = &hcomm->proxyState->uniRunnerState;
    FLAGCXCHECKGOTO(initUniRunner(comm, stream), res, plain);
    if (flagcxParamUniRunnerUseLocRed()) {
      /* initialize uniRunnerState for reduce test */
      FLAGCXCHECKGOTO(initUniRunnerStateLocRed(runnerState, sendbuff, recvbuff,
                                               count, datatype, op, comm),
                      res, dag_cleanup);
    } else if (flagcxParamUniRunnerUseRingAG()) {
      /* initialize uniRunnerState for p2p test */
      FLAGCXCHECKGOTO(initUniRunnerStateRingAG(runnerState, sendbuff, recvbuff,
                                               count, datatype, op, comm),
                      res, dag_cleanup);
    } else if (flagcxParamUniRunnerUseSlicedAR()) {
      /* initialize uniRunnerState for sliced AllReduce */
      FLAGCXCHECKGOTO(initUniRunnerStateSlicedAR(runnerState, sendbuff, recvbuff,
                                                 count, datatype, op, comm),
                      res, dag_cleanup);
    } else {
      /* initialize uniRunnerState for ring AllReduce */
      FLAGCXCHECKGOTO(initUniRunnerStateRingAR(runnerState, sendbuff, recvbuff,
                                               count, datatype, op, comm),
                      res, dag_cleanup);
    }
    FLAGCXCHECKGOTO(runUniRunner(comm), res, dag_cleanup);
  dag_cleanup:
    FLAGCXCHECK(cleanupUniRunner(comm));
    if (res == flagcxSuccess) {
      if (ascendDagProbe < 0) ascendDagProbe = 1;
      return flagcxSuccess;
    }
    if (ascendDagProbe < 0) ascendDagProbe = 0;
    WARN("uniRunnerAllReduce: DAG path failed (%d) - degrade to plain path (P8)",
         (int)res);
  }
plain:
  // ---------- P8 plain path (Kistich fallback, verified on 910C) ----------
  // The DAG engine (initUniRunner) needs UVA (device access to host fifo via
  // hostGetDevicePointer) + a persistent consumer kernel; when either is
  // missing at runtime we degrade here: allgather (Send/Recv, verified
  // working) + host reduce + H2D copy. Correctness over speed.
  {
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

    // slice-bounded temp buffers: the one-shot path allocated bytes*nranks
    // on EVERY call; on a 24GB card already holding ~21GB of training state
    // the second allreduce's allocation fails silently inside DEVCHECK and
    // surfaces as flagcxUnhandledDeviceError. Process in slices so the temp
    // footprint stays bounded (128MB/slice default, FLAGCX_HETERO_AR_SLICE_MB
    // configurable).
    size_t sliceBytes = 128 << 20; // 128MB per slice by default
    const char *sliceEnv = getenv("FLAGCX_HETERO_AR_SLICE_MB");
    if (sliceEnv) {
      long sliceMb = atol(sliceEnv);
      if (sliceMb > 0) sliceBytes = (size_t)sliceMb << 20;
    }
    size_t sliceCount = sliceBytes / esize;
    if (sliceCount == 0) sliceCount = 1;

    void *tmpDev = nullptr;
    FLAGCXCHECK(deviceAdaptor->deviceMalloc(&tmpDev, sliceCount * esize * nranks,
                                            flagcxMemDevice, stream));
    char *hostBuf = (char *)malloc(sliceCount * esize * nranks);
    if (hostBuf == nullptr) {
      FLAGCXCHECK(deviceAdaptor->deviceFree(tmpDev, flagcxMemDevice, stream));
      return flagcxSystemError;
    }

    for (size_t off = 0; off < count; off += sliceCount) {
      size_t n = (count - off) < sliceCount ? (count - off) : sliceCount;
      // 1) gather this slice from all ranks into tmpDev (nranks sub-slices)
      FLAGCXCHECK(uniRunnerAllGather(
          (const char *)sendbuff + off * esize, tmpDev, n, datatype, comm,
          stream));
      // 2) D2H the gathered slice, then reduce on host
      FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
          hostBuf, tmpDev, n * esize * nranks, flagcxMemcpyDeviceToHost, stream,
          nullptr));
      FLAGCXCHECK(deviceAdaptor->streamSynchronize(stream));
      {
        // Shadow count/bytes so the per-dtype reduce switch below (kept
        // verbatim) operates on the current slice.
        size_t count = n;
        size_t bytes = n * esize;

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
        // 3) H2D the reduced slice (bytes is the shadowed slice size)
        FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
            (char *)recvbuff + off * esize, hostBuf, bytes,
            flagcxMemcpyHostToDevice, stream, nullptr));
      } // end slice scope (count/bytes shadow)
    }   // end slice loop
    free(hostBuf);
    FLAGCXCHECK(deviceAdaptor->deviceFree(tmpDev, flagcxMemDevice, stream));
    return flagcxSuccess;
  }
#else
  // ---------- other platforms: original DAG logic (unchanged) ----------
  flagcxResult_t res = flagcxSuccess;
  flagcxHeteroComm_t hcomm = comm->heteroComm;
  flagcxUniRunnerState *runnerState = &hcomm->proxyState->uniRunnerState;
  FLAGCXCHECK(initUniRunner(comm, stream));
  if (flagcxParamUniRunnerUseLocRed()) {
    /* initialize uniRunnerState for reduce test */
    FLAGCXCHECKGOTO(initUniRunnerStateLocRed(runnerState, sendbuff, recvbuff,
                                             count, datatype, op, comm),
                    res, out);
  } else if (flagcxParamUniRunnerUseRingAG()) {
    /* initialize uniRunnerState for p2p test */
    FLAGCXCHECKGOTO(initUniRunnerStateRingAG(runnerState, sendbuff, recvbuff,
                                             count, datatype, op, comm),
                    res, out);
  } else if (flagcxParamUniRunnerUseSlicedAR()) {
    /* initialize uniRunnerState for sliced AllReduce */
    FLAGCXCHECKGOTO(initUniRunnerStateSlicedAR(runnerState, sendbuff, recvbuff,
                                               count, datatype, op, comm),
                    res, out);
  } else {
    /* initialize uniRunnerState for ring AllReduce */
    FLAGCXCHECKGOTO(initUniRunnerStateRingAR(runnerState, sendbuff, recvbuff,
                                             count, datatype, op, comm),
                    res, out);
  }
  FLAGCXCHECK(runUniRunner(comm));
out:
  FLAGCXCHECK(cleanupUniRunner(comm));
  return res;
#endif
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
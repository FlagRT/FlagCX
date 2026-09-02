#include "ascend_adaptor.h"
#include <unordered_set>

#ifdef USE_ASCEND_ADAPTOR

#include "adaptor.h"
#include "alloc.h"
// aclnn 单算子：设备侧 reduce（dst += src）
#include "aclnn/aclnn_base.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_add.h"
#ifdef FLAGCX_USE_ASCEND_DAG
#include "aclnn_flagcx_collective.h"
#endif

std::map<flagcxMemcpyType_t, aclrtMemcpyKind> memcpy_type_map = {
    {flagcxMemcpyHostToDevice, ACL_MEMCPY_HOST_TO_DEVICE},
    {flagcxMemcpyDeviceToHost, ACL_MEMCPY_DEVICE_TO_HOST},
    {flagcxMemcpyDeviceToDevice, ACL_MEMCPY_DEVICE_TO_DEVICE},
};

flagcxResult_t cannAdaptorDeviceSynchronize() {
  DEVCHECK(aclrtSynchronizeDevice());
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorDeviceMemcpy(void *dst, void *src, size_t size,
                                       flagcxMemcpyType_t type,
                                       flagcxStream_t stream, void *args) {
  if (stream == NULL) {
    DEVCHECK(aclrtMemcpy(dst, size, src, size, memcpy_type_map[type]));
  } else {
    DEVCHECK(aclrtMemcpyAsync(dst, size, src, size, memcpy_type_map[type],
                              stream->base));
  }
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorDeviceMemset(void *ptr, int value, size_t size,
                                       flagcxMemType_t type,
                                       flagcxStream_t stream) {
  if (type == flagcxMemHost) {
    memset(ptr, value, size);
  } else {
    if (stream == NULL) {
      DEVCHECK(aclrtMemset(ptr, size, value, size));
    } else {
      DEVCHECK(aclrtMemsetAsync(ptr, size, value, size, stream->base));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorDeviceMalloc(void **ptr, size_t size,
                                       flagcxMemType_t type,
                                       flagcxStream_t stream) {

  if (type == flagcxMemHost) {
    // #6: aclrtMallocHost 后必须显式 aclrtHostRegisterV2，否则
    // aclrtHostGetDevicePointer 返回成功但设备别名=NULL（t6 T1 实测）。
    // 用 V2（P0 探针验证组合）；旧 4 参 aclrtHostRegister 在 8.5.0 报
    // ACL_ERROR_RT_DRV_INTERNAL_ERROR(507899)（t6 T2 实测）。
    DEVCHECK(aclrtMallocHost(ptr, size));
    DEVCHECK(aclrtHostRegisterV2(*ptr, size,
                                  ACL_HOST_REG_PINNED | ACL_HOST_REG_MAPPED));
  } else {
    DEVCHECK(aclrtMalloc(ptr, size, ACL_MEM_MALLOC_HUGE_FIRST));
  }
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorDeviceFree(void *ptr, flagcxMemType_t type,
                                     flagcxStream_t stream) {
  if (type == flagcxMemHost) {
    // #6: 对称于 deviceMalloc 里的 aclrtHostRegisterV2；未注册时 unregister
    // 返回 ACL_ERROR_HOST_MEMORY_NOT_REGISTERED(507911)，忽略即可
    aclrtHostUnregister(ptr);
    DEVCHECK(aclrtFreeHost(ptr));
  } else {
    DEVCHECK(aclrtFree(ptr));
  }
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorSetDevice(int dev) {
  DEVCHECK(aclrtSetDevice(dev));
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorGetDevice(int *dev) {
  DEVCHECK(aclrtGetDevice(dev));
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorGetDeviceCount(int *count) {
  DEVCHECK(aclrtGetDeviceCount((uint32_t *)count));
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorGetVendor(char *vendor) {
  strcpy(vendor, "ASCEND");
  return flagcxSuccess;
}

// #6: hostGetDevicePointer 真实现（对标 cudaHostGetDevicePointer）。
// CANN 的 aclrtHostGetDevicePointer 参数顺序是 (pHost, pDevice, flag)，
// 与 CUDA 的 cudaHostGetDevicePointer(pDevice, pHost, 0) 相反，勿写反。
// 且对未 register 的内存会"返回成功但别名=NULL"（t6 T1 实测），必须判 NULL。
flagcxResult_t cannAdaptorHostGetDevicePointer(void **pDevice, void *pHost) {
  if (pDevice == NULL || pHost == NULL) {
    return flagcxInvalidArgument;
  }
  aclError e = aclrtHostGetDevicePointer(pHost, pDevice, 0);
  if (e != ACL_SUCCESS) {
    return flagcxUnhandledDeviceError;
  }
  if (*pDevice == NULL) {
    return flagcxInternalError; // 内存未 register，别名不可用
  }
  return flagcxSuccess;
}
// TODO:unsupport
flagcxResult_t cannAdaptorGdrMemAlloc(void **ptr, size_t size,
                                      void *memHandle) {
  if (ptr == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(aclrtMalloc(ptr, size, ACL_MEM_MALLOC_HUGE_FIRST));
  return flagcxSuccess;
}

// TODO:unsupported
flagcxResult_t cannAdaptorGdrMemFree(void *ptr, void *memHandle) {
  if (ptr == NULL) {
    return flagcxSuccess;
  }
  DEVCHECK(aclrtFree(ptr));
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorStreamCreate(flagcxStream_t *stream) {
  (*stream) = NULL;
  flagcxCalloc(stream, 1);
  DEVCHECK(aclrtCreateStream((aclrtStream *)(*stream)));
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorStreamDestroy(flagcxStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(aclrtDestroyStream(stream->base));
    free(stream);
    stream = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorStreamCopy(flagcxStream_t *newStream,
                                     void *oldStream) {
  (*newStream) = NULL;
  flagcxCalloc(newStream, 1);
  (*newStream)->base = (aclrtStream)oldStream;
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorStreamFree(flagcxStream_t stream) {
  if (stream != NULL) {
    free(stream);
    stream = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorStreamSynchronize(flagcxStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(aclrtSynchronizeStream(stream->base));
  }
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorStreamQuery(flagcxStream_t stream) {
  flagcxResult_t res = flagcxSuccess;
  if (stream != NULL) {
    aclrtStreamStatus status;
    DEVCHECK(aclrtStreamQuery(stream->base, &status));
    if (status == ACL_STREAM_STATUS_COMPLETE) {
      res = flagcxSuccess;
    } else if (status == ACL_STREAM_STATUS_NOT_READY) {
      res = flagcxInProgress;
    } else {
      res = flagcxUnhandledDeviceError;
    }
  }
  return res;
}

flagcxResult_t cannAdaptorStreamWaitEvent(flagcxStream_t stream,
                                          flagcxEvent_t event) {
  if (stream != NULL && event != NULL) {
    DEVCHECK(aclrtStreamWaitEvent(stream->base, event->base));
  }
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorEventCreate(flagcxEvent_t *event,
                                      flagcxEventType_t eventType) {
  (*event) = NULL;
  flagcxCalloc(event, 1);
  const unsigned int flags =
      (eventType == flagcxEventDefault) ? ACL_EVENT_TIME_LINE : ACL_EVENT_SYNC;
  DEVCHECK(aclrtCreateEventWithFlag(&((*event)->base), flags));
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorEventDestroy(flagcxEvent_t event) {
  if (event != NULL) {
    DEVCHECK(aclrtDestroyEvent(event->base));
    free(event);
    event = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorEventRecord(flagcxEvent_t event,
                                      flagcxStream_t stream) {
  if (event != NULL) {
    if (stream != NULL) {
      DEVCHECK(aclrtRecordEvent(event->base, stream->base));
    } else {
      return flagcxUnhandledDeviceError;
    }
  }
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorEventSynchronize(flagcxEvent_t event) {
  if (event != NULL) {
    DEVCHECK(aclrtSynchronizeEvent(event->base));
  }
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorEventQuery(flagcxEvent_t event) {
  flagcxResult_t res = flagcxSuccess;
  if (event != NULL) {
    aclrtEventWaitStatus status;
    DEVCHECK(aclrtQueryEventWaitStatus(event->base, &status));
    if (status == ACL_EVENT_WAIT_STATUS_COMPLETE) {
      res = flagcxSuccess;
    } else if (status == ACL_EVENT_WAIT_STATUS_NOT_READY) {
      res = flagcxInProgress;
    } else {
      res = flagcxUnhandledDeviceError;
    }
  }
  return res;
}

flagcxResult_t cannAdaptorIpcMemHandleCreate(flagcxIpcMemHandle_t *handle,
                                             size_t *size) {
  // to be implemented
  return flagcxNotSupported;
}

flagcxResult_t cannAdaptorIpcMemHandleGet(flagcxIpcMemHandle_t handle,
                                          void *devPtr) {
  // to be implemented
  return flagcxNotSupported;
}

flagcxResult_t cannAdaptorIpcMemHandleOpen(flagcxIpcMemHandle_t handle,
                                           void **devPtr) {
  // to be implemented
  return flagcxNotSupported;
}

flagcxResult_t cannAdaptorIpcMemHandleClose(void *devPtr) {
  // to be implemented
  return flagcxNotSupported;
}

flagcxResult_t cannAdaptorIpcMemHandleFree(flagcxIpcMemHandle_t handle) {
  // to be implemented
  return flagcxNotSupported;
}

flagcxResult_t cannAdaptorLaunchHostFunc(flagcxStream_t stream,
                                         void (*fn)(void *), void *args) {
  if (stream == NULL || stream->base == nullptr) {
    // Kistich(fix-stale-data): base==nullptr is torch_npu's DEFAULT stream
    // (ACL: null IS the default stream) — NOT "no stream". Running the host
    // func directly skips stream ordering: signalStart fired before the
    // tensor-producing kernels finished, so the proxy's D2H copied stale
    // tensor contents. Synchronize the device first to restore the
    // happens-before that CUDA gets for free via the legacy default stream.
    fprintf(stderr, "[P5-SYNC] NULL-stream branch: syncing device before host func\n");
    fflush(stderr);
    aclError syncErr = aclrtSynchronizeDevice();
    (void)syncErr;
    fn(args);
    return flagcxSuccess;
  }
  // Kistich(910C-hetero): ACL callbacks require the stream to be subscribed
  // to a thread first (aclrtSubscribeReport); otherwise aclrtLaunchCallback
  // fails with 107015 (ACL_ERROR_RT_STREAM_NO_CB_REG). Subscribe this
  // stream to the calling thread once, then launch + process the callback
  // in stream order (aclrtProcessReport blocks until the callback runs at
  // its stream position -- correct ordering for the host semaphore).
  static thread_local std::unordered_set<aclrtStream> subscribedStreams;
  if (subscribedStreams.insert(stream->base).second) {
    aclError serr = aclrtSubscribeReport((uint64_t)pthread_self(), stream->base);
    if (serr != ACL_SUCCESS) {
      fn(args);
      return flagcxSuccess;
    }
  }
  // Kistich(fix-stale-data): observed ACL callback execution does NOT wait
  // for prior work on the stream -- the proxy's D2H read tensors one
  // collective late (stale data on the wire). Explicitly synchronize the
  // stream first: group.cc enqueued streamWaitEvent(launchStream, op->event)
  // chained to eventRecord(op->event, op->stream), so this sync transitively
  // waits for all tensor-producing work before signalStart fires.
  aclError syncErr = aclrtSynchronizeStream(stream->base);
  (void)syncErr;
  aclError err =
      aclrtLaunchCallback(fn, args, ACL_CALLBACK_NO_BLOCK, stream->base);
  if (err != ACL_SUCCESS) {
    fn(args);
    return flagcxSuccess;
  }
  // Block until the callback executes (at its stream position).
  aclError perr = aclrtProcessReport(-1);
  (void)perr;
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorStreamWaitValue64(flagcxStream_t, void *, uint64_t,
                                            int) {
  return flagcxNotSupported;
}
flagcxResult_t cannAdaptorStreamWriteValue64(flagcxStream_t, void *, uint64_t,
                                             int) {
  return flagcxNotSupported;
}

flagcxResult_t cannAdaptorReduceSum(void *dst, const void *src, size_t count,
                                    flagcxDataType_t datatype,
                                    flagcxStream_t stream) {
  aclDataType aclDt;
  switch (datatype) {
  case flagcxFloat32: aclDt = ACL_FLOAT; break;
  case flagcxFloat16: aclDt = ACL_FLOAT16; break;
  case flagcxBfloat16: aclDt = ACL_BF16; break;
  default: return flagcxInvalidArgument;
  }
  int64_t dims[1] = {(int64_t)count};
  aclTensor *self = aclCreateTensor(dims, 1, aclDt, nullptr, 0, ACL_FORMAT_ND, nullptr, 0, dst);
  aclTensor *other = aclCreateTensor(dims, 1, aclDt, nullptr, 0, ACL_FORMAT_ND, nullptr, 0, const_cast<void *>(src));
  float one = 1.0f;
  aclScalar *alpha = aclCreateScalar(&one, ACL_FLOAT);
  uint64_t wsSize = 0;
  aclOpExecutor *executor = nullptr;
  aclnnStatus st = aclnnInplaceAddGetWorkspaceSize(self, other, alpha, &wsSize, &executor);
  if (st == 0) {
    void *ws = nullptr;
    aclrtMalloc(&ws, wsSize ? wsSize : 1, ACL_MEM_MALLOC_HUGE_FIRST);
    st = aclnnInplaceAdd(ws, wsSize, executor, stream->base);
    aclrtFree(ws);
  }
  aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyScalar(alpha);
  return st == 0 ? flagcxSuccess : flagcxUnhandledDeviceError;
}

flagcxResult_t cannAdaptorEventElapsedTime(float *, flagcxEvent_t,
                                           flagcxEvent_t) {
  return flagcxNotSupported;
}

flagcxResult_t cannAdaptorHostRegister(void *ptr, size_t size) {
  if (ptr == NULL || size == 0) {
    return flagcxInvalidArgument;
  }
  // #6: 对标 cudaHostRegister(ptr, size, cudaHostRegisterMapped)。
  // 必须用 aclrtHostRegisterV2：旧 4 参 aclrtHostRegister 在 8.5.0 上报
  // ACL_ERROR_RT_DRV_INTERNAL_ERROR(507899)（t6 T2 实测）；V2 为 P0 探针验证组合。
  DEVCHECK(aclrtHostRegisterV2(ptr, size,
                               ACL_HOST_REG_PINNED | ACL_HOST_REG_MAPPED));
  return flagcxSuccess;
}
flagcxResult_t cannAdaptorHostUnregister(void *ptr) {
  if (ptr == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(aclrtHostUnregister(ptr));
  return flagcxSuccess;
}

// Symmetric memory VMM stubs (not supported)
flagcxResult_t cannAdaptorSymPhysAlloc(void *, size_t, void **, void *,
                                       size_t *, size_t *) {
  return flagcxNotSupported;
}
flagcxResult_t cannAdaptorSymPhysFree(void *) { return flagcxNotSupported; }
flagcxResult_t cannAdaptorSymFlatMap(void *[], int, int, void *, size_t,
                                     void **) {
  return flagcxNotSupported;
}
flagcxResult_t cannAdaptorSymFlatUnmap(void *, size_t, int) {
  return flagcxNotSupported;
}
flagcxResult_t cannAdaptorSymMulticastSupported(int *supported) {
  if (supported)
    *supported = 0;
  return flagcxSuccess;
}
flagcxResult_t cannAdaptorSymMulticastCreate(size_t, int, const int *, void **,
                                             int *) {
  return flagcxNotSupported;
}
flagcxResult_t cannAdaptorSymMulticastBind(void *, int, void *, size_t, int,
                                           int, void **, size_t *) {
  return flagcxNotSupported;
}
flagcxResult_t cannAdaptorSymMulticastTeardown(void *, size_t) {
  return flagcxSuccess;
}
flagcxResult_t cannAdaptorSymMulticastFree(void *) {
  return flagcxNotSupported;
}

#ifdef FLAGCX_USE_ASCEND_DAG
// #7.2 昇腾持久 collective kernel launch（DAG FIFO 消费方）。
// 对标 CUDA flagcxLaunchCollectiveKernel：aclnn 异步 launch、不 sync，
// kernel 驻留自旋消费 FIFO 直至 terminate。fifoBuffer = host-mapped 设备别名
// （initUniRunner 经 hostGetDevicePointer 取得）。
// 调用序列与 t7/t8 单测完全一致（aclCreateTensor 9 参 + GetWorkspaceSize +
// ws aclrtMalloc + aclnnFlagcxCollective 异步）。
flagcxResult_t cannAdaptorLaunchCollectiveKernel(void *fifoBuffer,
                                                 size_t nthreads,
                                                 size_t nblocks,
                                                 void *stream) {
  if (fifoBuffer == NULL || stream == NULL) {
    WARN("cannAdaptorLaunchCollectiveKernel: null fifoBuffer/stream");
    return flagcxInvalidArgument;
  }
  int64_t shape[1] = {1};
  int64_t stride[1] = {1};
  aclTensor *fifoTensor = aclCreateTensor(shape, 1, ACL_INT64, stride, 0,
                                          ACL_FORMAT_ND, shape, 1, fifoBuffer);
  if (fifoTensor == NULL) {
    WARN("cannAdaptorLaunchCollectiveKernel: aclCreateTensor(fifo) failed");
    return flagcxSystemError;
  }
  uint64_t wsSize = 0;
  aclOpExecutor *exec = nullptr;
  aclError e = aclnnFlagcxCollectiveGetWorkspaceSize(fifoTensor, &wsSize, &exec);
  if (e != ACL_SUCCESS) {
    WARN("cannAdaptorLaunchCollectiveKernel: GetWorkspaceSize=%d", (int)e);
    return flagcxUnhandledDeviceError;
  }
  void *ws = nullptr;
  if (wsSize > 0) {
    e = aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (e != ACL_SUCCESS) {
      WARN("cannAdaptorLaunchCollectiveKernel: ws aclrtMalloc=%d", (int)e);
      return flagcxUnhandledDeviceError;
    }
  }
  // 异步 launch：kernel 驻留。ws/tensor 生命周期由 device 持有至任务完成、
  // 进程退出时回收（每 comm 一次 launch，泄漏可忽略；避免异步释放竞态）。
  e = aclnnFlagcxCollective(ws, wsSize, exec, ((flagcxStream_t)stream)->base);
  if (e != ACL_SUCCESS) {
    WARN("cannAdaptorLaunchCollectiveKernel: aclnn launch=%d", (int)e);
    return flagcxUnhandledDeviceError;
  }
  return flagcxSuccess;
}
#else
flagcxResult_t cannAdaptorLaunchCollectiveKernel(void *, size_t, size_t,
                                                 void *) {
  return flagcxNotSupported;
}
#endif // FLAGCX_USE_ASCEND_DAG


// Kistich(910C-hetero): PCI bus id for topology/nic-distance detection in
// flagcxHeteroCommInitRank. CANN has no direct API; resolve via sysfs, and
// fall back to a pseudo id derived from the device index when sysfs is not
// mounted (containers). Socket transport does not depend on precise topo.
static flagcxResult_t cannAdaptorGetDevicePciBusId(char *pciBusId, int len,
                                                    int dev) {
  if (pciBusId == NULL || len < 16) {
    return flagcxInvalidArgument;
  }
  char path[128];
  snprintf(path, sizeof(path), "/sys/class/davinci%d/device", dev);
  char *resolved = realpath(path, NULL);
  if (resolved != NULL) {
    char *p = resolved, *last = NULL;
    while ((p = strstr(p, "0000:")) != NULL) {
      last = p;
      p += 1;
    }
    if (last != NULL) {
      snprintf(pciBusId, len, "%s", last);
      char *slash = strchr(pciBusId, '/');
      if (slash)
        *slash = '\0';
      free(resolved);
      return flagcxSuccess;
    }
    free(resolved);
  }
  // Fallback pseudo bus id (device index encoded); enough for nic-distance
  // heuristics over socket transport.
  snprintf(pciBusId, len, "0000:80:%02x.0", dev);
  return flagcxSuccess;
}

struct flagcxDeviceAdaptor cannAdaptor {
  "CANN",
      // Basic functions
      cannAdaptorDeviceSynchronize, cannAdaptorDeviceMemcpy,
      cannAdaptorDeviceMemset, cannAdaptorDeviceMalloc, cannAdaptorDeviceFree,
      cannAdaptorSetDevice, cannAdaptorGetDevice, cannAdaptorGetDeviceCount,
      cannAdaptorGetVendor, cannAdaptorHostGetDevicePointer,
      // GDR functions
      NULL, // flagcxResult_t (*memHandleInit)(int dev_id, void **memHandle);
      NULL, // flagcxResult_t (*memHandleDestroy)(int dev, void *memHandle);
      cannAdaptorGdrMemAlloc, cannAdaptorGdrMemFree,
      NULL, // flagcxResult_t (*hostShareMemAlloc)(void **ptr, size_t size, void
            // *memHandle);
      NULL, // flagcxResult_t (*hostShareMemFree)(void *ptr, void *memHandle);
      NULL, // flagcxResult_t (*gdrPtrMmap)(void **pcpuptr, void *devptr, size_t
            // sz);
      NULL, // flagcxResult_t (*gdrPtrMunmap)(void *cpuptr, size_t sz);
      cannAdaptorReduceSum,
      // Stream functions
      cannAdaptorStreamCreate, cannAdaptorStreamDestroy, cannAdaptorStreamCopy,
      cannAdaptorStreamFree, cannAdaptorStreamSynchronize,
      cannAdaptorStreamQuery, cannAdaptorStreamWaitEvent,
      cannAdaptorStreamWaitValue64, cannAdaptorStreamWriteValue64,
      // Event functions
      cannAdaptorEventCreate, cannAdaptorEventDestroy, cannAdaptorEventRecord,
      cannAdaptorEventSynchronize, cannAdaptorEventQuery,
      cannAdaptorEventElapsedTime,
      // IpcMemHandle functions
      cannAdaptorIpcMemHandleCreate, cannAdaptorIpcMemHandleGet,
      cannAdaptorIpcMemHandleOpen, cannAdaptorIpcMemHandleClose,
      cannAdaptorIpcMemHandleFree,
      // Kernel launch
      NULL, // flagcxResult_t (*launchKernel)(void *func, unsigned int block_x,
            // unsigned int block_y, unsigned int block_z, unsigned int grid_x,
            // unsigned int grid_y, unsigned int grid_z, void **args, size_t
            // share_mem, void *stream, void *memHandle);
      NULL, // flagcxResult_t (*copyArgsInit)(void **args);
      NULL, // flagcxResult_t (*copyArgsFree)(void *args);
      NULL, // flagcxResult_t (*launchDeviceFunc)(flagcxStream_t stream, void
            // *args);
      // Others
      NULL, // flagcxResult_t (*getDeviceProperties)(struct flagcxDevProps
            // *props, int dev);
      cannAdaptorGetDevicePciBusId, // flagcxResult_t (*getDevicePciBusId)(char
                                    // *pciBusId, int len, int dev);
      NULL, // flagcxResult_t
            // (*getDeviceByPciBusId)(int
            // *dev, const char *pciBusId);
      cannAdaptorLaunchHostFunc,
      // DMA buffer
      NULL, // flagcxResult_t (*dmaSupport)(bool *dmaBufferSupport);
      NULL, // flagcxResult_t (*memGetHandleForAddressRange)(void *handleOut,
            // void *buffer, size_t size, unsigned long long flags);
      cannAdaptorHostRegister,   // flagcxResult_t (*hostRegister)(void *,
                                 // size_t);
      cannAdaptorHostUnregister, // flagcxResult_t (*hostUnregister)(void *);
      // Symmetric memory VMM functions (not supported)
      cannAdaptorSymPhysAlloc, cannAdaptorSymPhysFree, cannAdaptorSymFlatMap,
      cannAdaptorSymFlatUnmap, cannAdaptorSymMulticastSupported,
      cannAdaptorSymMulticastCreate, cannAdaptorSymMulticastBind,
      cannAdaptorSymMulticastTeardown, cannAdaptorSymMulticastFree,
      NULL, // flagcxResult_t (*getLastError)();
      cannAdaptorLaunchCollectiveKernel, // 持久 collective kernel launch
                                         //（DAG FIFO 消费方，#7.2）
};

#endif // USE_ASCEND_ADAPTOR

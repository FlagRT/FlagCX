#include "ascend_adaptor.h"
#include <unordered_set>

#ifdef USE_ASCEND_ADAPTOR

#include "adaptor.h"
#include "alloc.h"

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
    DEVCHECK(aclrtMallocHost(ptr, size));
  } else {
    DEVCHECK(aclrtMalloc(ptr, size, ACL_MEM_MALLOC_HUGE_FIRST));
  }
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorDeviceFree(void *ptr, flagcxMemType_t type,
                                     flagcxStream_t stream) {
  if (type == flagcxMemHost) {
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
    // No stream: run the host func directly.
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
flagcxResult_t cannAdaptorEventElapsedTime(float *, flagcxEvent_t,
                                           flagcxEvent_t) {
  return flagcxNotSupported;
}

flagcxResult_t cannAdaptorHostRegister(void *, size_t) {
  return flagcxNotSupported;
}
flagcxResult_t cannAdaptorHostUnregister(void *) { return flagcxNotSupported; }

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
      cannAdaptorGetVendor, NULL,
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
};

#endif // USE_ASCEND_ADAPTOR

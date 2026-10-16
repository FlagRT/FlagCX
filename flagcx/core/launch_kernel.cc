#include "launch_kernel.h"
#include "group.h"
#include <stdio.h>

flagcxLaunchFunc_t deviceAsyncKernel = NULL;

FLAGCX_PARAM(SemaphoreBufferPoolCapacity, "SEMAPHORE_BUFFER_POOL_CAPACITY", 32);

flagcxDeviceSemaphoreBufferPool::flagcxDeviceSemaphoreBufferPool()
    : capacity(-1), slotId(-1), signalsPool(nullptr), dSignalsPool(nullptr),
      events(nullptr) {}

flagcxDeviceSemaphoreBufferPool::~flagcxDeviceSemaphoreBufferPool() {
  free(events);
  dSignalsPool = nullptr;
  if (signalsPool != nullptr) {
    deviceAdaptor->deviceFree((void *)signalsPool, flagcxMemHost, nullptr);
  }
}

int flagcxDeviceSemaphoreBufferPool::getSlotId() {
  assert(capacity != -1);
  if (events[slotId] != nullptr) {
    // wait for the previous event to complete
    while (deviceAdaptor->eventQuery(events[slotId]) != flagcxSuccess) {
      sched_yield();
    }
    events[slotId] = nullptr;
  }
  // set this slot signals to zero
  int offset = FLAGCX_SIGNALS_PER_SEMAPHORE * slotId;
  memset(signalsPool + offset, 0, FLAGCX_SIGNALS_PER_SEMAPHORE * sizeof(int));
  int ret = slotId;
  // Move to next slot
  slotId = (slotId + 1) % capacity;
  return ret;
}

void flagcxDeviceSemaphoreBufferPool::initialize() {
  capacity = flagcxParamSemaphoreBufferPoolCapacity();
  slotId = 0;
  // Allocate host-pinned memory for all semaphores (3 ints each)
  deviceAdaptor->deviceMalloc((void **)&signalsPool,
                              capacity * FLAGCX_SIGNALS_PER_SEMAPHORE *
                                  sizeof(int),
                              flagcxMemHost, nullptr);
  // Get device pointer alias
  deviceAdaptor->hostGetDevicePointer(&dSignalsPool, (void *)signalsPool);
  // Init events to nullptr
  flagcxCalloc(&events, capacity);
  for (int i = 0; i < capacity; i++) {
    events[i] = nullptr;
  }
}

// Set event for a semaphore
void flagcxDeviceSemaphoreBufferPool::setEvent(int id, flagcxEvent_t event) {
  assert(id >= 0 && id < capacity);
  // events[id] should be set to nullptr before
  events[id] = event;
}

// Return pointer to the start of a semaphore’s signals (host/device)
int *flagcxDeviceSemaphoreBufferPool::getHostPtr(int id) {
  assert(id >= 0 && id < capacity);
  return signalsPool + FLAGCX_SIGNALS_PER_SEMAPHORE * id;
}
void *flagcxDeviceSemaphoreBufferPool::getDevicePtr(int id) {
  assert(id >= 0 && id < capacity);
  return static_cast<void *>((static_cast<char *>(dSignalsPool) +
                              FLAGCX_SIGNALS_PER_SEMAPHORE * id * sizeof(int)));
}

void cpuAsyncKernel(void *args) {
  flagcxHostSemaphore *semaphore = (flagcxHostSemaphore *)args;
  semaphore->signalStart();
  // Kistich(fix-hetero-deadlock): never block inside a launchHostFunc
  // callback. The original semaphore->wait() here spins while the CUDA
  // driver holds its callback/lock machinery, which blocks the proxy
  // thread's cudaMemcpyAsync -> op never completes -> wait never returns.
  // Completion is enforced by the main thread in flagcxGroupLaunch.
}
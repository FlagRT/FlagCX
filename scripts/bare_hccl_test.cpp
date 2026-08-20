// Minimal bare-HCCL diagnostics: 16 ranks via MPI, each binds device rank%16,
// does HcclAllReduce on a small buffer, prints result. No FlagCX involved.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "acl/acl.h"
#include "hccl/hccl.h"
#include "mpi.h"

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, nranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  uint32_t numDevices = 0;
  aclError aclRet = aclInit(nullptr);
  if (aclRet != ACL_SUCCESS) {
    printf("[rank %d] aclInit failed: %d\n", rank, aclRet);
    MPI_Finalize();
    return 1;
  }
  aclrtSetDevice(rank % 16);
  // verify device count
  aclrtGetDeviceCount(&numDevices);
  printf("[rank %d] aclInit OK, devices=%d, using device %d\n", rank,
         numDevices, rank % 16);

  // build HCCL comm from root info
  HcclRootInfo rootInfo;
  memset(&rootInfo, 0, sizeof(rootInfo));
  if (rank == 0) {
    HcclResult hcclRet = HcclGetRootInfo(&rootInfo);
    printf("[rank %d] HcclGetRootInfo -> %d (%s)\n", rank, hcclRet,
           HcclGetErrorString(hcclRet));
  }
  MPI_Bcast(&rootInfo, sizeof(rootInfo), MPI_BYTE, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  HcclComm comm;
  HcclResult hcclRet =
      HcclCommInitRootInfo(nranks, &rootInfo, rank, &comm);
  printf("[rank %d] HcclCommInitRootInfo -> %d (%s)\n", rank, hcclRet,
         HcclGetErrorString(hcclRet));
  if (hcclRet != HCCL_SUCCESS) {
    MPI_Finalize();
    return 1;
  }

  // stream
  aclrtStream stream;
  aclrtCreateStream(&stream);

  const int count = 1024;
  float *sendbuf = nullptr, *recvbuf = nullptr;
  aclrtMalloc((void **)&sendbuf, count * sizeof(float), ACL_MEM_MALLOC_NORMAL_ONLY);
  aclrtMalloc((void **)&recvbuf, count * sizeof(float), ACL_MEM_MALLOC_NORMAL_ONLY);
  std::vector<float> hsend(count), hrecv(count, -1.0f);
  for (int i = 0; i < count; i++)
    hsend[i] = (float)(rank + 1);
  aclrtMemcpyAsync(sendbuf, count * sizeof(float), hsend.data(),
                   count * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE, stream);

  hcclRet = HcclAllReduce(sendbuf, recvbuf, count, HCCL_DATA_TYPE_FP32,
                          HCCL_REDUCE_SUM, comm, stream);
  printf("[rank %d] HcclAllReduce -> %d (%s)\n", rank, hcclRet,
         HcclGetErrorString(hcclRet));
  aclrtSynchronizeStream(stream);
  aclrtMemcpy(hrecv.data(), count * sizeof(float), recvbuf,
              count * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  float expected = (float)(nranks * (nranks + 1) / 2);
  bool ok = true;
  for (int i = 0; i < count; i++)
    if (hrecv[i] != expected) {
      ok = false;
      break;
    }
  printf("[rank %d] result[0]=%f expected=%f %s\n", rank, hrecv[0], expected,
         ok ? "OK" : "MISMATCH");

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0)
    printf("=== bare HCCL test %s ===\n", ok ? "PASSED" : "FAILED");

  HcclCommDestroy(comm);
  aclrtDestroyStream(stream);
  aclrtFree(sendbuf);
  aclrtFree(recvbuf);
  aclrtResetDevice(rank % 16);
  aclFinalize();
  MPI_Finalize();
  return ok ? 0 : 1;
}

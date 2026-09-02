#include "flagcx_collective_tiling.h"
#include "register/op_def_registry.h"

/*
 * O2 阶段三 #7 op_host：FlagcxCollective（昇腾持久 kernel，DAG 引擎设备侧消费方）
 * - 单输入（fifo = FIFO 缓冲区 host-mapped 设备别名，int64 ND shape[1]）、无输出
 * - 持久 kernel：SetBlockDim(1)（单核单块），tiling 数据不依赖输入 shape
 * - InferShape/InferDataType 无输出：直接成功，不碰 GetOutputShape/SetOutputDataType
 *   （⚠️ msopgen 生成骨架引用了不存在的输出 0，必须删除，否则运行时报越界）
 */

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
  // 持久 kernel：单核单块；tiling->size 仅占位（kernel 不依赖 tiling）
  // 注意：8.5.0 生成的 tiling 类字段私有，须用 set_size()（不能 tiling->size=）
  FlagcxCollectiveTilingData tiling;
  tiling.set_size(1);
  context->SetBlockDim(1);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                      context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
  // 单输入无输出算子：无需推导输出 shape
  return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
  // 单输入无输出算子：无需推导输出 dtype
  return GRAPH_SUCCESS;
}
}

namespace ops {
class FlagcxCollective : public OpDef {
public:
    explicit FlagcxCollective(const char* name) : OpDef(name)
    {
        this->Input("fifo")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(FlagcxCollective);
}

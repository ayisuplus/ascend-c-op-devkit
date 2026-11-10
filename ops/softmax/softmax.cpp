/**
 * softmax 算子 —— 按行做数值稳定的 softmax：y[r,c] = exp(x[r,c] - max_r) / sum_c(exp(x[r,c] - max_r))
 *
 * 和 silu 的区别：silu 是「逐元素」，softmax 是「沿最后一维归约 + 广播」——
 * 这才是 Ascend C 里最容易写出低效代码、也最值得单步看数据流的一类算子。
 *
 * 每个 AI Core 认领 ROWS_PER_CORE 行，逐行处理，单行流水线：
 *   CopyIn(1 行) -> ReduceMax(AR) -> 减 max -> Exp -> ReduceSum(AR) -> 乘 1/sum -> CopyOut
 *
 * 在 VSCode 里的看点：
 *   1. ReduceMax/ReduceSum 的 pattern 参数：AR = 归约最后一维（v220 只支持 AR/RA，用 R 会 static_assert 报错）
 *   2. 归约结果落在 UB 里，用 LocalTensor::GetValue() 取回标量，再作为 Adds/Muls 的标量参与广播
 *   3. 「减最大值」不是可选项：把 ReduceMax 换成 rowMax = 0，喂 +100 的输入，
 *      溢出元素直接变 NaN（Host 侧 Verify 会因为「NaN 让所有比较为 false」而假 PASS，
 *      本文件已经加了 isfinite 检查，所以它会老老实实报 FAIL）
 *   4. 断点打在 KernelSoftmax::Compute()，逐行看 tileLocal 每一轮的值怎么变
 */
#include "kernel_operator.h"
#include "tikicpulib.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

// ============================ Tiling 常量 ============================
constexpr int32_t TOTAL_ROWS = 64;                                     // 批大小（行数）
constexpr int32_t COLS = 1024;                                         // 每行元素数（= 一次搬完的 tile）
constexpr int32_t USE_CORE_NUM = 8;                                    // 用几个 AI Core
constexpr int32_t ROWS_PER_CORE = TOTAL_ROWS / USE_CORE_NUM;           // = 8 行/核
constexpr int32_t BUFFER_NUM = 2;                                      // 队列深度
constexpr int32_t REDUCE_TMP_BYTES = COLS * sizeof(float) * 4;         // 归约 API 的临时空间（可调大）

constexpr float TOLERANCE = 1e-5f;

// ============================ Kernel 侧 ============================
class KernelSoftmax {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y)
    {
        // 第 blockIdx 个核认领 [rowBase, rowBase + ROWS_PER_CORE) 这些行
        const int64_t rowBase = static_cast<int64_t>(ROWS_PER_CORE) * AscendC::GetBlockIdx();
        const int64_t elemBase = rowBase * COLS;
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x) + elemBase, ROWS_PER_CORE * COLS);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(y) + elemBase, ROWS_PER_CORE * COLS);

        // 一次搬一行（COLS 个 float = 4 KB，天然 32B 对齐）
        pipe.InitBuffer(inQueueX, BUFFER_NUM, COLS * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, COLS * sizeof(float));
        pipe.InitBuffer(expBuf, COLS * sizeof(float));      // exp 的结果
        pipe.InitBuffer(redDstBuf, 32);                     // 归约输出：1 行归约只出 1 个数，但要求 32B 对齐
        pipe.InitBuffer(redTmpBuf, REDUCE_TMP_BYTES);       // 归约 API 的共享临时空间
    }

    __aicore__ inline void Process()
    {
        for (int32_t r = 0; r < ROWS_PER_CORE; ++r) {
            CopyIn(r);
            Compute();
            CopyOut(r);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t row)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(COLS * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(xLocal, xGm[row * COLS], copyParams, padParams);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        AscendC::LocalTensor<float> expLocal = expBuf.Get<float>();
        AscendC::LocalTensor<float> redDst = redDstBuf.Get<float>();
        AscendC::LocalTensor<uint8_t> redTmp = redTmpBuf.Get<uint8_t>();

        // AR = 归约最后一维；srcShape 必须二维 {1, COLS}。
        // 最后一个参数 srcInnerPad：CANN 9.0.0 的 API 自检强制要求传 true，
        // 传 false 会直接 abort（"srcInnerPad must be set to true!"）；
        // 其语义是「最后一维已按 32B 向上对齐」，COLS=1024 个 float = 4096B 本来就对齐。
        const uint32_t srcShape[2] = {1u, static_cast<uint32_t>(COLS)};

        // ---- 1) 行最大值（数值稳定化）----
        AscendC::ReduceMax<float, AscendC::Pattern::Reduce::AR>(redDst, xLocal, redTmp, srcShape, true);
        const float rowMax = redDst.GetValue(0);  // 从 UB 直接读标量（这条在 NPU 上也很便宜）

        // ---- 2) 减 max 后取 exp ----
        AscendC::Adds(yLocal, xLocal, -rowMax, COLS);
        AscendC::Exp(expLocal, yLocal, COLS);

        // ---- 3) 行求和，再乘 1/sum ----
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR>(redDst, expLocal, redTmp, srcShape, true);
        const float rowSum = redDst.GetValue(0);
        AscendC::Muls(yLocal, expLocal, 1.0f / rowSum, COLS);

        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t row)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(COLS * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad(yGm[row * COLS], yLocal, copyParams);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> expBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> redDstBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> redTmpBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
};

__global__ __aicore__ void softmax_custom(GM_ADDR x, GM_ADDR y)
{
    KernelSoftmax op;
    op.Init(x, y);
    op.Process();
}

// ============================ Host 侧 ============================
static void FillInputs(float *x)
{
    // 造点「不自找麻烦就一定会挂」的数据：每行第一个元素给 +100 / -100，
    // 不减去行最大值时 exp(100) 直接溢出成 inf。
    for (int32_t r = 0; r < TOTAL_ROWS; ++r) {
        for (int32_t c = 0; c < COLS; ++c) {
            x[r * COLS + c] = static_cast<float>((r * 37 + c * 11) % 53 - 26) * 0.5f;
        }
        x[r * COLS] = (r % 2 == 0) ? 100.0f : -100.0f;
    }
}

static bool Verify(const float *x, const float *y)
{
    int32_t bad = 0;
    double maxErr = 0.0;
    int32_t maxIdx = -1;
    for (int32_t r = 0; r < TOTAL_ROWS; ++r) {
        // 每行用 double 重算一遍 golden softmax
        double rowMax = static_cast<double>(x[r * COLS]);
        for (int32_t c = 1; c < COLS; ++c) {
            const double v = static_cast<double>(x[r * COLS + c]);
            if (v > rowMax) {
                rowMax = v;
            }
        }
        double sum = 0.0;
        for (int32_t c = 0; c < COLS; ++c) {
            sum += std::exp(static_cast<double>(x[r * COLS + c]) - rowMax);
        }
        double rowSumOut = 0.0;
        for (int32_t c = 0; c < COLS; ++c) {
            const double golden = std::exp(static_cast<double>(x[r * COLS + c]) - rowMax) / sum;
            const double got = static_cast<double>(y[r * COLS + c]);
            rowSumOut += got;
            // NaN/inf 必须单独判：它会让 err > TOLERANCE 这种比较恒为 false，
            // 于是「数值爆炸」的算子会假 PASS。
            if (!std::isfinite(got)) {
                if (bad < 5) {
                    std::printf("  non-finite row=%d col=%d: got=%f expect=%f\n", r, c, got, golden);
                }
                ++bad;
                continue;
            }
            const double err = std::fabs(got - golden);
            if (err > maxErr) {
                maxErr = err;
                maxIdx = r * COLS + c;
            }
            if (err > TOLERANCE) {
                if (bad < 5) {
                    std::printf("  mismatch row=%d col=%d: got=%f expect=%f\n", r, c, got, golden);
                }
                ++bad;
            }
        }
        if (r < 3) {
            std::printf("  row %d: max=%+.1f sum(out)=%.6f (应为 1.0)  out[0]=%.6f  out[1]=%.6f\n", r, rowMax,
                        rowSumOut, y[r * COLS], y[r * COLS + 1]);
        }
    }
    if (bad == 0) {
        std::printf("[PASS] %d rows x %d cols, max_abs_err=%.3e (@%d)\n", TOTAL_ROWS, COLS, maxErr, maxIdx);
        return true;
    }
    std::printf("[FAIL] %d/%d mismatched, max_abs_err=%.3e (@%d)\n", bad, TOTAL_ROWS * COLS, maxErr, maxIdx);
    return false;
}

int32_t main(int32_t argc, char *argv[])
{
    (void)argc;
    (void)argv;

    constexpr size_t byteSize = static_cast<size_t>(TOTAL_ROWS) * COLS * sizeof(float);
    auto *x = static_cast<uint8_t *>(AscendC::GmAlloc(byteSize));
    auto *y = static_cast<uint8_t *>(AscendC::GmAlloc(byteSize));
    std::memset(y, 0, byteSize);
    FillInputs(reinterpret_cast<float *>(x));

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(softmax_custom, USE_CORE_NUM, x, y);

    const bool ok = Verify(reinterpret_cast<const float *>(x), reinterpret_cast<const float *>(y));

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    return ok ? 0 : 1;
}

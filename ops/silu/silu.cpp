/**
 * silu 算子 —— SiLU / Swish 激活函数：y = x * sigmoid(x) = x / (1 + e^(-x))
 *
 * 结构：Kernel 侧（Ascend C 核函数）+ Host 侧（造数据 / 调用 / 精度比对）同一文件，
 *       用 tikicpulib 的 ICPU_RUN_KF 在 x86 CPU 上真实按 8 核执行核函数。
 *
 * 在 VSCode 里的看点：
 *   1. 一个「逐元素算子」的完整骨架：Tiling 常量 -> Init 按核切 GM -> CopyIn/Compute/CopyOut
 *   2. 用 4 个基础向量 API 拼出 sigmoid：Muls(-1) -> Exp -> Adds(1) -> Div
 *   3. double buffer（BUFFER_NUM=2）：搬入/计算/搬出排队重叠；改成 1 就退化成串行
 *   4. 断点打在 KernelSilu::Compute()，每进一个核停一次，用 AscendC::GetBlockIdx() 区分是哪个核
 */
#include "kernel_operator.h"
#include "tikicpulib.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

// ============================ Tiling 常量 ============================
// 形状先写死，跑通再参数化。改 TOTAL_LENGTH 时保持
//   TOTAL_LENGTH == USE_CORE_NUM * TILE_NUM * BUFFER_NUM * TILE_LENGTH
constexpr int32_t TOTAL_LENGTH = 8 * 2048;                             // 总元素数（fp32，64 KB）
constexpr int32_t USE_CORE_NUM = 8;                                    // 用几个 AI Core（CPU 模式 = 几个线程）
constexpr int32_t BLOCK_LENGTH = TOTAL_LENGTH / USE_CORE_NUM;          // 每个核认领的元素数 = 2048
constexpr int32_t TILE_NUM = 4;                                        // 每个核把数据切成几轮
constexpr int32_t BUFFER_NUM = 2;                                      // 队列深度：2 = double buffer
constexpr int32_t TILE_LENGTH = BLOCK_LENGTH / TILE_NUM / BUFFER_NUM;  // 单次搬运元素数 = 256（1 KB，32B 对齐）

constexpr float TOLERANCE = 1e-5f;  // 精度门限（绝对误差）

// ============================ Kernel 侧 ============================
class KernelSilu {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y)
    {
        // 第 blockIdx 个核只认领自己那一段 GM：起点 = BLOCK_LENGTH * blockIdx
        const int64_t offset = static_cast<int64_t>(BLOCK_LENGTH) * AscendC::GetBlockIdx();
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x) + offset, BLOCK_LENGTH);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(y) + offset, BLOCK_LENGTH);

        // GM->UB 的输入队列（VECIN）、UB->GM 的输出队列（VECOUT），各 BUFFER_NUM 块
        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, TILE_LENGTH * sizeof(float));
        // 计算用临时 UB：一个放 -x，一个放 exp(-x)
        pipe.InitBuffer(negBuf, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(expBuf, TILE_LENGTH * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const int32_t loopCount = TILE_NUM * BUFFER_NUM;  // = 8 轮
        for (int32_t i = 0; i < loopCount; ++i) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        // GM <-> UB 统一用 DataCopyPad：尾块不满 32B 也安全（DataCopy 会越界读）
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(TILE_LENGTH * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(xLocal, xGm[progress * TILE_LENGTH], copyParams, padParams);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        (void)progress;
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        AscendC::LocalTensor<float> negLocal = negBuf.Get<float>();
        AscendC::LocalTensor<float> expLocal = expBuf.Get<float>();

        // ===== 算子逻辑：y = x / (1 + exp(-x)) =====
        AscendC::Muls(negLocal, xLocal, -1.0f, TILE_LENGTH);  // 1) -x
        AscendC::Exp(expLocal, negLocal, TILE_LENGTH);        // 2) exp(-x)
        AscendC::Adds(expLocal, expLocal, 1.0f, TILE_LENGTH); // 3) 1 + exp(-x)
        AscendC::Div(yLocal, xLocal, expLocal, TILE_LENGTH);  // 4) x / 分母
        // =========================================

        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(xLocal);  // Alloc 与 Free 必须配对，否则 UB 很快被吃光
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(TILE_LENGTH * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad(yGm[progress * TILE_LENGTH], yLocal, copyParams);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> negBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> expBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
};

// 核函数入口：ICPU_RUN_KF 在 CPU 上按 USE_CORE_NUM 个核并发调用它
__global__ __aicore__ void silu_custom(GM_ADDR x, GM_ADDR y)
{
    KernelSilu op;
    op.Init(x, y);
    op.Process();
}

// ============================ Host 侧 ============================
static void FillInputs(float *x)
{
    // 覆盖正负数、大小数：步长 0.125，范围约 [-6, 6]
    for (int32_t i = 0; i < TOTAL_LENGTH; ++i) {
        x[i] = static_cast<float>(i % 97 - 48) * 0.125f;
    }
}

static double GoldenSilu(double v)
{
    // 参考值用 double 算，避免「参考实现自己也不准」
    return v / (1.0 + std::exp(-v));
}

static bool Verify(const float *x, const float *y)
{
    int32_t bad = 0;
    double maxErr = 0.0;
    int32_t maxIdx = -1;
    for (int32_t i = 0; i < TOTAL_LENGTH; ++i) {
        const double golden = GoldenSilu(static_cast<double>(x[i]));
        const double got = static_cast<double>(y[i]);
        // NaN/inf 要让比较恒为 false，必须单独判，否则「算炸了」也会显示 [PASS]
        if (!std::isfinite(got)) {
            if (bad < 5) {
                std::printf("  non-finite @%d: x=%f got=%f expect=%f\n", i, x[i], got, golden);
            }
            ++bad;
            continue;
        }
        const double err = std::fabs(got - golden);
        if (err > maxErr) {
            maxErr = err;
            maxIdx = i;
        }
        if (err > TOLERANCE) {
            if (bad < 5) {
                std::printf("  mismatch @%d: x=%f got=%f expect=%f\n", i, x[i], got, golden);
            }
            ++bad;
        }
    }
    std::printf("  peek: ");
    for (int32_t i = 0; i < 4; ++i) {
        std::printf("silu(%+.3f)=%+.4f  ", x[i], y[i]);
    }
    std::printf("\n");
    if (bad == 0) {
        std::printf("[PASS] %d elements, max_abs_err=%.3e (@%d)\n", TOTAL_LENGTH, maxErr, maxIdx);
        return true;
    }
    std::printf("[FAIL] %d/%d mismatched, max_abs_err=%.3e (@%d)\n", bad, TOTAL_LENGTH, maxErr, maxIdx);
    return false;
}

int32_t main(int32_t argc, char *argv[])
{
    (void)argc;
    (void)argv;

    constexpr size_t byteSize = static_cast<size_t>(TOTAL_LENGTH) * sizeof(float);
    auto *x = static_cast<uint8_t *>(AscendC::GmAlloc(byteSize));
    auto *y = static_cast<uint8_t *>(AscendC::GmAlloc(byteSize));
    std::memset(y, 0, byteSize);
    FillInputs(reinterpret_cast<float *>(x));

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(silu_custom, USE_CORE_NUM, x, y);

    const bool ok = Verify(reinterpret_cast<const float *>(x), reinterpret_cast<const float *>(y));

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    return ok ? 0 : 1;
}

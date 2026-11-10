/**
 * add 算子模板 —— CPU 域孪生调试版本（无需昇腾硬件）
 *
 * 结构：Kernel 侧（核函数）+ Host 侧（数据准备 / 调用 / 精度比对）放在同一文件，
 *       用 tikicpulib 的 ICPU_RUN_KF 在 x86 CPU 上真实执行核函数。
 *
 * 修改指引：
 *   1. Compute() 里换成你的算子逻辑（AscendC::Add → Mul/Exp/...）
 *   2. 需要多输入/多输出时，照抄 inQueueY / outQueueZ 的建队-搬入-搬出模式
 *   3. 换数据类型时同步改 half/float、sizeof()、以及 Host 侧校验逻辑
 */
#include "kernel_operator.h"
#include "tikicpulib.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

// ---------------- Tiling 常量（Shape 固定，先跑通再参数化） ----------------
constexpr int32_t TOTAL_LENGTH = 8 * 2048;                             // 元素总数
constexpr int32_t USE_CORE_NUM = 8;                                    // 使用的 AI Core 数（CPU 模式下=线程数）
constexpr int32_t BLOCK_LENGTH = TOTAL_LENGTH / USE_CORE_NUM;          // 每个核处理长度
constexpr int32_t TILE_NUM = 8;                                        // 每个核切分份数
constexpr int32_t BUFFER_NUM = 2;                                      // 队列深度（double buffer）
constexpr int32_t TILE_LENGTH = BLOCK_LENGTH / TILE_NUM / BUFFER_NUM;  // 单次搬运长度

class Kerneladd {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z)
    {
        const int64_t offset = static_cast<int64_t>(BLOCK_LENGTH) * AscendC::GetBlockIdx();
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x) + offset, BLOCK_LENGTH);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(y) + offset, BLOCK_LENGTH);
        zGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(z) + offset, BLOCK_LENGTH);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_LENGTH * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const int32_t loopCount = TILE_NUM * BUFFER_NUM;
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
        AscendC::LocalTensor<float> yLocal = inQueueY.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm[progress * TILE_LENGTH], TILE_LENGTH);
        AscendC::DataCopy(yLocal, yGm[progress * TILE_LENGTH], TILE_LENGTH);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = inQueueY.DeQue<float>();
        AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();

        // >>> 算子逻辑写在这里 <<<
        AscendC::Add(zLocal, xLocal, yLocal, TILE_LENGTH);

        outQueueZ.EnQue<float>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
        AscendC::DataCopy(zGm[progress * TILE_LENGTH], zLocal, TILE_LENGTH);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    AscendC::GlobalTensor<float> zGm;
};

// 核函数入口：由 ICPU_RUN_KF 在 CPU 上按 numBlocks 并发调用
__global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z)
{
    Kerneladd op;
    op.Init(x, y, z);
    op.Process();
}

// ---------------- Host 侧 ----------------
static void FillInputs(float *x, float *y)
{
    for (int32_t i = 0; i < TOTAL_LENGTH; ++i) {
        x[i] = static_cast<float>((i % 17) - 8) * 0.5f;
        y[i] = static_cast<float>((i % 13) - 6) * 0.25f;
    }
}

static bool Verify(const float *x, const float *y, const float *z)
{
    int32_t bad = 0;
    double maxErr = 0.0;
    for (int32_t i = 0; i < TOTAL_LENGTH; ++i) {
        const float golden = x[i] + y[i];  // <<< 与 Compute() 保持一致的参考实现
        const double err = std::fabs(static_cast<double>(z[i]) - static_cast<double>(golden));
        if (err > maxErr) {
            maxErr = err;
        }
        if (err > 1e-6) {
            if (bad < 5) {
                std::printf("  mismatch @%d: got %f expect %f\n", i, z[i], golden);
            }
            ++bad;
        }
    }
    if (bad == 0) {
        std::printf("[PASS] %d elements, max_abs_err=%.3e\n", TOTAL_LENGTH, maxErr);
        return true;
    }
    std::printf("[FAIL] %d/%d mismatched, max_abs_err=%.3e\n", bad, TOTAL_LENGTH, maxErr);
    return false;
}

int32_t main(int32_t argc, char *argv[])
{
    (void)argc;
    (void)argv;

    constexpr size_t byteSize = static_cast<size_t>(TOTAL_LENGTH) * sizeof(float);
    auto *x = static_cast<uint8_t *>(AscendC::GmAlloc(byteSize));
    auto *y = static_cast<uint8_t *>(AscendC::GmAlloc(byteSize));
    auto *z = static_cast<uint8_t *>(AscendC::GmAlloc(byteSize));
    std::memset(z, 0, byteSize);
    FillInputs(reinterpret_cast<float *>(x), reinterpret_cast<float *>(y));

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(add_custom, USE_CORE_NUM, x, y, z);  // CPU 调试专用启动宏

    const bool ok = Verify(reinterpret_cast<const float *>(x), reinterpret_cast<const float *>(y),
                           reinterpret_cast<const float *>(z));

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(z);
    return ok ? 0 : 1;
}

# silu —— 逐元素激活（y = x / (1 + e⁻ˣ)）

Swish / SiLU 是 LLaMA、YOLO 系列里最常见的激活函数之一，也是**逐元素算子**的标准样本：
没有归约、没有广播，只有 GM↔UB 搬运 + 一串向量 API。想先把 Ascend C 的流水线骨架摸熟，从这个入手最省事。

## 一条命令跑通

```bash
bash run.sh test          # 编译 + 运行 + 精度校验，退出码 0 即通过
```

实测输出（8 核并发，每核一个 pid）：

```
[SUCCESS][CORE_0][pid 19878] exit success!
...
  peek: silu(-6.000)=-0.0148  silu(-5.875)=-0.0165  silu(-5.750)=-0.0182  silu(-5.625)=-0.0202
[PASS] 16384 elements, max_abs_err=3.979e-07 (@90)
```

`3.98e-07` 是 fp32 单次 `exp` + 一次除法的正常误差量级（不是 bug）。

## Tiling 长什么样

| 常量 | 值 | 含义 |
| -- | -- | -- |
| `TOTAL_LENGTH` | 16384 | 总元素数（fp32，64 KB） |
| `USE_CORE_NUM` | 8 | 起几个 AI Core（CPU 模式下 = 几个进程） |
| `BLOCK_LENGTH` | 2048 | `TOTAL_LENGTH / USE_CORE_NUM`，每核认领的元素数 |
| `TILE_NUM` | 4 | 每核把数据切几轮 |
| `BUFFER_NUM` | 2 | 队列深度，2 = double buffer |
| `TILE_LENGTH` | 256 | 单次搬运元素数 = `BLOCK_LENGTH / TILE_NUM / BUFFER_NUM`（1 KB，32B 对齐） |

不变式（改任何一个都要一起改，否则会漏算或多算）：

```
TOTAL_LENGTH == USE_CORE_NUM * TILE_NUM * BUFFER_NUM * TILE_LENGTH
```

每核 8 轮（`TILE_NUM * BUFFER_NUM`），每轮 `CopyIn → Compute → CopyOut` 三段。

## 算子逻辑只有四行

`KernelSilu::Compute()` 里：

```cpp
AscendC::Muls(negLocal, xLocal, -1.0f, TILE_LENGTH);   // -x
AscendC::Exp (expLocal, negLocal, TILE_LENGTH);       // e^(-x)
AscendC::Adds(expLocal, expLocal, 1.0f, TILE_LENGTH); // 1 + e^(-x)
AscendC::Div (yLocal, xLocal, expLocal, TILE_LENGTH); // x / 分母
```

没有 `sigmoid` 这个 API，`1 + e^(-x)` 就是自己拼的。这四个都是**基础向量 API**（`tikcfw/interface`），
参数是 `(dst, src, count)`，`count` 必须是 32B 对齐长度（fp32 即 8 的倍数）。

## 在 VSCode 里玩什么

断点打在 `KernelSilu::Compute()`（`ops/silu/silu.cpp` 里 `inline void Compute` 那一行），`F5`：

```gdb
# 断下来之后（每次进一个核停一次）
p AscendC::GetBlockIdx()        # 当前是第几个核：0~7
p progress                      # 这是该核的第几轮：0~7
p xLocal.GetValue(0)            # UB 里这一 tile 的第 0 个元素（原始输入）
p yLocal.GetValue(0)            # 计算结果
```

### 1. 看 double buffer 到底省了什么

把 `BUFFER_NUM` 从 `2` 改成 `1`，重新编译再单步：
`loopCount` 从 8 变 4、`TILE_LENGTH` 从 256 变 512 —— 搬运次数减半但每块更大，队列退化成"搬一块算一块"。
真机上这就是搬运与计算能不能重叠的分界线。

### 2. 把核数「偷偷」改小，看它漏算

`ICPU_RUN_KF` 那一行改成 `ICPU_RUN_KF(silu_custom, 4, x, y)`（只起 4 核）：
每核仍按 8 核的 `BLOCK_LENGTH=2048` 切数据，于是**后一半输出恒为 0**，
`[FAIL] 8192/16384 mismatched`。这说明「一个核干多少活」是 **kernel 里的 offset 算术**决定的，
不是启动参数决定的 —— 这是昇腾多核算子最常见的错法。

### 3. 把搬运改成不对齐长度

`TILE_LENGTH` 改 `257`（`257*4=1028` 字节，不是 32B 倍数）：
`DataCopyPad` 仍能跑（它会补齐），但如果把 `DataCopyPad` 换成 `DataCopy` 就会越界读、结果随机。
模板统一用 `DataCopyPad` 就是为了不掉这个坑。

### 4. 精度门限实验

`TOLERANCE` 从 `1e-5f` 改成 `1e-8f`，重新跑 → 会在 `3.98e-07` 那批点上 FAIL。
这不是实现错了，而是 fp32 的 `exp` 本身只有约 1e-7 量级的相对误差；
Host 侧参考值用 `double` 算（`GoldenSilu`），所以误差全部来自 kernel 侧，边界清晰。

### 5. 顺手看看核函数是怎么被调起来的

`ICPU_RUN_KF(silu_custom, 8, x, y)` → CPU 侧 `fork` 8 个子进程，各自调一次
`__global__ __aicore__ void silu_custom(...)`。
所以 `launch.json` 里配了 `set follow-fork-mode child`，否则断点只会停在父进程。

## UB 占用

`inQueueX(2×1KB) + outQueueY(2×1KB) + negBuf(1KB) + expBuf(1KB)` ≈ 6 KB，
910B 单核 UB 是 192 KB，所以这个算子有巨大余量可以加中间 buffer。

## 坑（本算子踩过的）

- `AllocTensor` / `FreeTensor` 必须配对，`DeQue` 出来的 tensor 用完要还队列，否则 UB 很快耗尽。
- 向量 API 的 dst 别和 src 完全相同（in-place 在部分实现上有重叠断言），模板统一另开 `negBuf`。
- `Exp` 没有 `cast` 版本可用（CANN 9.0.0 这套 tikicpulib 头文件里找不到 `Cast`），所以本算子只做 fp32。

# softmax —— 按行归约 + 广播（y[r,c] = e^(x[r,c]-max_r) / Σc e^(x[r,c]-max_r)）

如果 `silu` 教的是「逐元素流水线」，`softmax` 教的是**归约 + 标量回读 + 广播**：
一行的结果依赖整行的最大值和总和，所以必须先归约出一个标量，再拿它去算 1024 个输出。
这类结构（LayerNorm、RMSNorm、Softmax、LogSoftmax）是昇腾上最容易写出低效代码、也最值得单步看数据流的一类。

## 一条命令跑通

```bash
bash run.sh test
```

实测输出：

```
  row 0: max=+100.0 sum(out)=1.000000 (应为 1.0)  out[0]=1.000000  out[1]=0.000000
  row 1: max=+13.0 sum(out)=1.000000 (应为 1.0)  out[0]=0.000000  out[1]=0.002717
  row 2: max=+100.0 sum(out)=1.000000 (应为 1.0)  out[0]=1.000000  out[1]=0.000000
[PASS] 64 rows x 1024 cols, max_abs_err=1.998e-09 (@3110)
```

输入是特意造的「不减去最大值就一定挂」的数据：偶数行首元素 `+100`，奇数行首元素 `-100`。
`sum(out)=1.000000` 是 softmax 最直观的自检量。

## Tiling

| 常量 | 值 | 含义 |
| -- | -- | -- |
| `TOTAL_ROWS` | 64 | 批大小 |
| `COLS` | 1024 | 每行元素数，正好一次搬完（4 KB，32B 对齐） |
| `USE_CORE_NUM` | 8 | 起几个核 |
| `ROWS_PER_CORE` | 8 | 每核认领 8 行 |
| `BUFFER_NUM` | 2 | 队列深度 |
| `REDUCE_TMP_BYTES` | 16 KB | 归约 API 的共享临时空间 |

切分按**行**来，不切列：softmax 的归约轴是行内，把一行切开就得做跨核 reduce，那是另一个难度。
每核 8 行，逐行跑一条 `CopyIn → Compute → CopyOut`。

## 计算五步

`KernelSoftmax::Compute()`：

```cpp
const uint32_t srcShape[2] = {1u, COLS};

AscendC::ReduceMax<float, Pattern::Reduce::AR>(redDst, xLocal,  redTmp, srcShape, true);  // 1) 行最大值
const float rowMax = redDst.GetValue(0);                                                  // 2) 从 UB 读标量
AscendC::Adds(yLocal,  xLocal,   -rowMax, COLS);                                          // 3) 减最大值
AscendC::Exp (expLocal, yLocal,   COLS);                                                  //    e^(x-max)
AscendC::ReduceSum<float, Pattern::Reduce::AR>(redDst, expLocal, redTmp, srcShape, true); // 4) 行和
const float rowSum = redDst.GetValue(0);
AscendC::Muls(yLocal, expLocal, 1.0f / rowSum, COLS);                                     // 5) 归一化
```

四个容易踩的点：

1. **pattern 只能 AR / RA**。`Pattern::Reduce::R`（全归约）在这个实现里是 `static_assert`
   直接编译失败：`failed to check the reduce pattern, it only supports AR/RA pattern!`。
   AR = 归约最后一维（每行出一个结果）；RA = 归约倒数第二维。
2. **`srcShape` 必须二维** `{rows, cols}`。这里一次只归约 1 行，所以是 `{1, COLS}`。
3. **`srcInnerPad` 必须传 `true`**。CANN 9.0.0 的 API 自检里是硬要求，传 `false` 会在运行时
   abort：`[ReduceMax] srcInnerPad must be set to true!`（连编译都过了，坑在运行期）。
4. **归约结果要 `GetValue(0)` 读回标量**，再作为 `Adds` / `Muls` 的标量参数广播回 1024 个元素。
   高阶写法是 `Brcb` 把标量广播成向量，但对 1024 长度没收益，这里先用最直白的。

## 在 VSCode 里玩什么

断点打在 `KernelSoftmax::Compute()` 的 `rowMax` / `rowSum` 两行，`F5`：

```gdb
p AscendC::GetBlockIdx()   # 0~7，第几个核
p rowMax                   # 这一行的最大值（偶数行应看到 100，奇数行约 13）
p rowSum                   # 归一化分母，应该 ≥ 1 且有限
p yLocal.GetValue(0)       # 归一化后的第一个输出
p redDst.GetValue(0)       # redDst 就是 rowMax/rowSum 的来源，可以对照
```

因为每核 8 行、共 8 核，`F5` 会停 **64 次**。想少停几次，可以把断点条件设成
`AscendC::GetBlockIdx() == 0 && row == 0`（VSCode 断点上右键 → Edit Breakpoint → Expression）。

### 1. 亲手把「减最大值」拆掉，看它怎么挂

把这两行改掉，等于跳过数值稳定化：

```cpp
// AscendC::ReduceMax<float, Pattern::Reduce::AR>(redDst, xLocal, redTmp, srcShape, true);
const float rowMax = 0.0f;                      // 原来是 redDst.GetValue(0)
```

结果（实测）：

```
row 0: max=+100.0 sum(out)=nan (应为 1.0)  out[0]=nan  out[1]=0.000000
row 1: max=+13.0 sum(out)=1.000000 (应为 1.0)  out[0]=0.000000  out[1]=0.002717
row 2: max=+100.0 sum(out)=nan (应为 1.0)  out[0]=nan  out[1]=0.000000
[FAIL] 32/65536 mismatched, max_abs_err=1.939e-09 (@35886)
```

偶数行（首元素 `+100`）里，`exp(100)` 溢出成 `inf`，行和也是 `inf`，`1/inf=0`，
再 `inf * 0 = NaN` —— 溢出元素自己变 NaN，同行其它元素变 0；奇数行（最大值只有 13）完全不受影响。
注意只有 **32** 个元素被判错（32 个偶数行 × 1 个溢出元素）：同行其它元素算出来是 0，
而它们的 golden 本来也≈0，误差反而没超门限 —— 换句话说，光看 mismatch 数量会低估问题规模，
还得看那行 `sum(out)=nan`。
**这就是所有框架的 softmax 里都要塞一个 `-max` 的全部原因**，它不是优化项，是正确性项。

> 顺便暴露了一个更阴的坑：这一版 `Verify()` 最初只写 `if (err > TOLERANCE)`，
> 而 `NaN > x` 恒为 `false`，于是这个已经算炸的算子竟然报了 **`[PASS]`**。
> 现在 `Verify()` 里加了 `std::isfinite(got)` 单独判非有限值，所以上面看到的才是 FAIL。
> 精度比对代码自己也会骗人 —— 这条比 softmax 本身值钱。

### 2. 把 pattern 改错，看编译期报错

`Pattern::Reduce::AR` 改成 `Pattern::Reduce::R`，重新编译：

```
error: static assertion failed: failed to check the reduce pattern,
       it only supports AR/RA pattern!
```

这是本算子唯一「编译期就拦住你」的错误，值得亲眼见一次。

### 3. 把 `srcInnerPad` 改回 false，看运行期 abort

改成 `false`，编译能过，运行时立刻：

```
[ERROR][Block_4][AIV_0][.../reduce_check.h:89][CheckSrcInnerPad][19523]
       [ReduceMax] srcInnerPad must be set to true!
SIGABRT Signal (Abort Signal from abort) catched
```

配合 `journalctl`／backtrace 能一路看到 `softmax.cpp` 的行号。API 自检的报错信息是全套 CANN 里最好用的。

### 4. 一次归约 8 行（练习）

现在是「逐行 `ReduceMax({1,1024})`」，8 行就调 8 次。
把 shape 换成 `{8, 1024}`，一次就能拿到 8 个行的最大值（`redDst[0..7]`），省 7 次 API 调用；
但接下来要把这 8 个标量广播回 8×1024 —— 得用 `Brcb` 或 `Copy` 逐行搬，复杂度立刻上来。
这就是昇腾上「归约容易、广播贵」的典型权衡，可以自己试一版比一比。

### 5. 改形状

`COLS` 改成 512 或 2048（保持 32B 对齐即可），`TOTAL_ROWS` 保持 64，都能直接跑通。
`ROWS_PER_CORE = TOTAL_ROWS / USE_CORE_NUM`，所以 `TOTAL_ROWS` 必须是 `USE_CORE_NUM` 的整数倍。

## UB 占用

`inQueueX(2×4KB) + outQueueY(2×4KB) + expBuf(4KB) + redDstBuf(32B) + redTmpBuf(16KB)` ≈ 36 KB。
`REDUCE_TMP_BYTES` 是拍出来的够用值；真机上应该用 `GetReduceMaxMinTmpSize()` 之类的接口算，
这里为了代码短，先固定。

## 坑（本算子踩过的）

- `srcInnerPad=false` 编译通过、运行 abort，错误信息在 `reduce_check.h:89`。
- `pattern=R` 编译期 `static_assert`，错误指向 `reduce_max_v220_impl.h:42`。
- `redDst` 只需要 1 个数，但必须按 **32B（8 个 float）** 申请，否则后续访存会越界/对齐异常。
- 归约 API 需要一块**独立**的 `sharedTmpBuffer`（`TPosition::VECCALC`），不能和输入/输出共用。

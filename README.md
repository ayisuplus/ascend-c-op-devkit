# 昇腾算子开发与测试环境

一台 **没有昇腾 NPU** 的 Windows 工作站，也能完整地写 Ascend C 算子、编译、跑核函数、看精度、
用 gdb 单步进 kernel —— 靠的是 CANN 官方的 **CPU 域孪生调试**（`tikicpulib`）。
本仓库就是把这条链路固化下来：一条命令起环境，一条命令建算子，一条命令跑精度。

```
Windows 11 ──► WSL2 发行版 AscendCANN (Ubuntu 22.04) ──► CANN 9.0.0 Toolkit
                                                              │
                        ops/<算子>/ ──cmake+g++──► build/<算子> ──► 在 x86 CPU 上按 8 核真实执行 Ascend C 核函数
                                                              └──► [PASS] max_abs_err=0.000e+00
```

## 快速开始

```powershell
# 1) 环境自检（应全部 PASS）
.\scripts\ascend.ps1 doctor

# 2) 直接跑现成的两个算子（编译 + 运行 + 精度校验）
.\scripts\ascend.ps1 test silu
.\scripts\ascend.ps1 test softmax

# 3) 全部算子一起跑（add + silu + softmax，可当门禁）
.\scripts\ascend.ps1 test

# 4) 想自己加第三个：从模板新建 ops/myop/ 再把 Compute() 改掉
.\scripts\ascend.ps1 new myop
.\scripts\ascend.ps1 test myop
```

首次在别的机器上重建环境（下 rootfs、装依赖、装 CANN，约 10 分钟）：

```powershell
.\scripts\ascend.ps1 bootstrap
```

进 WSL 手动折腾：`.\scripts\ascend.ps1 shell`（工作区在 `/work`，即 `E:\projects\昇腾社区`）。

## 开箱可玩的两个算子

`ops/` 下已经放好两个**跑通精度、带详细注释、专门为单步调试准备**的算子，不用从模板改起：

| 算子 | 类型 | 数学 | 实测精度 | 学什么 |
| -- | -- | -- | -- | -- |
| [silu](ops/silu/README.md) | 逐元素 | `y = x / (1 + e⁻ˣ)` | `max_abs_err=3.98e-07` | 8 核切分、double buffer、GM↔UB 搬运、基础向量 API 拼激活函数 |
| [softmax](ops/softmax/README.md) | 按行归约 | `y[r,c] = e^(x[r,c]-max_r) / Σc e^(x[r,c]-max_r)` | `max_abs_err=2.00e-09`（`sum(out)=1`） | `ReduceMax/ReduceSum` 的 `pattern`、`GetValue` 取标量、标量广播、数值稳定化 |

```powershell
.\scripts\ascend.ps1 test silu        # 或 bash ops/silu/run.sh test（在 WSL 里）
.\scripts\ascend.ps1 test softmax
.\scripts\ascend.ps1 test             # 三个算子（add + silu + softmax）一起回归
```

两个算子都刻意留了「动手改一下就能看到差别」的实验，写在各自的 README 里，比如：

- **silu**：只起 4 核 → 后一半输出恒为 0，`[FAIL] 8192/16384`（核数由 `GetBlockIdx` 的 offset 算术决定，不是启动参数）；
  把 `BUFFER_NUM` 从 2 改 1 → 单步看队列退化成「搬一块算一块」。
- **softmax**：把 `ReduceMax` 换成 `rowMax = 0`（跳过数值稳定化）→ 偶数行 `exp(100)` 溢出、输出 NaN。
  这个实验还顺带暴露了精度比对的一个陷阱：`NaN > TOLERANCE` 恒为 `false`，最早的 `Verify()`
  会给算炸的算子报 `[PASS]`；现在已加 `std::isfinite()` 单独判非有限值。

想在 VSCode 里点着玩：左下的 `><` 连到 `AscendCANN`、打开 `/work`，打开 `ops/silu/silu.cpp`，
`Ctrl+Shift+B` 编译、`F5` 调试，断点下在 `KernelSilu::Compute()` / `KernelSoftmax::Compute()` 里。

## 在 VSCode 里开发（WSL 远程）

已装好 `ms-vscode-remote.remote-wsl`，并已把 `vscode-server`（`/root/.vscode-server`）与
`ms-vscode.cpptools` 装进 `AscendCANN` 发行版，开箱可用：

```powershell
code --remote wsl+AscendCANN /work
```

或在 VSCode 里：左下角 `><` → **Connect to WSL** → 选 `AscendCANN` → 打开 `/work`。
窗口左下角显示 `WSL: AscendCANN` 即为远程模式（**必须**在这种模式下编译/调试；
直接在 Windows 打开 `E:\projects\昇腾社区` 只能读代码，编译和 gdb 都在 WSL 里）。

配置在 [.vscode/](.vscode/)：

| 文件 | 作用 |
| -- | -- |
| `tasks.json` | 7 个任务：编译当前算子（`Ctrl+Shift+B` 默认）、运行+精度校验、编译指定算子、新建算子、全量回归、环境自检、清理 |
| `launch.json` | cppdbg + gdb：`F5` 直接调试当前打开的算子（先自动编译），或选「调试指定算子」 |
| `c_cpp_properties.json` | IntelliSense 指向 CANN 9.0.0 的 tikcfw / tikicpulib 头文件 |
| `extensions.json` | 推荐扩展（remote-wsl / cpptools / cmake-tools） |

调试要点：

- `ICPU_RUN_KF` 为**每个核 fork 一个子进程**（输出里每核 pid 不同），所以 `launch.json` 里带了
  `set follow-fork-mode child` + `set detach-on-fork off`；断点下在 `KernelXxx::Compute()` 里，
  `F5` 后每进入一个核会停一次，可用 `AscendC::GetBlockIdx()` 区分。
- 终端里如果 `echo $ASCEND_HOME_PATH` 为空，重开一个终端即可（登录 shell 会加载
  `/etc/profile.d/ascend-cann.sh`）；`run.sh` 自己也会 source，不依赖终端环境。

## 目录结构

```
├── env\
│   ├── cann.env            # 唯一配置源：CANN 版本/下载地址、WSL 发行版、工作区映射、默认 SoC
│   ├── check_env.sh        # 环境自检（主机/工具链/CANN/ASC/CPU 调试库/设备）
│   └── wsl\bootstrap.sh    # 幂等引导：apt 源→依赖→静默装 CANN→写 /etc/profile.d
├── scripts\
│   ├── ascend.ps1          # Windows 侧统一入口：bootstrap|doctor|new|build|run|test|shell
│   └── new_op.sh           # 从 templates/op 生成 ops/<name> 并替换 __OP_NAME__
├── templates\op\           # 算子模板：核函数+Host+精度校验单文件，自带 Add 示例
├── ops\add\                # 模板生成的样例算子（已跑通，可作为参考）
├── ops\silu\               # 逐元素算子：y = x/(1+e^-x)，练 8 核切分 + double buffer
├── ops\softmax\            # 按行归约算子：练 ReduceMax/ReduceSum + 标量广播 + 数值稳定化
├── tests\run_all.sh        # 批量跑所有算子，汇总 PASS/FAIL
├── .vscode\                # WSL 远程开发配置：任务、gdb 调试、IntelliSense
├── third_party\asc-devkit\ # 官方样例仓库（gitcode，branch 9.0.0，sparse: examples/docs/getting-started）
└── 华为昇腾社区与算子开发生态调研报告.md
```

## 环境细节

| 组件 | 版本 / 位置 |
| -- | -- |
| WSL 发行版 | `AscendCANN`，Ubuntu 22.04.5 LTS，`wsl --import` 自官方 rootfs（不占 Windows 磁盘以外的空间） |
| CANN Toolkit | 9.0.0，装在 `/usr/local/Ascend/cann-9.0.0`，`cann` 软链指向它 |
| 编译器 | gcc/g++ 11.4.0、cmake 3.22.1、gdb 12.1（≥12.1 才能调试核函数） |
| 默认 SoC | `Ascend910B3`（对应 `dav-2201`，即 Atlas A2/A3）；改 `env/cann.env` 或 `SOC_VERSION=xxx` |
| CPU 调试库 | `$ASCEND_HOME_PATH/tools/tikicpulib`（`tikicpulib::<soc>` 由 CMake 导入） |

为什么模板是 `.cpp` 而不是官方的 `.asc`：`.asc` 是 ASC 语言（bisheng 编译器）的源码格式，
实测 CANN 9.0.0 下 `-DCMAKE_ASC_RUN_MODE=cpu` 走不通（`trap` 未声明 / `kernel call to non-global function`）；
而 `tikicpulib` 路线只需要普通 C++ 编译器，`ICPU_RUN_KF` 会在 CPU 上按核并发调起核函数，
这是当前唯一实测可用的无硬件执行路径。详见 [templates/op/README.md](templates/op/README.md)。

## 项目内 Skill（仅本项目加载）

来自 [ascend-ai-coding/awesome-ascend-skills](https://github.com/ascend-ai-coding/awesome-ascend-skills)
的 `base` + `ops` 两个包，共 12 项技能，保存在本项目的两个入口：

- `.agents/skills/`：供 Codex 及读取该项目目录的兼容客户端使用。
- `.claude/skills/`：供 Claude Code 在本项目中使用。

在 `E:\projects\昇腾社区` 开启新会话即可加载；其他项目不再从用户全局目录加载这组技能。

| Skill | 用途 |
| -- | -- |
| `ascendc` | **算子开发主线**：Ascend C 算子设计/生成/编译/精度调优/性能评测的端到端流程 |
| `ascend-opplugin` | 算子接入 PyTorch / torch_npu，Host 侧注册与 pybind 联调 |
| `triton-ascend-migration` | 把 CUDA Triton 或 PyTorch 算子迁移成 Triton-Ascend |
| `npu-op-benchmark` | 单算子性能基准（需真实 NPU） |
| `ascend-dmi` / `npu-smi` | NPU 卡状态、带宽、算力、故障诊断（需真实 NPU） |
| `ascend-docker` / `npu-docker-launcher` | 带 NPU 设备映射的容器环境 |
| `ascend-avi-vnpu` | vNPU 虚拟化实例管理（需真实 NPU） |
| `remote-npu-test` / `remote-server-guide` | 连远程 NPU 服务器跑推理/测试 |
| `torch_npu` | torch_npu 安装与使用参考 |

维护或升级时只更新上述项目目录。此前带 `-g -a '*'` 的全局安装命令已经移除，避免再次向所有客户端分发。

2026-10-25 的迁移范围、源路径和内容校验清单见 [迁移记录](docs/skill-migration-20261025/README.md)。

## 能力边界

- **能**：写核函数、编译、CPU 上真实执行、gdb 单步调试、精度比对、Tiling 逻辑验证、多核并发行为观察。
- **不能**：测真实性能（CPU 上无 Cube/Vector 硬件时序）、跑 `npu-smi`/`ascend-dmi`、验证内存带宽、
  真实 SoC 指令集差异。这类需求走免费的 [CANNLab 云开发环境](https://www.hiascend.com/developer/ascendhub)
  （Atlas A2 真机，约 100 卡时），或用 `remote-npu-test` skill 连远程 NPU 服务器。
- CPU 模式打印 `[TmSim]: Run in serial mode.` 属正常：`tikicpulib` 在当前版本按串行方式模拟多核，
  每核一行 `[SUCCESS][CORE_n] exit success!` 表示该核正常跑完。

## 常见坑（已踩过）

| 现象 | 原因 / 解法 |
| -- | -- |
| `set_env.sh: LD_LIBRARY_PATH: unbound variable` | 脚本用了 `set -u`；source CANN 环境时不要开 `-u` |
| 运行时报 `libmmpa.so: cannot open shared object file` | 忘了 source `set_env.sh`（本模板的 `run.sh` 已在 run/test 前自动加载） |
| `tikicpulib::` 目标不存在 | 没传 `-Dsoc_version=Ascend910B3`（或版本不在 `tikicpulib-config.cmake` 的产品表里） |
| `#include "kernel_operator.h"` 找不到 | 同上：include 目录由 `tikicpulib` 目标带出来，目标没解析成功就没有 |
| `wsl -d X -- bash -c '...$VAR...'` 变量丢了 | Windows→WSL 的内联命令会吞 `$`，把逻辑写进 `.sh` 文件再执行 |
| VSCode 装扩展报 `OPENSSL_internal:BAD_DECRYPT` / `bad record mac` | 代理对 marketplace 的 TLS 有干扰：安装扩展前先 `$env:HTTP_PROXY=''; $env:HTTPS_PROXY=''` 直连 | 

# 算子模板（CPU 域孪生调试）

由 `scripts/new_op.sh <名字>` 复制到 `ops/<名字>/`，并把 `__OP_NAME__` 替换成真实算子名。
模板本身已经是**可直接编译、可直接跑通精度校验**的 Add 算子（fp32，shape 8×2048），
改算子只要动三处：`Compute()` 里的计算 API、Host 侧 `Verify()` 的参考实现、Tiling 常量。

## 为什么是这套写法

昇腾官方提供两条“没有硬件也能跑 Ascend C 核函数”的路线，本项目选的是**已实测可用**的那条：

| 路线 | 入口 | 本机实测 |
| -- | -- | -- |
| **tikicpulib（本项目采用）** | `find_package(tikicpulib)` + `ICPU_RUN_KF(...)`，纯 g++ 编译，核函数在 x86 CPU 上按核并发执行 | ✅ 可编译可运行，8 核 `[SUCCESS] ... exit success!`，支持 gdb 单步进核函数 |
| ASC 语言 `CMAKE_ASC_RUN_MODE=cpu` | `find_package(ASC)` + `project(... LANGUAGES ASC)` + `<<<>>>` 直调 | ❌ CANN 9.0.0 下 cpu 模式编译报 `trap` 未声明 / `kernel call to non-global function`，官方 example 也未提供该组合 |

所以模板用 `.cpp` 而不是 `.asc`：`.asc` 是给 ASC 编译器（bisheng）的，`tikicpulib` 只需普通 C++ 编译器。
将来接上真机，再按官方 `hello_world_npu` 的 CMake 另开一套 `npu` 模式即可。

## 目录内容

```
ops/<name>/
├── <name>.cpp        # 核函数 + Host 侧调用 + 精度校验，单文件
├── CMakeLists.txt    # find_package(tikicpulib) + 链接 ascendcl / tikicpulib::<soc>
├── run.sh            # build | run | test | clean
└── README.md         # 本文件
```

## 常用命令

```bash
bash run.sh build        # 编译，产物 build/<name>
bash run.sh run          # 只运行
bash run.sh test         # 运行 + 校验，退出码 0 表示通过
bash run.sh clean
SOC_VERSION=Ascend910B3 bash run.sh build   # 换 SoC（默认 Ascend910B3 ↔ dav-2201）
```

从 Windows 侧：`.\scripts\ascend.ps1 test <op>`，或 `.\scripts\ascend.ps1 shell` 进 WSL 手动跑。

## gdb 断点调试

```bash
bash run.sh build
cd build
gdb --args ./<name>
(gdb) break Kernel<name>::Compute
(gdb) run
```

`ICPU_RUN_KF` 在 CPU 上按 `USE_CORE_NUM` 起线程，gdb 里可用 `set follow-fork-mode child`
或直接在 `Compute` 上打断点观察每核数据。

## 改算子时的坑（昇腾社区高频问题）

- 搬运长度必须是 32 字节对齐（fp32 即 8 的倍数），否则 `DataCopy` 结果异常。
- `AllocTensor` 必须配对 `FreeTensor`，`DeQue` 出来的 tensor 用完要还队列。
- 多流水线并行时注意 `PipeBarrier<PIPE_ALL>()`，否则出现数据竞争型精度错误。
- fp16 大数吃小数：累加类算子优先 fp32 累加。

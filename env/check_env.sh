#!/usr/bin/env bash
# ============================================================================
# 昇腾算子模拟环境 · 自检
#   bash /work/env/check_env.sh
# 逐项检查系统依赖、CANN 安装、Ascend C 工具链、CPU 域调试依赖是否齐全。
# 退出码 0 = 全部通过；1 = 有 FAIL。
# ============================================================================
set -uo pipefail

CONF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$CONF_DIR/cann.env"
. /etc/profile.d/ascend-cann.sh 2>/dev/null || true

FAIL=0
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    G='\033[1;32m'; Y='\033[1;33m'; R='\033[1;31m'; D='\033[0;37m'; N='\033[0m'
else
    G=''; Y=''; R=''; D=''; N=''
fi

ok()   { printf "  ${G}PASS${N}  %-34s ${D}%s${N}\n" "$1" "${2:-}"; }
bad()  { printf "  ${R}FAIL${N}  %-34s ${D}%s${N}\n" "$1" "${2:-}"; FAIL=1; }
warn() { printf "  ${Y}WARN${N}  %-34s ${D}%s${N}\n" "$1" "${2:-}"; }

have() { command -v "$1" >/dev/null 2>&1; }

section() { printf "\n\033[1m== %s\033[0m\n" "$1"; }

section "主机"
printf "  %-38s %s\n" "OS" "$(grep PRETTY_NAME /etc/os-release | cut -d= -f2- | tr -d '\"')"
printf "  %-38s %s\n" "Kernel" "$(uname -r)"
printf "  %-38s %s\n" "CPU / 内存" "$(nproc) 核 / $(free -g | awk '/^Mem:/{print $2}') GiB"
printf "  %-38s %s\n" "工作区" "$(readlink -f /work 2>/dev/null || echo '<未建立 /work 软链>')"
printf "  %-38s %s\n" "磁盘剩余（工作区所在盘）" "$(df -h /work 2>/dev/null | awk 'NR==2{print $4}' || echo '?')"

section "编译与调试工具"
for t in gcc g++ cmake make gdb python3; do
    if have "$t"; then ok "$t" "$($t --version 2>&1 | head -1)"; else bad "$t" "未安装"; fi
done
if have gdb; then
    gv=$(gdb --version | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
    if [[ -n "${gv:-}" ]] && awk "BEGIN{exit !($gv >= 12.1)}"; then
        ok "gdb >= 12.1（核函数断点调试要求）" "版本 $gv"
    else
        warn "gdb < 12.1" "CPU Debug 断点调试可能异常，版本 ${gv:-未知}"
    fi
fi

section "CANN Toolkit"
if [[ -n "${ASCEND_HOME_PATH:-}" && -d "${ASCEND_HOME_PATH}" ]]; then
    ok "ASCEND_HOME_PATH" "$ASCEND_HOME_PATH"
else
    bad "ASCEND_HOME_PATH" "未设置或目录不存在（先跑 bootstrap）"
fi

TOOLKIT_SH="${ASCEND_HOME_PATH:-}/set_env.sh"
if [[ -f "$TOOLKIT_SH" ]]; then
    ok "set_env.sh" "$TOOLKIT_SH"
elif [[ -f "$CANN_INSTALL_PATH/ascend-toolkit/set_env.sh" ]]; then
    ok "set_env.sh" "$CANN_INSTALL_PATH/ascend-toolkit/set_env.sh"
else
    bad "set_env.sh" "缺失：$TOOLKIT_SH"
fi

INFO_FILE=""
for f in "$CANN_INSTALL_PATH/ascend_toolkit_install.info" \
         "${ASCEND_HOME_PATH:-/nonexistent}/x86_64-linux/ascend_toolkit_install.info" \
         "/etc/Ascend/ascend_cann_install.info"; do
    [[ -f "$f" ]] && { INFO_FILE="$f"; break; }
done
if [[ -n "$INFO_FILE" ]]; then
    ver=$(grep -i 'version' "$INFO_FILE" | head -1 | cut -d= -f2)
    ok "CANN 版本信息" "${ver:-未知}（期望 ${CANN_VERSION}，见 $INFO_FILE）"
else
    warn "ascend_toolkit_install.info" "未找到，无法确认版本"
fi

section "Ascend C 工具链"
found_any=0
for t in ascendc bisheng ccec msobjdump msopgen; do
    if have "$t"; then ok "$t" "$(command -v "$t")"; found_any=1; fi
done
[[ $found_any -eq 0 ]] && bad "Ascend C 编译器" "ascendc/bisheng/ccec 都不在 PATH 中"

if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
    # CANN 9.x：ASC 语言支持由 tikcpp/ascendc_kernel_cmake + lib64/cmake/asc-config.cmake 提供
    asc_cmake=$(find "${ASCEND_HOME_PATH}" -maxdepth 6 \
                  \( -name 'asc-config.cmake' -o -name 'ASCConfig.cmake' \) 2>/dev/null | head -1)
    if [[ -n "$asc_cmake" ]]; then ok "CMake ASC 包（find_package(ASC)）" "$asc_cmake"
    else bad "CMake ASC 包" "未找到 asc-config.cmake，find_package(ASC) 会失败"; fi

    asc_lang=$(find "${ASCEND_HOME_PATH}" -maxdepth 6 -name 'FindASC.cmake' 2>/dev/null | head -1)
    if [[ -n "$asc_lang" ]]; then ok "ASC 编译语言支持" "$asc_lang"
    else bad "ASC 编译语言支持" "未找到 FindASC.cmake，project(... LANGUAGES ASC) 会失败"; fi

    cpulib=$(find "${ASCEND_HOME_PATH}" -maxdepth 6 -type d -name 'tikicpulib' 2>/dev/null | head -1)
    if [[ -n "$cpulib" ]]; then ok "CPU 域执行库（ticpulib）" "$cpulib"
    else warn "CPU 域执行库" "未找到 tikicpulib，cpu 模式可能不可用"; fi
fi

section "NPU 设备（模拟环境预期为空）"
if have npu-smi; then
    if npu-smi info >/dev/null 2>&1; then
        warn "检测到 NPU" "本环境按无卡设计，有卡可直接切到 npu 模式"
    else
        ok "无 NPU（符合模拟环境预期）" "CPU/sim 模式可用"
    fi
else
    ok "无 NPU 工具（符合模拟环境预期）" "CPU/sim 模式可用"
fi

printf "\n"
if [[ $FAIL -eq 0 ]]; then
    printf "${G}自检通过${N}\n"
else
    printf "${R}自检存在失败项${N}，修复后重跑： bash /work/env/check_env.sh\n"
fi
exit $FAIL

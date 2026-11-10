#!/usr/bin/env bash
# ============================================================================
# 单个算子的构建 / 运行 / 精度校验入口
#
#   bash run.sh build          # 编译（cpu 模式，默认）
#   bash run.sh run            # 运行
#   bash run.sh test           # 运行并校验精度（退出码非 0 即失败）
#   bash run.sh clean          # 删除 build 目录
#   bash run.sh build npu      # 预留给真机：当前 CANN 环境未接 NPU，会直接报错退出
#
# 环境变量：SOC_VERSION（默认 Ascend910B3）、CANN_HOME（默认从 set_env.sh 推导）
# ============================================================================
set -eo pipefail

ACTION="${1:-build}"
MODE="${2:-cpu}"

OP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OP_NAME="$(basename "$OP_DIR")"
SOC_VERSION="${SOC_VERSION:-Ascend910B3}"
BUILD_DIR="$OP_DIR/build"

# 非交互输出（管道/CI 日志）关闭颜色，避免 ANSI 转义污染
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    G='\033[1;32m'; R='\033[1;31m'; Y='\033[1;33m'; N='\033[0m'
else
    G=''; R=''; Y=''; N=''
fi
info() { printf "${Y}[%s]${N} %s\n" "$OP_NAME" "$*"; }
ok()   { printf "${G}[%s] %s${N}\n" "$OP_NAME" "$*"; }
die()  { printf "${R}[%s] %s${N}\n" "$OP_NAME" "$*" >&2; exit 1; }

# ---- CANN 环境（每次命令前重新 source，不依赖交互 shell 状态） ----
load_cann() {
    if [ -z "${ASCEND_HOME_PATH:-}" ]; then
        # shellcheck disable=SC1091
        [ -f /etc/profile.d/ascend-cann.sh ] && . /etc/profile.d/ascend-cann.sh >/dev/null 2>&1 || true
    fi
    if [ -z "${ASCEND_HOME_PATH:-}" ]; then
        local sh
        sh=$(ls -d /usr/local/Ascend/cann*/set_env.sh 2>/dev/null | head -1)
        [ -n "$sh" ] && . "$sh" >/dev/null 2>&1 || true
    fi
    [ -n "${ASCEND_HOME_PATH:-}" ] || die "未找到 CANN 环境，请先运行 scripts/ascend.ps1 bootstrap"
}

case "$MODE" in
    cpu) ;;
    sim|npu)
        die "当前模拟环境只支持 cpu 模式（tikicpulib）。$MODE 模式需要 ASC 编译器 + 对应 npu-arch，尚未在此环境验证。"
        ;;
    *) die "未知模式: $MODE（可选 cpu）" ;;
esac

case "$ACTION" in
    clean)
        rm -rf "$BUILD_DIR"
        ok "已清理 $BUILD_DIR"
        ;;
    build)
        load_cann
        info "编译（cpu / soc_version=$SOC_VERSION）"
        cmake -S "$OP_DIR" -B "$BUILD_DIR" -Dsoc_version="$SOC_VERSION" >"$OP_DIR/build.log" 2>&1 \
            || { tail -30 "$OP_DIR/build.log"; die "cmake 配置失败（完整日志 build.log）"; }
        cmake --build "$BUILD_DIR" -j"$(nproc)" >>"$OP_DIR/build.log" 2>&1 \
            || { tail -40 "$OP_DIR/build.log"; die "编译失败（完整日志 build.log）"; }
        ok "编译完成 -> build/${OP_NAME}"
        ;;
    run|test)
        load_cann   # 运行时同样需要 LD_LIBRARY_PATH（libmmpa.so 等）
        [ -x "$BUILD_DIR/$OP_NAME" ] || { "$0" build >/dev/null || exit 1; }
        info "运行"
        if [ "$ACTION" = "test" ]; then
            "$BUILD_DIR/$OP_NAME"; rc=$?
            [ $rc -eq 0 ] && ok "精度校验通过" || die "精度校验失败（退出码 $rc）"
        else
            "$BUILD_DIR/$OP_NAME"
        fi
        ;;
    *)
        die "未知动作: $ACTION（build | run | test | clean）"
        ;;
esac

#!/usr/bin/env bash
# ============================================================================
# 批量跑所有算子的精度用例
#
#   bash tests/run_all.sh [cpu|sim|npu] [all|<op>]
#
# 逐个执行 ops/<op>/run.sh test <mode>，最后汇总。任一个失败则退出码非 0，
# 便于本地或 CI 直接当门禁使用。
# ============================================================================
set -uo pipefail

MODE="${1:-cpu}"
TARGET="${2:-all}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPS_DIR="$REPO_ROOT/ops"

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    G='\033[1;32m'; Y='\033[1;33m'; R='\033[1;31m'; C='\033[1;36m'; N='\033[0m'
else
    G=''; Y=''; R=''; C=''; N=''
fi

[[ -d "$OPS_DIR" ]] || { printf "${R}没有 ops 目录${N}\n"; exit 1; }

if [[ "$TARGET" == "all" ]]; then
    mapfile -t OPS < <(find "$OPS_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
else
    OPS=("$TARGET")
fi

[[ ${#OPS[@]} -gt 0 ]] || { printf "${Y}ops/ 下没有算子，先跑: bash scripts/new_op.sh <名字>${N}\n"; exit 0; }

printf "${C}== 批量测试：%d 个算子，模式=%s ==${N}\n" "${#OPS[@]}" "$MODE"

declare -a PASSED=() FAILED=()
START=$(date +%s)

for op in "${OPS[@]}"; do
    printf "\n${C}---------------- %s ----------------${N}\n" "$op"
    if bash "$OPS_DIR/$op/run.sh" test "$MODE"; then
        PASSED+=("$op")
    else
        FAILED+=("$op")
    fi
done

ELAPSED=$(( $(date +%s) - START ))

printf "\n${C}== 汇总（%ds）==${N}\n" "$ELAPSED"
[[ ${#PASSED[@]} -gt 0 ]] && printf "  ${G}PASS${N}  %s\n" "${PASSED[*]}"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    printf "  ${R}FAIL${N}  %s\n" "${FAILED[*]}"
    exit 1
fi
printf "  ${G}全部通过${N}\n"

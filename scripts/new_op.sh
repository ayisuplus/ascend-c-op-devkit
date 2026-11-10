#!/usr/bin/env bash
# ============================================================================
# 从 templates/op 生成一个新算子工程（生成即可编译、可跑通精度校验）
#
#   bash scripts/new_op.sh silu
#   -> ops/silu/{silu.cpp, CMakeLists.txt, run.sh, README.md}
#
# 命名约束：小写字母开头，仅含 [a-z0-9_]，且不能与已有算子重名。
# ============================================================================
set -euo pipefail

NAME="${1:-}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMPLATE_DIR="$REPO_ROOT/templates/op"

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    G='\033[1;32m'; R='\033[1;31m'; N='\033[0m'
else
    G=''; R=''; N=''
fi
die() { printf "${R}%s${N}\n" "$*" >&2; exit 1; }

[[ -n "$NAME" ]] || die "用法: bash scripts/new_op.sh <算子名>   例: bash scripts/new_op.sh silu"
[[ "$NAME" =~ ^[a-z][a-z0-9_]*$ ]] || die "算子名需以小写字母开头，且只含 [a-z0-9_]，当前: $NAME"
[[ -d "$TEMPLATE_DIR" ]] || die "模板目录缺失: $TEMPLATE_DIR"

DEST="$REPO_ROOT/ops/$NAME"
[[ -e "$DEST" ]] && die "已存在: $DEST（换一个名字，或先删除旧目录）"

mkdir -p "$REPO_ROOT/ops" "$DEST"
cp -r "$TEMPLATE_DIR/." "$DEST/"
mv "$DEST/__OP_NAME__.cpp" "$DEST/$NAME.cpp"

# 模板里的 __OP_NAME__ 全部替换为真实算子名
grep -rl '__OP_NAME__' "$DEST" 2>/dev/null | while read -r f; do
    sed -i "s/__OP_NAME__/$NAME/g" "$f"
done

chmod +x "$DEST/run.sh"

printf "${G}已创建 ops/%s${N}\n" "$NAME"
echo "  下一步： bash ops/$NAME/run.sh test cpu"
echo "  然后编辑 ops/$NAME/$NAME.cpp 换成你自己的算子实现"

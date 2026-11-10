<#
.SYNOPSIS
    昇腾算子模拟环境 · Windows 侧统一入口。

.DESCRIPTION
    把命令转发进 WSL2 (AscendCANN) 里执行，Linux 侧逻辑放在 env/ 与 ops/ 下的 shell 脚本中。
    本脚本负责：解析 env/cann.env、确保 WSL 发行版存在、透传命令、统一 UTF-8 输出。

.EXAMPLE
    .\scripts\ascend.ps1 bootstrap        # 首次：导入 WSL 发行版 + 装系统依赖 + 装 CANN
    .\scripts\ascend.ps1 doctor           # 环境自检
    .\scripts\ascend.ps1 new rms_norm     # 从模板创建新算子 ops/rms_norm
    .\scripts\ascend.ps1 build add        # 编译 ops/add（默认 cpu 模式）
    .\scripts\ascend.ps1 run add -Mode sim
    .\scripts\ascend.ps1 test             # 跑全部算子精度用例
    .\scripts\ascend.ps1 shell            # 进入 WSL 交互 shell
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)][string]$Command = 'help',
    [Parameter(Position = 1)][string]$Op,
    [Parameter(Position = 2)][string]$Extra,
    [ValidateSet('cpu', 'sim', 'npu')][string]$Mode = 'cpu',
    [switch]$Rebuild
)

$ErrorActionPreference = 'Stop'
# WSL 输出是 UTF-8，Windows 控制台默认 GBK，不设置会乱码
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:WSL_UTF8 = '1'

$RepoRoot = Split-Path -Parent $PSScriptRoot

function Read-Conf {
    $path = Join-Path $RepoRoot 'env\cann.env'
    if (-not (Test-Path $path)) { throw "找不到配置文件: $path" }
    $cfg = @{}
    foreach ($line in Get-Content $path -Encoding UTF8) {
        $t = $line.Trim()
        if ($t -eq '' -or $t.StartsWith('#')) { continue }
        $i = $t.IndexOf('=')
        if ($i -lt 1) { continue }
        $cfg[$t.Substring(0, $i).Trim()] = $t.Substring($i + 1).Trim()
    }
    return $cfg
}

$Conf = Read-Conf
$Distro = $Conf['WSL_DISTRO']
$Link = $Conf['WORKSPACE_LINK']

function Test-DistroPresent {
    $names = (& wsl.exe -l -q) 2>$null
    if ($LASTEXITCODE -ne 0) { return $false }
    return ($names | ForEach-Object { $_.Trim().TrimEnd("`0") }) -contains $Distro
}

function Get-RootfsPath {
    Join-Path $RepoRoot '.cache\wsl\ubuntu-jammy-wsl-amd64.rootfs.tar.gz'
}

function Ensure-Distro {
    if (Test-DistroPresent) { return }
    Write-Host "[setup] WSL 发行版 $Distro 不存在，开始导入" -ForegroundColor Cyan

    $tarball = Get-RootfsPath
    if (-not (Test-Path $tarball)) {
        New-Item -ItemType Directory -Force -Path (Split-Path $tarball) | Out-Null
        $url = $Conf['WSL_ROOTFS_URL']
        Write-Host "[setup] 下载 Ubuntu 22.04 rootfs: $url" -ForegroundColor Cyan
        $sw = [Diagnostics.Stopwatch]::StartNew()
        $c = New-Object System.Net.WebClient
        $c.Headers.Add('User-Agent', 'curl/8')
        $c.DownloadFile($url, $tarball)
        $sw.Stop()
        Write-Host ("[setup] 下载完成 {0:N1} MB / {1:N0}s" -f ((Get-Item $tarball).Length / 1MB), $sw.Elapsed.TotalSeconds)
    }

    $importDir = $Conf['WSL_IMPORT_DIR']
    New-Item -ItemType Directory -Force -Path $importDir | Out-Null
    & wsl.exe --import $Distro $importDir $tarball --version 2
    if ($LASTEXITCODE -ne 0) { throw "wsl --import 失败（退出码 $LASTEXITCODE）" }
    Write-Host "[setup] 已导入 $Distro -> $importDir" -ForegroundColor Green
}

function Invoke-Wsl([string]$Script, [switch]$Interactive) {
    $args = @('-d', $Distro, '-u', 'root')
    if (-not $Interactive) { $args += @('--exec', 'bash', '-lc', $Script) }
    else { $args += @('--cd', '/') }
    if ($Interactive) {
        & wsl.exe @args
    } else {
        & wsl.exe @args
    }
    return $LASTEXITCODE
}

function Invoke-InWorkspace([string]$Script) {
    # 每个命令都重新 source CANN 环境，避免依赖交互 shell 的状态；
    # /work 是工作区的 ASCII 软链（中文路径对部分工具链不友好），缺失时自动补建。
    $ws = $Conf['WORKSPACE_WSL']
    $full = "set -o pipefail; [ -e $Link ] || ln -sfn '$ws' $Link; . /etc/profile.d/ascend-cann.sh 2>/dev/null || true; cd $Link; $Script"
    $code = Invoke-Wsl $full
    if ($code -ne 0) { Write-Host "[exit $code]" -ForegroundColor Red }
    return $code
}

switch ($Command.ToLower()) {
    'bootstrap' {
        Ensure-Distro
        Write-Host "[bootstrap] 在 $Distro 内安装依赖与 CANN" -ForegroundColor Cyan
        $ws = $Conf['WORKSPACE_WSL']
        $code = Invoke-Wsl "ln -sfn '$ws' $Link; bash $Link/env/wsl/bootstrap.sh"
        if ($code -eq 0) { Write-Host "[bootstrap] 完成，可运行: scripts\ascend.ps1 doctor" -ForegroundColor Green }
        exit $code
    }
    'doctor' {
        Ensure-Distro
        $ws = $Conf['WORKSPACE_WSL']
        Invoke-Wsl "ln -sfn '$ws' $Link; bash $Link/env/check_env.sh"
        exit $LASTEXITCODE
    }
    'new' {
        if (-not $Op) { throw "用法: scripts\ascend.ps1 new <算子名>" }
        Ensure-Distro
        $code = Invoke-InWorkspace "bash scripts/new_op.sh $Op"
        exit $code
    }
    { $_ -in 'build', 'run' } {
        if (-not $Op) { throw "用法: scripts\ascend.ps1 $Command <算子名> [-Mode cpu|sim|npu]" }
        Ensure-Distro
        $extra = if ($Rebuild) { ' --clean' } else { '' }
        $code = Invoke-InWorkspace "bash ops/$Op/run.sh $Command $Mode$extra"
        exit $code
    }
    'test' {
        Ensure-Distro
        $target = if ($Op) { $Op } else { 'all' }
        $code = Invoke-InWorkspace "bash tests/run_all.sh $Mode $target"
        exit $code
    }
    'shell' {
        Ensure-Distro
        Write-Host "进入 WSL $Distro（/work 指向工作区），exit 退出" -ForegroundColor Cyan
        & wsl.exe -d $Distro -u root --cd "/work"
        exit $LASTEXITCODE
    }
    'help' { Get-Help $PSCommandPath -Detailed; exit 0 }
    default {
        Write-Host "未知命令: $Command" -ForegroundColor Red
        Write-Host "可用命令: bootstrap | doctor | new | build | run | test | shell" -ForegroundColor Yellow
        exit 1
    }
}

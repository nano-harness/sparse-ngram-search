---
name: sngs-helper
description: |
  sngs（sparse-ngram-search）是一款面向工作空间的高速全文搜索工具，帮助用户或 agent 快速在代码仓库、项目目录等工作空间内搜索任意文本内容。
  当需要在工作空间中搜索代码、文件内容或文本模式时触发；也适用于询问如何安装、初始化、使用或管理 sngs 索引的场景。
---

# sngs-helper

## 核心概念

`sngs` 是 **sparse-ngram-search** 项目的命令行工具，基于稀疏 N-gram 算法为源码目录构建全文搜索索引。它通过倒排索引对字节序列进行快速检索，支持字面量和正则两种查询模式，无需在每次搜索时读取全部文件。

## Agent 使用策略

当需要在代码库中查找符号、字符串、配置项、错误信息或正则模式时，优先使用 `sngs`：

1. 如果仓库尚未建立索引，使用 `sngs search --auto-init <pattern>` 完成首次搜索。
2. 只需要文件清单时用 `sngs search -l <pattern>`（最轻量）；需要命中统计时用 `-c`。
3. 如果需要机器解析，使用 `sngs search --json <pattern>`；退出码遵循 rg 约定（0 有匹配 / 1 无匹配 / 2 出错），可直接用于脚本分支。
4. 如果需要像 grep/rg 一样阅读上下文，使用 `sngs search --grep -C 2 <pattern>`。
5. 如果搜索不到，再退回到 `rg` 做兜底扫描。
6. 修改工作区后，在 git 仓库中 `sngs` 会通过 git-aware overlay 感知变更；大仓库下搜索会复用 dirty-scan 缓存（自适应 TTL，最长 30 秒），刚刚发生的改动可能暂时搜不到——需要立即看到最新改动时，先运行 `sngs sync`（始终全量刷新）；也可用 `SNGS_SYNC_TTL_SECS` 调整该窗口（0 = 禁用缓存）。

推荐快捷流程：

```bash
sngs doctor
sngs search --auto-init --json "ConnectionPool"
sngs search --grep -C 2 --glob "src/*" "TODO"
sngs stats
```

## 下载与安装

### 源码编译（推荐）

需要 Rust 1.75+：

```bash
git clone https://github.com/nano-harness/sparse-ngram-search.git
cd sparse-ngram-search
cargo build -p sngs --release
mkdir -p ~/.local/bin
install -m 755 ./target/release/sngs ~/.local/bin/sngs
```

请确保 `~/.local/bin` 已加入 `PATH`。

## CLI 用法

```
sngs <子命令> [选项]
```

### `init` — 构建索引

为目录（默认为当前目录）构建稀疏 N-gram 搜索索引。

```bash
sngs init                       # 索引当前目录
sngs init --path /path/to/repo  # 索引指定目录
sngs init --no-ignore           # 非 Git 目录：包含被忽略的文件
```

Git 仓库中 `--no-ignore` 目前不会生效，并会输出警告。

### `search` — 搜索

在已索引的目录中执行字面量或正则查询。默认按字面量处理，加 `-e` 按正则处理。

```bash
# 字面量搜索
sngs search "fn parse_header"

# 正则搜索
sngs search -e "impl\s+Display"

# 仅输出文件名（不显示偏移量）
sngs search -l "TODO"

# 每个文件的匹配次数
sngs search -c "unsafe"

# 限制结果数量
sngs search -n 20 "unsafe"

# 搜索指定目录
sngs search --path /path/to/repo "ConnectionPool"

# 首次搜索自动构建索引
sngs search --auto-init "ConnectionPool"

# JSON 输出，适合 agent / 编辑器
sngs search --json "ConnectionPool"

# 类 grep 输出和路径过滤（--glob 包含、--exclude 排除、--type 按扩展名）
sngs search --grep --glob "src/*" --exclude "*/tests/*" --type rs "ConnectionPool"
```

默认输出格式：

```
src/parser.rs:3 match(es)
src/proto/header.rs:1 match(es)
[sngs] 2 result(s) in 0.004s
```

**退出码**（与 ripgrep 一致，适合脚本判断）：`0` = 有匹配，`1` = 无匹配，`2` = 出错。

**全局选项**：

- `--timeout <SECS>`：超时秒数（0 = 不限），超大仓库的兜底保护。
- `--index-dir <DIR>`：把索引放到仓库外的目录（只读仓库或共享索引场景）。

### `sync` — 更新索引

文件变更后更新索引。Git 仓库使用增量 overlay 更新；非 Git 目录目前会执行全量重建。

```bash
sngs sync                       # 同步当前目录的索引
sngs sync --path /path/to/repo
```

### `status` — 查看索引状态

显示索引元数据，包括文件数、N-gram 数、最后更新时间等。

```bash
sngs status
sngs status --path /path/to/repo
```

示例输出：

```
Index directory : "/my/repo/.sngi-store"
Version         : 1
Files           : 312
N-gram tokens   : 48291
Last updated    : 183 days after 2024-01-01 (unix epoch 1719878400)
lookup.bin      : 378 KiB
postings.bin    : 512 KiB
```

### `rebuild` — 强制完全重建

忽略现有索引，对目录执行完整的重新索引。

```bash
sngs rebuild
sngs rebuild --path /path/to/repo --no-ignore
```

### 诊断、预热与统计

```bash
sngs doctor              # 检查安装、索引位置和健康状态
sngs warmup              # 预热索引，适合批量搜索前运行
sngs explain-index       # 解释 .sngi-store 布局和当前状态
sngs stats               # 本地使用统计；设置 SNGS_NO_STATS=1 可关闭
sngs completion zsh      # 生成 shell 补全脚本
```

## 工作流程示例

典型使用场景：在大型代码仓库中快速定位某个函数或模式。

```bash
# 1. 进入项目目录并构建索引（首次使用也可以直接 --auto-init）
cd /path/to/my-project
sngs search --auto-init "ConnectionPool"

# 2. 搜索字面量，输出 JSON 给 agent 解析
sngs search --json "ConnectionPool"

# 3. 搜索正则（查找所有 async fn 定义）
sngs search --grep -C 2 -e "async fn \w+"

# 4. 修改代码后增量同步索引
sngs sync

# 5. 查看索引状态
sngs status
```

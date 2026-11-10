# sparse-ngram-search

基于**稀疏 N-gram** 算法的源码全文搜索索引。对文件中的字节序列构建紧凑倒排索引，支持字面量和正则两种查询，无需逐一读取所有文件即可在毫秒级返回结果。

[English](README.md)

典型使用场景：

- 在大型仓库中快速跳转到符号、字符串或配置项；
- 为 AI agent 提供快速、可 JSON 解析的代码搜索能力；
- 对同一工作区反复执行字面量或正则搜索，避免每次全量扫描。

---

## 工作原理

### 稀疏 N-gram 算法

传统 N-gram 索引将文本切分为所有可能的定长 token，空间和时间开销较大。本项目采用**权重引导边界**方案：

1. 对每个相邻字节对 `(a, b)` 依据其在典型源码中出现的稀有程度赋予一个**权重**（存储在预构建的频率表中）。
2. 当候选 token 两端的字节对权重都**严格大于**所有内部字节对权重时，在该位置划定**边界**。
3. 由此产生的 token 倾向于在不常见的字符转换处开始和结束，保持 token 简短且有区分度。

提供两种提取模式：

| 模式 | 函数 | 用途 |
|---|---|---|
| 全量 N-gram | `build_all` | 构建索引（提取文件中每个 token） |
| 覆盖集 | `build_covering` | 生成查询（覆盖 pattern 的最小 token 集合） |

### 索引结构

默认情况下，CLI 索引存储在源码目录下的 `.sngi-store/` 子目录中：

```
.sngi-store/
  state.json
  segments/
    segment_0/
      lookup.bin    # 有序数组，存储 (ngram_hash → postings_offset)
      postings.bin  # 每个 ngram 对应的文件 ID 列表，路径以长度前缀编码
      meta.json     # 版本、文件数、ngram 数、基准 commit、时间戳
```

两个二进制文件均通过 `memmap2` 内存映射，启动开销为 O(1)，查找使用二分搜索。

### 查询流程

```
pattern（模式）
  │
  ├─ 字面量  →  提取覆盖 N-gram  →  交集 postings  →  memmem 精确验证
  │
  └─ 正则    →  分解为 QueryAtom  →  交集 postings  →  正则引擎精确验证
               （AND / OR / Any 字面量）
```

postings 交集充当**预过滤器**：只有包含所有必要 N-gram 的候选文件才会进入开销较大的验证步骤，假阴性率接近零，且比逐文件 grep 产生更少的误报。

### Git 感知索引

`sparse-ngram-git` 为 git 仓库提供双层架构：

```
GitAwareIndex
  ├─ base_index   （磁盘索引，锚定到某个具体 commit）
  └─ overlay      （内存增量，记录上次 sync 后的新增/修改/删除）
```

调用 `sync()` 时，只有自 `base_commit` 以来发生变化的文件才会被重新索引到 overlay 中。搜索会合并两层的结果，overlay 会屏蔽基础索引中已删除或已被替换的条目。

### 搜索时效性与 dirty-scan 缓存

感知未提交的工作区变更需要扫描整个工作目录，在超大仓库上耗时可达数秒。为了保证搜索速度，扫描结果会缓存到 `.sngi-store/dirty_scan.json`，在新鲜期内直接复用：

- TTL 是自适应的：`min(上次扫描耗时 × 10, 30 秒)`。扫描开销小的仓库实际上始终保持新鲜；大仓库则在一个有限窗口内跳过重扫。
- 在缓存扫描**之后**才变脏的文件，可能要等窗口过期后才能被搜到；而已在缓存列表中的文件每次都会被重新读取和校验。
- `sngs sync` 始终执行全量扫描（`GitAwareIndex::sync()` 亦然）。
- 可通过 `SNGS_SYNC_TTL_SECS` 固定 TTL（设为 0 则完全禁用缓存）。

---

## 项目结构

```
sparse-ngram-search/
├── crates/
│   ├── core/       # sparse-ngram-core — 算法、索引与查询引擎
│   ├── git/        # sparse-ngram-git  — Git 感知索引（GitAwareIndex）
│   ├── cli/        # sngs              — 命令行工具
│   ├── freq-gen/   # freq-gen          — 频率表生成工具
│   └── bench/      # sparse-ngram-bench — 性能基准测试套件
├── Cargo.toml      # workspace 配置
└── README.md
```

---

## 安装

### 源码编译（推荐）

需要 **Rust 1.75+**（edition 2021）。

```bash
git clone https://github.com/nano-harness/sparse-ngram-search.git
cd sparse-ngram-search
cargo build -p sngs --release

# 可选：安装到当前用户目录
mkdir -p ~/.local/bin
install -m 755 ./target/release/sngs ~/.local/bin/sngs
```

请确保 `~/.local/bin` 已加入 `PATH`。

---

## Skill 安装

`sngs-helper` Skill 位于本地 [`sngs-helper/SKILL.md`](sngs-helper/SKILL.md)。请根据 AI 助手文档将其包含到工作区或作为指令集加载。

---

## CLI 使用

```
sngs <子命令> [选项]
```

### `init` — 构建索引

```bash
sngs init                    # 索引当前目录
sngs init --path /my/repo    # 索引指定目录
sngs init --no-ignore        # 非 Git 目录：包含被忽略的文件
```

Git 仓库的索引遵循 Git 对已跟踪和未跟踪文件的管理方式；`--no-ignore` 目前不会生效，并会输出警告。

### 外部索引与只读工作区

默认情况下，`init` 会在工作区中创建 `.sngi-store/`，并为 Git 仓库将该目录添加到
`.gitignore`。首次搜索 Git 仓库时，也会自动初始化缺失的索引。

如果不希望在工作区内生成索引文件，请选择一个位于**工作区之外**的目录：

```bash
sngs --index-dir "$HOME/.cache/sngs" search --path /my/repo "ConnectionPool"
# 或为所有命令配置相同的存储位置：
export SNGS_INDEX_DIR="$HOME/.cache/sngs"
sngs init --path /my/repo
```

外部索引根据工作区路径的哈希值，为每个工作区使用独立的子目录。使用外部存储时，
初始化（包括自动初始化）不会创建或修改工作区中的 `.gitignore`，因此支持只读工作区。
索引更新和工作区变更扫描缓存也使用同一外部位置。后续命令应继续使用该位置；
`--index-dir` 的优先级高于 `SNGS_INDEX_DIR`。

### `search` — 搜索

```bash
# 字面量搜索
sngs search "fn parse_header"

# 正则搜索
sngs search -e "impl\s+Display"

# 仅显示文件名（不输出偏移量）
sngs search -l "TODO"

# 限制结果数量
sngs search -n 20 "unsafe"

# 搜索指定目录
sngs search --path /my/repo "ConnectionPool"

# 首次搜索时自动构建索引
sngs search --auto-init "ConnectionPool"

# 面向 agent / 编辑器的 JSON 输出
sngs search --json "ConnectionPool"

# 类 grep 输出和路径过滤
sngs search --grep --glob "src/*" --type rs "ConnectionPool"
```

默认输出格式：

```
src/parser.rs:3 match(es)
src/proto/header.rs:1 match(es)
[sngs] 2 result(s) in 0.004s
```

### `sync` — 更新索引

```bash
sngs sync                    # 文件变更后更新索引
sngs sync --path /my/repo
```

Git 仓库使用增量 overlay 更新；非 Git 目录目前会执行全量重建。

### `status` — 查看索引元数据

```bash
sngs status
```

输出示例：

```
Index directory : "/my/repo/.sngi-store"
Version         : 1
Files           : 312
N-gram tokens   : 48291
Last updated    : 93 days after 2024-01-01 (unix epoch 1719878400)
lookup.bin      : 378 KiB
postings.bin    : 512 KiB
```

### `rebuild` — 强制完全重建

```bash
sngs rebuild
sngs rebuild --path /my/repo --no-ignore
```

### 诊断与 agent 友好命令

```bash
sngs doctor              # 检查安装与索引健康状态
sngs warmup              # 打开索引以预热 OS page cache
sngs explain-index       # 解释 .sngi-store 布局和当前元数据
sngs stats               # 本地、可关闭的使用统计
sngs completion zsh      # 生成 shell completion
```

---

## 库接口

### core crate

```rust
use sparse_ngram_core::{IndexBuilder, Index, QueryEngine};

// ── 构建索引 ───────────────────────────────────────────────────────────────
let mut builder = IndexBuilder::with_default_weights();
builder.add_file(Path::new("src/main.rs"), b"fn main() { println!(\"hello\"); }")?;
let index = builder.build(Path::new("target/demo-index"))?;

// ── 打开索引（重启后）─────────────────────────────────────────────────────
let index = Index::open(Path::new("target/demo-index"))?;

// ── 搜索 ───────────────────────────────────────────────────────────────────
let engine = QueryEngine::new(&index);

// 字面量搜索
let results = engine.search_literal("println", |_id, path| std::fs::read(path).ok())?;

// 正则搜索
let results = engine.search(r"fn\s+\w+\(", |_id, path| std::fs::read(path).ok())?;

for r in &results {
    println!("{}: {:?} offsets", r.path.display(), r.match_offsets);
}
```

### git-aware crate

```rust
use sparse_ngram_git::GitAwareIndex;

let mut gi = GitAwareIndex::open_or_build(Path::new("/my/repo"))?;

// 将 overlay 同步到当前工作区变更
let stats = gi.sync()?;
println!("added={} modified={} deleted={}", stats.added, stats.modified, stats.deleted);

// 在基础索引 + overlay 上搜索
let results = gi.search("ConnectionPool")?;
```

---

## 性能基准测试

独立的性能测试二进制程序，跨五个语料库场景衡量索引时间、搜索时间和索引大小。

```bash
cargo run -p sparse-ngram-bench --release
```

测试程序会生成合成的 Rust 风格源文件，对每个场景执行若干次迭代，并将 Markdown 格式的结果表格输出到 stdout。进度信息输出到 stderr，便于观察长时间运行的场景。

**测试场景**

| 场景                 |   文件数 | 单文件大小 |
|----------------------|--------:|-----------:|
| 100 files × 1 KiB   |     100 |     1 KiB  |
| 100 files × 4 KiB   |     100 |     4 KiB  |
| 1000 files × 4 KiB  |   1 000 |     4 KiB  |
| 1000 files × 16 KiB |   1 000 |    16 KiB  |
| 5000 files × 4 KiB  |   5 000 |     4 KiB  |

**示例输出**（现代笔记本电脑 release 构建）：

```
## 1. Indexing Performance

| Scenario                  |      Files |  Corpus Size | Avg Index Time |   Index Size | N-gram Count |  Size Ratio |
| ------------------------- | ---------- | ------------ | -------------- | ------------ | ------------ | ----------- |
| 100 files × 1 KiB         |        100 |     117.6 KB |       3.45 ms  |     45.2 KiB |         3821 |       38.4% |
| 100 files × 4 KiB         |        100 |     470.4 KB |      12.10 ms  |    142.7 KiB |        10294 |       30.3% |
| ...                       |        ... |          ... |            ... |          ... |          ... |         ... |

## 2. Search Performance

| Scenario                  | Query              | Kind    |  Avg Search Time | Result Files |
| ------------------------- | ------------------ | ------- | ----------------: | -----------: |
| 100 files × 1 KiB         | fn func_           | literal |          0.3 µs  |          100 |
| 100 files × 1 KiB         | fn \w+_\d+         | regex   |          1.2 µs  |          100 |
| ...                       | ...                | ...     |              ... |          ... |
```

> **提示：** 请务必使用 `--release` 模式运行——debug 构建由于 n-gram 提取循环未优化，速度会慢 10–20 倍。

---

## 测试

```bash
# 全量测试（单元测试 + 集成测试）
cargo test --workspace

# 仅 core 库
cargo test -p sparse-ngram-core

# 仅 git 感知索引
cargo test -p sparse-ngram-git

# CLI 端到端测试
cargo test -p sngs
```

测试覆盖范围：

- 字面量 / 正则搜索命中与未命中
- 无假阴性保证
- 多文件索引一致性
- 重建幂等性
- Unicode 与 emoji 内容
- `GitAwareIndex` 新增 / 修改 / 删除同步
- 所有 CLI 子命令（`init`、`search`、`sync`、`status`、`rebuild`）

---

## 许可证

MIT

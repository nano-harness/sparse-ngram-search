# Contributing

Thank you for contributing to sparse-ngram-search.

## Development setup

Install Rust 1.75 or newer, then build and test the workspace:

```bash
cargo build --workspace
cargo test --workspace
```

Before submitting a pull request, run:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

## Changes

- Keep pull requests focused and explain user-visible behavior changes.
- Add or update tests for bug fixes and new functionality.
- Update both `README.md` and `README.zh-CN.md` when usage or behavior changes.
- Do not commit `.sngi-store` directories, local indexes, credentials, or build artifacts.

Report bugs and feature requests through GitHub Issues. For security reports, see `SECURITY.md`.

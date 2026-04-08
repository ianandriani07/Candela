# simple_tensor

A Rust-first tensor runtime that emphasizes **memory-layout-aware execution**, **lazy computation graphs**, and **practical performance paths on CPU** via Intel MKL/VML.

This repository is an experimental systems project: it is not a full deep learning framework yet, but it already contains a non-trivial execution model with node fusion, reusable buffers, explicit caching controls, and low-level stride/slice semantics.

---

## Project Overview

`simple_tensor` provides a tensor abstraction with two distinct modes:

- **`Tensor<T>`**: concrete data (allocated buffer + layout metadata).
- **`TensorPromise<T>` / `CachedTensorPromise<T>`**: deferred computation nodes in a DAG that can be materialized on demand.

The current implementation focuses on `f64` CPU compute and uses Intel MKL/VML-backed kernels for core elementwise operations. The architecture is intentionally designed for future extension into broader BLAS coverage, better fusion, and potentially GPU backends.

## Why this project exists

Most educational tensor libraries stop at shape math or eager elementwise kernels. This codebase goes further by exploring real runtime concerns:

- avoiding unnecessary allocations,
- preserving and exploiting layout/stride information,
- doing topological materialization over a DAG,
- enabling selective cache persistence for repeated subgraphs,
- integrating vendor-tuned numeric kernels.

In short: this is a **learning-driven but performance-conscious tensor engine prototype**.

---

## Core Goals

1. Build a minimal but serious tensor runtime in Rust.
2. Support lazy graph composition with explicit materialization.
3. Encode shape/stride/offset semantics cleanly, including views/slices/transposes.
4. Reuse memory aggressively where safe.
5. Fuse compatible scalar operations to reduce graph and runtime overhead.
6. Keep the architecture open for future matmul/BLAS/GPU expansion.

---

## Key Features

- **Lazy DAG execution** with `TensorPromise` graph nodes.
- **Topological graph materialization** with reference counting for eager intermediate reclamation.
- **Optional persisted node caching** through `CachedTensorPromise::cache()`.
- **Operation fusion** for scalar chains (`+/-/*//`) and selected layout-transform combinations.
- **Layout-aware tensor model**:
  - shape / stride / adjusted stride / offset / logical length,
  - contiguous and non-contiguous iteration paths,
  - views, slicing, transpose, and transpose-by-axes.
- **CPU kernels for `f64`**:
  - scalar ops,
  - tensor-tensor elementwise ops via Intel VML (`vdAdd`, `vdSub`, `vdMul`, `vdDiv`).
- **Macro conveniences** (`arange!`, `srange!`, `zeros!`, `ones!`, `s!`).

---

## Architecture

### 1) Data and Layout Layer

- `Storage<T>` wraps `Arc<RwLock<Vec<T>>>` for shared backing buffers.
- `TensorData<T>` combines storage + `Layout` + `reusable` marker.
- `Layout` stores:
  - `shape`,
  - `stride`,
  - `adj_stride` (precomputed stepping deltas for iterators),
  - `offset`,
  - `len`.

This enables non-contiguous tensor representations without copying.

### 2) API Surface Layer

- `Tensor<T>` = concrete graph edge (material data).
- `TensorPromise<T>` = lazy node (`TensorGraphNode`) with op + inputs + inferred layout.
- `CachedTensorPromise<T>` = lazy node with internal `OnceCell` cache.

Operator overloading (`Add/Sub/Mul/Div`) is implemented across all combinations of `Tensor`, `TensorPromise`, and cached promises, allowing natural expression-building syntax.

### 3) Graph Runtime Layer

Nodes are represented as:

- `Edge` (material tensor),
- `Node` (deferred op),
- `Cache` (deferred op with persisted output).

Materialization performs:

1. DFS topological sort,
2. reference counting of node usages,
3. compute in dependency order,
4. eager release of intermediates when refcount reaches zero,
5. cache reuse/population for cached nodes.

### 4) Compute Layer

Compute dispatch goes through `cpu_compute(op, output_layout, inputs)` with type-specific implementations (`ComputeWrapperSpec`).

Current concrete backend: `f64`.

Implemented operation families:

- scalar op kernels,
- fused scalar chains,
- layout-only transforms (`view`, `slice`, `transpose`, `transpose_axes`, `as_contiguous`),
- elementwise tensor-tensor ops via MKL VML,
- no-op passthrough.

`Matmul` layout inference exists, and partial matmul kernel scaffolding is present, but full matmul execution is not complete.

---

## Repository Structure

```text
.
├── Cargo.toml
├── LICENSE
├── project_decision.txt
├── src
│   ├── main.rs
│   └── tensor
│       ├── mod.rs
│       ├── tensor.rs             # Tensor wrapper over graph edge
│       ├── promise.rs            # Lazy promise types + materialization
│       ├── graph.rs              # DAG nodes, topological sort, execution
│       ├── storage.rs            # Buffer ownership, TensorData, reuse markers
│       ├── mem_formats
│       │   ├── layout.rs         # shape/stride/offset metadata + transforms
│       │   └── slice.rs          # slice range parsing + slice layout derivation
│       ├── iter.rs               # contiguous/non-contiguous/chunked iterators
│       ├── ops
│       │   ├── impl_op.rs        # API ops + operator overloading
│       │   ├── impl_layout.rs    # output layout inference + validation
│       │   ├── impl_compute_op.rs# kernel dispatch + MKL/VML integration
│       │   ├── reusable.rs       # allocation reuse strategy
│       │   └── fusion.rs         # scalar operation fusion
│       ├── convenience.rs        # macros (arange/zeros/ones/s)
│       ├── traits.rs
│       ├── errors.rs
│       ├── internals.rs
│       ├── impl_generics.rs      # Display impl macro
│       └── mkl_extension.rs      # low-level FFI declarations
```

---

## Technologies Used

- **Language**: Rust (Edition 2024)
- **Numerics / BLAS ecosystem**:
  - `intel-mkl-src`
  - `intel-mkl-sys`
  - `cblas`
  - `cblas-sys`
  - `lapacke`
- **Synchronization primitives**: `parking_lot`

---

## Build Instructions

> Prerequisite: a Rust toolchain (`cargo`, `rustc`).

```bash
cargo build
```

For debug checks controlled by feature flags:

```bash
cargo build --features debug_only_check
```

---

## Run Instructions

The current `main.rs` demonstrates chained lazy scalar ops over an `arange!` tensor and materialization:

```bash
cargo run
```

---

## Example Usage

```rust
use simple_tensor::tensor::{arange, Dimension};

let t = arange![12];
let mut p = t.as_promise();

for i in 0..20 {
    p = p + i as f64;   // lazy graph growth, potentially fuseable
}

let out = (p * 2.0).materialize();
println!("{}", out);
```

Other supported patterns include:

- `view(...)` for shape reinterpretation (contiguous-only check in debug mode),
- `slice(s![...])`,
- `transpose()` / `transpose_axes(...)`,
- tensor-tensor elementwise arithmetic,
- `.cache()` for explicit reuse across materializations.

---

## Performance-Oriented Design Decisions

### Lazy graph + explicit materialization
Computation is deferred until `.materialize()`, allowing fusion and global scheduling opportunities.

### Topological execution with refcount-driven reclamation
Intermediate tensors are not kept longer than needed; ownership and reference counting drive eager discard.

### Reuse-aware buffer strategy
`TensorData` can be marked reusable, and execution attempts to strip/reuse buffers (`Arc::try_unwrap`) before allocating new contiguous storage.

### Layout-preserving transforms
View/slice/transpose operations are layout rewrites whenever possible, reducing copies and keeping memory traffic lower.

### Specialized contiguous vs non-contiguous iterator paths
The engine branches between fast contiguous iteration and stride-aware traversal for general layouts.

### Scalar fusion
Compatible scalar chains are collapsed (e.g., repeated adds/muls), reducing node count and compute passes.

---

## Implementation Details Worth Highlighting

- `adj_stride` precomputation is used to implement efficient multidimensional stepping without recomputing index math each step.
- `CachedTensorPromise` uses `OnceCell` to persist computed outputs and skip recomputation for stable subgraphs.
- Operation ergonomics follow a deliberate policy:
  - inline operator misuse panics for clearly-invalid programmer errors,
  - method-style operations are designed to return `Result` where appropriate.
- `project_decision.txt` documents design intent around panic-vs-result behavior, caching philosophy, and current fusion limits.

---

## Limitations / Current Assumptions

- Compute backend specialization is currently centered on `f64`.
- Matmul execution is incomplete (layout path exists; full compute path is not finished).
- Fusion is intentionally simple and order-dependent; complex multi-input fusion cases are not broadly handled.
- There are no formal benchmark artifacts or automated tests in the current repository.
- Some modules include WIP/legacy code and compiler warnings; this is an actively evolving prototype.

---

## Suggested Future Improvements

1. Complete and validate BLAS-backed matmul kernels (including batched paths).
2. Expand compute specialization to additional numeric types.
3. Add correctness/property tests for layout transforms and graph execution.
4. Add reproducible benchmarks (micro + end-to-end) and profiling scripts.
5. Extend fusion beyond scalar chains to richer algebraic/graph patterns.
6. Improve diagnostics, logging controls, and panic/error consistency.
7. Evaluate lock-free or reduced-lock storage strategies for high-concurrency workloads.

---

## Contributing

Contributions are welcome, especially around:

- kernel implementation and optimization,
- graph/fusion improvements,
- matmul completion,
- test coverage and benchmarking harnesses,
- API hardening.

If you contribute, please prefer changes that preserve explicit layout semantics and avoid hidden allocations unless clearly justified.

---

## License

MIT License (see [`LICENSE`](./LICENSE)).

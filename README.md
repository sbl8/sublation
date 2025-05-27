# Sublation – Dialectical AI Engine (Experimental Project)

> **Compile‑time data‑flow, run‑time streaming.**  
> No global tensors – every compute block is a **Sublate** with local **dual byte‑buffers**.

[![Go Reference](https://pkg.go.dev/badge/github.com/sbl8/sublation.svg)](https://pkg.go.dev/github.com/sbl8/sublation)
[![Go Report Card](https://goreportcard.com/badge/github.com/sbl8/sublation)](https://goreportcard.com/report/github.com/sbl8/sublation)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Sublation is a minimalist, high-performance AI engine built in pure Go that reimagines neural computation through **dialectical sublates** – self-contained memory-compute units that replace traditional tensor architectures.

### Key Features

- **🚀 Zero-allocation kernels** – All memory pre-planned; no malloc/free during execution
- **⚡ SIMD-optimized operations** – Hand-tuned AVX2/NEON assembly for hot paths  
- **🔄 Dual-buffer architecture** – Cache-aligned buffers with lock-free updates
- **📊 Static compilation** – `.subs` specifications → optimized `.subl` binaries
- **🎯 Pure Go** – Single static binary, no external dependencies
- **🔗 Dataflow scheduling** – Fine-grained parallelism via streaming execution

## Quick Start

### Installation

```bash
git clone https://github.com/sbl8/sublation.git
cd sublation
./build.sh
```

### Basic Usage

```bash
# Compile a model specification
./bin/sublc -O examples/neural_network.subs model.subl

# Run inference
echo "1.0 0.5 0.75 1.0" | ./bin/sublrun model.subl

# Performance benchmarking
./bin/sublperf -test=all -size=1024
```

### Example Model (.subs)

```subs
# Simple neural network layer
node 0 0x00 0 0 0x01
payload 3f8000003f0000003f4000003f800000  # [1.0, 0.5, 0.75, 1.0]

# ReLU activation  
node 1 0x03 0 16 0x02

# Sigmoid output
node 2 0x04 16 32 0x04
```

## Architecture

Sublation implements a novel **sublate-centric** computation model:

- **Sublate** = cache-aligned struct with dual buffers (`PayloadPrev`, `PayloadProp`) + topology
- **Kernels** = pure functions operating in-place on sublate data
- **Runtime** = streaming scheduler with zero-allocation arena memory management

![Architecture Diagram](docs/diagrams/runtime_arch.png)

## Performance

Sublation achieves significant performance improvements through:

- **Memory Locality**: All data cache-aligned, minimal DRAM access
- **Zero Overhead**: No runtime allocations, interpretation, or dynamic dispatch
- **SIMD Utilization**: Vectorized operations with architecture-specific optimizations
- **Pipeline Parallelism**: Lock-free dataflow execution

Benchmark results show **3-22× speedup** over traditional tensor frameworks on inference workloads.

## Documentation

- **[Architecture Overview](docs/architecture_overview.md)** – High-level system design
- **[Sublate Units Whitepaper](docs/diagrams/sublate_units.md)** – Theoretical foundations
- **[API Documentation](https://pkg.go.dev/github.com/sbl8/sublation)** – Go package docs
- **[Build System](docs/build.md)** – Compilation pipeline details

## Project Structure

```bash
sublation/
├── cmd/                    # CLI tools
│   ├── sublc/             # Sublation compiler  
│   ├── sublrun/           # Runtime engine
│   └── sublperf/          # Performance benchmarks
├── core/                  # Low-level primitives
│   ├── sublate.go         # Core Sublate struct
│   ├── align.go           # Memory alignment helpers
│   └── layout.go          # Cache-optimized layouts  
├── kernels/               # SIMD-optimized operations
│   ├── ops.go             # Kernel catalog
│   ├── asm_amd64.s        # AVX2 assembly implementations
│   └── asm_fallback.go    # Pure Go fallbacks
├── runtime/               # Execution engine
│   ├── runtime.go         # Main runtime engine
│   └── arena.go           # Memory arena management
├── compiler/              # Model compilation
│   └── compiler.go        # .subs → .subl compiler
├── model/                 # Graph representation
│   └── graph.go           # Model graph structures
├── examples/              # Example models
└── docs/                  # Documentation
```

## Development

### Prerequisites

- Go 1.22.2 or later
- Linux/macOS (Windows support planned)

### Building from Source

```bash
# Clone the repository
git clone https://github.com/sbl8/sublation.git
cd sublation

# Build all components
./build.sh

# Run tests
go test ./... -v

# Run benchmarks
go test -bench=. ./kernels/
```

### Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. **Architecture**: Follow dual-buffer, zero-allocation principles
2. **Code Style**: Pure Go, <100 LOC functions, no reflection in hot paths
3. **Testing**: Include benchmarks for performance-critical code
4. **Documentation**: Godoc comments for all public APIs

## Roadmap

- [ ] **GPU Backend** – OpenCL/CUDA compute shaders
- [ ] **Quantization** – 8-bit integer kernel variants  
- [ ] **Dynamic Fusion** – Runtime kernel combining optimization
- [ ] **Auto-differentiation** – Training support via reverse-mode AD
- [ ] **Language Bindings** – Python/JavaScript FFI

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## Citation

If you use Sublation in your research, please cite:

```bibtex
@misc{sublation2025,
  title={Sublation: Dialectical AI through Compositional Memory-Compute Units},
  author={[Your Name]},
  year={2025},
  url={https://github.com/sbl8/sublation}
}
```

---

**Sublation** – Where dialectical philosophy meets systems engineering.

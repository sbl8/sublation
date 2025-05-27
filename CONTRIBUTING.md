# Contributing to Sublation

We welcome contributions to the Sublation dialectical AI engine! This document provides guidelines for contributing to the project.

## Code of Conduct

This project adheres to a code of professional conduct. Be respectful, constructive, and collaborative in all interactions.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a branch** for your feature/fix
4. **Make your changes** following our coding standards
5. **Test thoroughly** including benchmarks for performance code
6. **Submit a pull request** with a clear description

## Development Setup

### Prerequisites

- Go 1.22.2 or later
- Linux or macOS (Windows support planned)
- Git

### Local Development

```bash
# Clone your fork
git clone https://github.com/sbl8/sublation.git
cd sublation

# Build the project
./build.sh

# Run tests
go test ./... -v

# Run benchmarks
go test -bench=. ./kernels/

# Lint code
golangci-lint run
```

## Coding Standards

### Architecture Principles

1. **Pure Go Only**: External dependencies only if faster, stable, vendor-agnostic, <3k SLOC
2. **Zero Allocations**: No runtime allocations in kernels - pre-allocate everything
3. **Dual-Buffer Semantics**: Never mutate `PayloadPrev`; write to `PayloadProp`, then swap
4. **Cache Alignment**: All data structures aligned to cache boundaries for SIMD efficiency

### Code Style

1. **Elegance over Cleverness**: Remove duplication, keep functions <100 LOC
2. **Performance First**: Inner loops branch-free, pointer arithmetic OK (`unsafe` allowed)
3. **Observability**: Every kernel returns timing/µalloc stats in development builds
4. **Clear Naming**: Use `buf`, `stride`, `delta`, not `thesis`/`antithesis`

### What to Avoid

- Tensor APIs, reflection, JSON marshaling inside kernels
- Hidden global state or magic singletons
- Metaphorical variable names
- Runtime allocations in hot paths
- `interface{}` in performance-critical code

### Package Organization

- `core/`: Low-level primitives (`Sublate`, `Arena`, align helpers)
- `kernels/`: ONLY pure functions that accept `(prev, prop []float32, out []float32)` or `*Sublate`
- `runtime/`: Scheduler, worker-pool, lineage tracker, metrics
- `compiler/`: Spec→graph builder, buffer-lifetime planner, kernel fusion
- `cmd/`: CLI tools (`sublc`, `sublrun`, `sublperf`)

## Testing Requirements

### Unit Tests

- **Coverage**: Aim for >80% test coverage
- **Benchmarks**: Include benchmarks for all performance-critical functions
- **Race Detection**: All tests must pass with `-race` flag
- **Multiple Platforms**: Test on both Linux and macOS

### Benchmark Standards

```go
func BenchmarkMyKernel(b *testing.B) {
    data := make([]byte, 1024*4) // 1K float32s
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        myKernel(data)
    }
    b.ReportAllocs() // Must report 0 allocs for kernels
}
```

### Test Organization

- Place tests in same package as source, suffixed `_test.go`
- Use table-driven tests for multiple input scenarios
- Include both positive and negative test cases
- Test edge cases and error conditions

## Documentation Standards

### Go Documentation

- All public APIs must have godoc comments
- Package-level documentation explaining purpose and usage
- Examples for complex APIs
- Clear parameter and return value descriptions

### Comments

```go
// ExampleFunction demonstrates proper documentation.
// It takes a buffer and performs an in-place operation.
// The buffer must be aligned to 32-byte boundaries.
// Returns an error if the buffer is invalid.
func ExampleFunction(buf []byte) error {
    // Implementation details...
}
```

## Performance Guidelines

### SIMD Optimization

- Provide both assembly and pure Go implementations
- Use build tags to select appropriate version
- Include benchmarks comparing implementations
- Document performance characteristics

### Memory Management

- Pre-allocate all buffers at startup
- Use object pools (`sync.Pool`) for temporary allocations
- Align data structures to cache boundaries
- Minimize pointer chasing and indirection

### Profiling

- Include `pprof` labels for long-running kernels
- Provide benchmark comparisons for performance changes
- Test memory allocation patterns
- Validate cache performance with hardware counters

## Commit Guidelines

### Commit Messages

```go
type(scope): brief description

Longer explanation if needed, including:
- Why the change was made
- What alternatives were considered
- Any breaking changes

Fixes #123
```

Types: `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `style`, `ci`

### Pull Request Process

1. **One logical change per PR** (feature, fix, or performance patch)
2. **Include benchstat diff** if performance critical
3. **Update documentation** for API changes
4. **Add tests** for new functionality
5. **Run full test suite** before submitting

### Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests added/updated and passing
- [ ] Documentation updated
- [ ] Performance impact assessed
- [ ] No unnecessary dependencies added
- [ ] SIMD code includes fallbacks
- [ ] Memory allocations avoided in hot paths

## Areas for Contribution

### High Priority

- **New SIMD kernels**: `axpy`, `gemv`, quantization converters
- **Scheduler optimizations**: Work stealing, ready queue improvements
- **Memory layout**: Static buffer layout calculators
- **GPU backends**: OpenCL/CUDA compute shaders

### Medium Priority

- **Language bindings**: Python/JavaScript FFI
- **Quantization**: 8-bit integer kernel variants
- **Auto-differentiation**: Training support
- **Documentation**: More examples and tutorials

### Experimental

- **Dynamic fusion**: Runtime kernel combining
- **Distributed execution**: Multi-node coordination
- **Hardware acceleration**: FPGA/custom silicon support

## Getting Help

- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: Design questions and architectural decisions
- **Code Review**: Feedback on implementation approaches

## Release Process

1. Update version numbers and changelog
2. Tag release with semantic version
3. Build binaries for supported platforms
4. Update documentation site
5. Announce on relevant channels

---

Thank you for contributing to Sublation! Together we're building the future of dialectical AI computation.

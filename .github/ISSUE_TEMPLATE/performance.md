---
name: Performance Issue
about: Report performance regression or optimization opportunity
title: '[PERF] '
labels: ['performance']
assignees: ''
---

## Performance Issue Description

Clear description of the performance problem or optimization opportunity.

## Affected Components

- [ ] Kernel operations (`kernels/`)
- [ ] Memory management (`core/`, `runtime/`)
- [ ] Compilation (`compiler/`)
- [ ] Scheduling (`runtime/`)
- [ ] I/O operations
- [ ] Other: ___________

## Benchmark Results

### Current Performance

```bash
# Command used for benchmarking
go test -bench=BenchmarkXXX -benchmem ./package/

# Results
BenchmarkXXX-8    1000000    1200 ns/op    256 B/op    4 allocs/op
```

### Expected Performance

```bash
# Target or previous performance
BenchmarkXXX-8    2000000     600 ns/op      0 B/op    0 allocs/op
```

### Regression Information

If this is a performance regression:

- **Last known good version**: [e.g. v0.9.0, commit abc123]
- **First bad version**: [e.g. v1.0.0, commit def456]
- **Suspected commit**: [link to commit if known]

## Environment

- **CPU**: [e.g. Intel i7-12700K, AMD Ryzen 9 5900X, Apple M2]
- **Memory**: [e.g. 32GB DDR4-3200]
- **OS**: [e.g. Ubuntu 22.04, macOS 13.1]
- **Go Version**: [e.g. 1.22.2]
- **SIMD Support**: [e.g. AVX2, AVX-512, NEON]

## Profiling Data

### CPU Profile

```bash
# How to reproduce profiling
go test -bench=BenchmarkXXX -cpuprofile=cpu.prof
go tool pprof cpu.prof
```

Attach `cpu.prof` file or provide top functions:

```
(pprof) top10
Showing nodes accounting for 2.50s, 83.33% of 3.00s total
```

### Memory Profile

```bash
# Memory profiling command
go test -bench=BenchmarkXXX -memprofile=mem.prof
```

Key findings:

- Total allocations: XXX MB
- Peak memory usage: XXX MB
- Hot allocation sites: [describe]

## Analysis

### Root Cause (if known)

- [ ] Excessive memory allocations
- [ ] Cache misses / poor locality
- [ ] Unvectorized operations
- [ ] Lock contention
- [ ] Algorithm inefficiency
- [ ] Compiler optimization issue

### Impact Assessment

- **Performance degradation**: X% slower
- **Memory overhead**: X% more memory
- **Affected use cases**: [describe]
- **Frequency**: [always/sometimes/rarely]

## Proposed Solution

If you have optimization ideas:

### Approach 1: [Description]

```go
// Pseudo-code or actual implementation
func optimizedFunction() {
    // ...
}
```

**Trade-offs:**

- Pros: [benefits]
- Cons: [drawbacks]

### Approach 2: [Alternative]

...

## Reproducible Test Case

### Model/Code

```subs
# Minimal .subs model that demonstrates the issue
```

```bash
# Exact commands to reproduce
sublc model.subs output.subl
time sublrun output.subl < input.txt
```

### Benchmark Code

```go
func BenchmarkIssue(b *testing.B) {
    // Minimal benchmark that shows the problem
    for i := 0; i < b.N; i++ {
        // problematic operation
    }
}
```

## Additional Context

- Related issues or PRs
- Performance requirements or SLA
- Business impact if applicable
- Timeline/urgency

## Contribution

- [ ] I can help implement the optimization
- [ ] I can provide more profiling data
- [ ] I can test proposed solutions
- [ ] I need help with optimization

# Sublation Build System

This document describes the build process, compilation pipeline, and deployment procedures for the Sublation dialectical AI engine.

## Overview

Sublation uses a custom build system that produces optimized binaries for multiple architectures while maintaining zero external dependencies. The build process includes compilation, testing, and documentation generation.

## Build Components

### Core Binaries

- **`sublc`** - Sublation compiler (`.subs` → `.subl`)
- **`sublrun`** - Runtime execution engine
- **`sublperf`** - Performance benchmarking suite

### Build Script

The main build script `build.sh` handles:

```bash
#!/bin/bash
# Complete build process
./build.sh

# Individual components
./build.sh sublc      # Build only compiler
./build.sh sublrun    # Build only runtime
./build.sh test       # Run test suite
./build.sh bench      # Run benchmarks
./build.sh docs       # Generate documentation
```

## Architecture Support

### Primary Targets

- **Linux AMD64** - Primary development platform with AVX2 optimizations
- **Linux ARM64** - Server deployment with NEON optimizations
- **macOS AMD64** - Development support
- **macOS ARM64** - Apple Silicon support

### Build Tags

```go
//go:build amd64
// AMD64-specific SIMD implementations

//go:build arm64  
// ARM64-specific NEON implementations

//go:build !amd64 && !arm64
// Pure Go fallback implementations
```

## Compilation Pipeline

### Model Compilation

```bash
.subs specification → Parser → Validator → Optimizer → .subl binary
```

1. **Parse**: Convert DSL to internal graph representation
2. **Validate**: Check graph consistency and detect cycles
3. **Optimize**: Reorder nodes, compact memory layout
4. **Emit**: Generate cache-aligned binary format

### Example

```bash
# Compile with optimizations
sublc -O -validate examples/neural_network.subs model.subl

# Compile with debug symbols
sublc -debug -verbose examples/neural_network.subs model.subl

# Validate only (no output)
sublc -validate examples/neural_network.subs /dev/null
```

## Performance Optimization

### Compiler Flags

- `-O` - Enable layout optimizations for cache locality
- `-validate` - Perform graph validation (default: true)
- `-debug` - Include debug symbols and metadata
- `-verbose` - Show detailed compilation progress

### Runtime Optimizations

- **Memory Pre-allocation**: All buffers allocated at startup
- **SIMD Kernels**: Vectorized operations for mathematical kernels
- **Cache Alignment**: Data structures aligned to cache boundaries
- **Work Stealing**: Load balancing across worker goroutines

### Build Optimizations

```bash
# Production build with maximum optimization
go build -ldflags="-s -w" -o bin/sublc cmd/sublc/*.go

# Development build with debugging
go build -race -o bin/sublc cmd/sublc/*.go

# Profile-guided optimization (when available)
go build -pgo=cpu.prof -o bin/sublc cmd/sublc/*.go
```

## Testing Strategy

### Test Categories

1. **Unit Tests**: Individual component testing
2. **Integration Tests**: End-to-end pipeline testing  
3. **Benchmark Tests**: Performance regression detection
4. **Property Tests**: Invariant checking

### Test Execution

```bash
# Full test suite
go test ./... -v

# Race condition detection
go test -race ./...

# Coverage analysis
go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out

# Benchmark testing
go test -bench=. -benchmem ./kernels/

# Memory profiling
go test -bench=. -memprofile=mem.prof ./kernels/
```

### Continuous Integration

The CI pipeline runs on every commit:

```yaml
# .github/workflows/ci.yml
- Go 1.22.x and 1.23.x
- Ubuntu Latest and macOS Latest  
- Race detection enabled
- Benchmark regression detection
- Security scanning with gosec
```

## Cross-Compilation

### Supported Platforms

```bash
# Linux AMD64 (primary)
GOOS=linux GOARCH=amd64 go build -o bin/sublc-linux-amd64

# Linux ARM64 (server deployment)
GOOS=linux GOARCH=arm64 go build -o bin/sublc-linux-arm64

# macOS Universal
GOOS=darwin GOARCH=amd64 go build -o bin/sublc-darwin-amd64
GOOS=darwin GOARCH=arm64 go build -o bin/sublc-darwin-arm64

# Windows (experimental)
GOOS=windows GOARCH=amd64 go build -o bin/sublc-windows-amd64.exe
```

### Docker Builds

```dockerfile
# Multi-stage build for minimal deployment
FROM golang:1.22-alpine AS builder
WORKDIR /src
COPY . .
RUN ./build.sh

FROM alpine:latest
RUN apk --no-cache add ca-certificates
COPY --from=builder /src/bin/* /usr/local/bin/
```

## Documentation Generation

### API Documentation

```bash
# Local documentation server
go install golang.org/x/pkgsite/cmd/pkgsite@latest
pkgsite -http=localhost:8080

# Generate static documentation
godoc -html sublation > docs/api.html
```

### GitHub Pages

Documentation is automatically generated and deployed via GitHub Actions:

- **API Reference**: Go package documentation
- **Architecture Guide**: System design and principles
- **Examples**: Sample models and usage patterns
- **Performance**: Benchmark results and optimization guides

## Deployment

### Binary Distribution

```bash
# Create release binaries
./scripts/build-release.sh v1.0.0

# Package for distribution
tar -czf sublation-v1.0.0-linux-amd64.tar.gz bin/sublc bin/sublrun bin/sublperf
```

### Container Deployment

```bash
# Build container image
docker build -t sublation:latest .

# Run inference server
docker run -p 8080:8080 sublation:latest sublrun --server model.subl
```

## Development Workflow

### Local Development

```bash
# Clone and setup
git clone https://github.com/sbl8/sublation.git
cd sublation

# Install dependencies
go mod download

# Build development version
./build.sh

# Run tests
go test ./...

# Format code
go fmt ./...
goimports -w .

# Lint code  
golangci-lint run
```

### Release Process

1. Update version numbers and changelog
2. Run full test suite and benchmarks
3. Build release binaries for all platforms
4. Generate documentation and examples
5. Create Git tag and GitHub release
6. Update package repositories and documentation

## Troubleshooting

### Common Build Issues

**Assembly compilation errors on non-AMD64**:

```bash
# Use build tags to exclude assembly
go build -tags noasm
```

**Missing dependencies**:

```bash
# Verify Go modules
go mod verify
go mod tidy
```

**Performance regression**:

```bash
# Compare benchmarks
go test -bench=. -count=5 ./kernels/ > new.txt
benchstat old.txt new.txt
```

### Debug Builds

```bash
# Enable debug features
go build -tags debug -race -o bin/sublc-debug

# Profile memory usage
go build -o bin/sublc-prof
./bin/sublc-prof -cpuprofile=cpu.prof -memprofile=mem.prof
```

For more detailed build information, see the [development documentation](../docs/development.md).

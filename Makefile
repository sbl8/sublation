# Sublation Development Makefile
.PHONY: all build test bench clean lint docs help install

# Build configuration
BINARY_NAME=sublation
BUILD_DIR=bin
GO_VERSION=$(shell go version | cut -d' ' -f3)
GIT_COMMIT=$(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")
BUILD_TIME=$(shell date -u '+%Y-%m-%d_%H:%M:%S')

# Go build flags
LDFLAGS=-ldflags "-X main.version=$(GIT_COMMIT) -X main.buildTime=$(BUILD_TIME) -s -w"
BUILD_FLAGS=-trimpath $(LDFLAGS)

# Default target
all: clean lint test build

# Help target
help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Build targets
build: ## Build all binaries
	@echo "Building Sublation binaries..."
	@mkdir -p $(BUILD_DIR)
	go build $(BUILD_FLAGS) -o $(BUILD_DIR)/sublc ./cmd/sublc
	go build $(BUILD_FLAGS) -o $(BUILD_DIR)/sublrun ./cmd/sublrun
	go build $(BUILD_FLAGS) -o $(BUILD_DIR)/sublperf ./cmd/sublperf
	@echo "✓ Build complete"

install: ## Install binaries to GOPATH/bin
	go install $(BUILD_FLAGS) ./cmd/sublc
	go install $(BUILD_FLAGS) ./cmd/sublrun
	go install $(BUILD_FLAGS) ./cmd/sublperf

# Testing targets
test: ## Run all tests
	@echo "Running tests..."
	go test -race -coverprofile=coverage.out ./...
	@echo "✓ Tests complete"

test-verbose: ## Run tests with verbose output
	go test -race -v -coverprofile=coverage.out ./...

coverage: test ## Generate and open coverage report
	go tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report generated: coverage.html"

bench: ## Run benchmarks
	@echo "Running benchmarks..."
	go test -bench=. -benchmem ./kernels/ | tee benchmark.txt
	@echo "✓ Benchmarks complete"

bench-cpu: ## Run CPU profiling benchmarks
	go test -bench=. -cpuprofile=cpu.prof ./kernels/
	go tool pprof cpu.prof

bench-mem: ## Run memory profiling benchmarks
	go test -bench=. -memprofile=mem.prof ./kernels/
	go tool pprof mem.prof

# Code quality targets
lint: ## Run linter
	@echo "Running linter..."
	golangci-lint run
	@echo "✓ Lint complete"

vet: ## Run go vet
	go vet ./...

fmt: ## Format code
	go fmt ./...
	goimports -w .

# Documentation targets
docs: ## Generate documentation
	@echo "Generating documentation..."
	@mkdir -p docs/api
	godoc -http=:6060 &
	sleep 2
	curl -s http://localhost:6060/pkg/github.com/sbl8/sublation/ > docs/api/index.html || true
	pkill godoc || true
	@echo "✓ Documentation generated"

docs-serve: ## Serve documentation locally
	@echo "Starting documentation server on http://localhost:6060"
	godoc -http=:6060

# Development targets
dev: ## Development mode with file watching
	@echo "Starting development mode..."
	@echo "Watching for changes..."
	find . -name "*.go" | entr -r make build test

clean: ## Clean build artifacts
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR)
	rm -f coverage.out coverage.html
	rm -f cpu.prof mem.prof
	rm -f benchmark.txt
	go clean -cache
	@echo "✓ Clean complete"

# Model targets
compile-examples: build ## Compile example models
	@echo "Compiling example models..."
	$(BUILD_DIR)/sublc examples/neural_network.subs -o examples/neural_network.subl
	$(BUILD_DIR)/sublc examples/example.subs -o examples/example.subl
	@echo "✓ Examples compiled"

run-examples: compile-examples ## Run compiled examples
	@echo "Running examples..."
	$(BUILD_DIR)/sublrun examples/neural_network.subl
	$(BUILD_DIR)/sublrun examples/example.subl

# Release targets
release-prep: clean lint test build bench ## Prepare for release
	@echo "Release preparation complete"
	@echo "Git commit: $(GIT_COMMIT)"
	@echo "Go version: $(GO_VERSION)"
	@echo "Build time: $(BUILD_TIME)"

# CI targets
ci: lint test build ## CI pipeline
	@echo "CI pipeline complete"

# Check dependencies
deps: ## Check and tidy dependencies
	go mod tidy
	go mod verify
	go mod download

# Performance analysis
profile: ## Run performance analysis
	@echo "Running performance analysis..."
	go test -bench=. -cpuprofile=cpu.prof -memprofile=mem.prof ./kernels/
	@echo "Profiles generated: cpu.prof, mem.prof"
	@echo "Use 'go tool pprof cpu.prof' to analyze CPU usage"
	@echo "Use 'go tool pprof mem.prof' to analyze memory usage"

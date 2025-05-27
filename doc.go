// Package sublation implements a dialectical AI engine using compositional memory-compute units.
//
// Sublation reimagines neural computation through "sublates" - self-contained memory blocks
// with local transformation kernels that replace traditional tensor architectures. The design
// follows dialectical principles where each computation step simultaneously preserves previous
// state while proposing new state, enabling both stability and change.
//
// # Architecture Overview
//
// The Sublation engine consists of several key components:
//
//   - Sublates: Cache-aligned structs with dual buffers (PayloadPrev, PayloadProp)
//   - Kernels: Pure functions operating in-place with zero allocations
//   - Runtime: Streaming scheduler with pre-allocated arena memory
//   - Compiler: DSL parser that generates optimized binary models
//
// # Performance Characteristics
//
// Sublation achieves high performance through:
//
//   - Zero-allocation execution: All memory pre-planned at startup
//   - SIMD optimization: Hand-tuned AVX2/NEON assembly for hot paths
//   - Cache efficiency: Memory layouts optimized for spatial locality
//   - Lock-free concurrency: Dual-buffer updates without synchronization
//
// # Basic Usage
//
//	// Compile a model specification
//	sublc -O examples/neural_network.subs model.subl
//
//	// Load and execute
//	engine, err := runtime.Load("model.subl")
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	input := []float32{1.0, 0.5, 0.75, 1.0}
//	output, err := engine.Execute(input)
//	if err != nil {
//	    log.Fatal(err)
//	}
//
// # Package Structure
//
//   - core: Fundamental Sublate primitives and memory management
//   - kernels: SIMD-optimized mathematical operations
//   - runtime: Execution engine and streaming scheduler
//   - compiler: Model compilation and optimization
//   - model: Graph representation and serialization
//   - cmd: Command-line tools (sublc, sublrun, sublperf)
//
// For more information, see the documentation at https://pkg.go.dev/sublation
// and the project repository at https://github.com/sbl8/sublation
package sublation

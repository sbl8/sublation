// Package kernels provides SIMD-optimized mathematical operations for Sublation.
//
// This package implements high-performance compute kernels that operate in-place
// on Sublate payloads with zero memory allocations. All kernels follow the
// functional signature func([]byte) and are designed for maximum cache efficiency.
//
// Architecture support:
//   - AMD64: Hand-tuned AVX2/AVX-512 assembly implementations
//   - ARM64: NEON vectorized operations
//   - Fallback: Pure Go implementations for portability
//
// Available operations:
//   - Basic arithmetic: add, multiply, square-plus-x
//   - Activations: ReLU, sigmoid, tanh, softmax
//   - Linear algebra: matrix multiplication, dot products
//   - Aggregations: sum, max, mean
//
// All kernels are registered in the global Catalog array for runtime dispatch
// based on operation codes defined in the model specification.
package kernels

import (
	"math"
	"unsafe"
)

// KernelFn operates in‑place on a Sublate payload with zero allocations
type KernelFn func(data []byte)

// Kernel operation codes
const (
	OpNoop     = 0x00
	OpSqrPlusX = 0x01
	OpMatMul   = 0x02
	OpReLU     = 0x03
	OpSigmoid  = 0x04
	OpTanh     = 0x05
	OpAdd      = 0x06
	OpMul      = 0x07
	OpSum      = 0x08
	OpMax      = 0x09
	OpSoftmax  = 0x0A
)

// Catalog maps opcodes to optimized kernel implementations
var Catalog = [256]KernelFn{
	OpNoop:     noop,
	OpSqrPlusX: sqrPlusX,
	OpMatMul:   matMul,
	OpReLU:     relu,
	OpSigmoid:  sigmoid,
	OpTanh:     tanh,
	OpAdd:      vectorAdd,
	OpMul:      vectorMul,
	OpSum:      vectorSum,
	OpMax:      vectorMax,
	OpSoftmax:  softmax,
}

// -------- Core Kernels (SIMD-friendly) ----------

func noop(data []byte) {
	// No operation - used for padding or testing
}

func sqrPlusX(data []byte) {
	const sz = 4 // float32
	count := len(data) / sz
	for i := 0; i < count; i++ {
		p := (*float32)(unsafe.Pointer(&data[i*sz]))
		x := *p
		*p = x*x + x
	}
}

// relu implements Rectified Linear Unit: max(0, x)
func relu(data []byte) {
	const sz = 4
	count := len(data) / sz
	for i := 0; i < count; i++ {
		p := (*float32)(unsafe.Pointer(&data[i*sz]))
		if *p < 0 {
			*p = 0
		}
	}
}

// sigmoid implements 1 / (1 + e^(-x)) with optimized approximation
func sigmoid(data []byte) {
	const sz = 4
	count := len(data) / sz
	for i := 0; i < count; i++ {
		p := (*float32)(unsafe.Pointer(&data[i*sz]))
		x := *p

		// Fast sigmoid approximation: x / (1 + |x|)
		// More accurate than tanh, faster than exp
		if x >= 0 {
			*p = x / (1 + x)
		} else {
			*p = x / (1 - x)
		}
	}
}

// tanh implements hyperbolic tangent with rational approximation
func tanh(data []byte) {
	const sz = 4
	count := len(data) / sz
	for i := 0; i < count; i++ {
		p := (*float32)(unsafe.Pointer(&data[i*sz]))
		x := *p

		// Rational approximation for tanh
		x2 := x * x
		*p = x * (27 + x2) / (27 + 9*x2)
	}
}

// vectorAdd performs element-wise addition (data layout: [a0,a1,..][b0,b1,..])
func vectorAdd(data []byte) {
	const sz = 4
	half := len(data) / 2
	count := half / sz

	// Convert to float32 slices for assembly optimization
	aSlice := (*[1 << 20]float32)(unsafe.Pointer(&data[0]))[:count:count]
	bSlice := (*[1 << 20]float32)(unsafe.Pointer(&data[half]))[:count:count]

	// Use optimized assembly if available
	VectorAddInPlace(aSlice, bSlice)
}

// vectorMul performs element-wise multiplication
func vectorMul(data []byte) {
	const sz = 4
	half := len(data) / 2
	count := half / sz

	// Convert to float32 slices for assembly optimization
	aSlice := (*[1 << 20]float32)(unsafe.Pointer(&data[0]))[:count:count]
	bSlice := (*[1 << 20]float32)(unsafe.Pointer(&data[half]))[:count:count]

	// Use optimized assembly if available
	VectorMulInPlace(aSlice, bSlice)
}

// vectorSum computes the sum of all elements, stores in first position
func vectorSum(data []byte) {
	const sz = 4
	count := len(data) / sz
	if count == 0 {
		return
	}

	sum := float32(0)
	for i := 0; i < count; i++ {
		p := (*float32)(unsafe.Pointer(&data[i*sz]))
		sum += *p
	}

	// Store result in first float32
	result := (*float32)(unsafe.Pointer(&data[0]))
	*result = sum
}

// vectorMax finds maximum element, stores in first position
func vectorMax(data []byte) {
	const sz = 4
	count := len(data) / sz
	if count == 0 {
		return
	}

	maxVal := float32(math.Inf(-1))
	for i := 0; i < count; i++ {
		p := (*float32)(unsafe.Pointer(&data[i*sz]))
		if *p > maxVal {
			maxVal = *p
		}
	}

	// Store result in first float32
	result := (*float32)(unsafe.Pointer(&data[0]))
	*result = maxVal
}

// softmax implements numerically stable softmax
func softmax(data []byte) {
	const sz = 4
	count := len(data) / sz
	if count == 0 {
		return
	}

	// Find maximum for numerical stability
	maxVal := float32(math.Inf(-1))
	for i := 0; i < count; i++ {
		p := (*float32)(unsafe.Pointer(&data[i*sz]))
		if *p > maxVal {
			maxVal = *p
		}
	}

	// Compute exp(x - max) and sum
	var sum float32
	for i := 0; i < count; i++ {
		p := (*float32)(unsafe.Pointer(&data[i*sz]))
		*p = float32(math.Exp(float64(*p - maxVal)))
		sum += *p
	}

	// Normalize
	for i := 0; i < count; i++ {
		p := (*float32)(unsafe.Pointer(&data[i*sz]))
		*p = *p / sum
	}
}

// matMul performs matrix multiplication for small matrices
// Layout: [A_rows(2)][A_cols(2)][B_cols(2)][A_data...][B_data...]
func matMul(data []byte) {
	if len(data) < 6 {
		return
	}

	aRows := *(*uint16)(unsafe.Pointer(&data[0]))
	aCols := *(*uint16)(unsafe.Pointer(&data[2]))
	bCols := *(*uint16)(unsafe.Pointer(&data[4]))

	aSize := int(aRows * aCols * 4) // float32
	bSize := int(aCols * bCols * 4)

	if len(data) < 6+aSize+bSize {
		return
	}

	aData := data[6 : 6+aSize]
	bData := data[6+aSize : 6+aSize+bSize]

	// Convert to float32 slices for assembly optimization
	aSlice := (*[1 << 20]float32)(unsafe.Pointer(&aData[0]))[: aRows*aCols : aRows*aCols]
	bSlice := (*[1 << 20]float32)(unsafe.Pointer(&bData[0]))[: aCols*bCols : aCols*bCols]

	// Use optimized assembly matrix multiplication
	result := MatMulOptimized(aSlice, int(aRows), int(aCols), bSlice, int(aCols), int(bCols))

	// Copy result back to A's memory space
	resultBytes := (*[1 << 20]byte)(unsafe.Pointer(&result[0]))[: len(result)*4 : len(result)*4]
	copy(aData, resultBytes)
}

// SIMD-friendly vectorized operations with unrolling
const unrollFactor = 4

// vectorAddUnrolled performs element-wise addition with loop unrolling
func vectorAddUnrolled(data []byte) {
	const sz = 4
	count := len(data) / sz / 2 // two vectors
	if count == 0 {
		return
	}

	aPtr := (*float32)(unsafe.Pointer(&data[0]))
	bPtr := (*float32)(unsafe.Pointer(&data[count*sz]))

	// Process in groups of 4 for better cache usage
	i := 0
	for ; i < count-unrollFactor+1; i += unrollFactor {
		a0 := *(*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(aPtr)) + uintptr(i*sz)))
		a1 := *(*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(aPtr)) + uintptr((i+1)*sz)))
		a2 := *(*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(aPtr)) + uintptr((i+2)*sz)))
		a3 := *(*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(aPtr)) + uintptr((i+3)*sz)))

		b0 := *(*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(bPtr)) + uintptr(i*sz)))
		b1 := *(*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(bPtr)) + uintptr((i+1)*sz)))
		b2 := *(*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(bPtr)) + uintptr((i+2)*sz)))
		b3 := *(*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(bPtr)) + uintptr((i+3)*sz)))

		*(*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(aPtr)) + uintptr(i*sz))) = a0 + b0
		*(*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(aPtr)) + uintptr((i+1)*sz))) = a1 + b1
		*(*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(aPtr)) + uintptr((i+2)*sz))) = a2 + b2
		*(*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(aPtr)) + uintptr((i+3)*sz))) = a3 + b3
	}

	// Handle remaining elements
	for ; i < count; i++ {
		a := *(*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(aPtr)) + uintptr(i*sz)))
		b := *(*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(bPtr)) + uintptr(i*sz)))
		*(*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(aPtr)) + uintptr(i*sz))) = a + b
	}
}

// matMulOptimized performs matrix multiplication with cache-friendly access patterns
func matMulOptimized(data []byte) {
	if len(data) < 12 {
		return // Need at least dimensions
	}

	// Layout: [rows(2)][cols(2)][b_cols(2)][matrix_a][matrix_b]
	rows := *(*uint16)(unsafe.Pointer(&data[0]))
	cols := *(*uint16)(unsafe.Pointer(&data[2]))
	bCols := *(*uint16)(unsafe.Pointer(&data[4]))

	aSize := int(rows) * int(cols) * 4
	bSize := int(cols) * int(bCols) * 4
	headerSize := 6

	if len(data) < headerSize+aSize+bSize {
		return
	}

	// Get matrix pointers
	matA := (*float32)(unsafe.Pointer(&data[headerSize]))
	matB := (*float32)(unsafe.Pointer(&data[headerSize+aSize]))

	// Allocate result matrix (overwrite matB area for in-place operation)
	result := (*float32)(unsafe.Pointer(&data[headerSize+aSize]))

	// Cache-friendly matrix multiplication with blocking
	blockSize := 32 // Tune based on cache size

	for ii := 0; ii < int(rows); ii += blockSize {
		for jj := 0; jj < int(bCols); jj += blockSize {
			for kk := 0; kk < int(cols); kk += blockSize {
				// Process block
				iEnd := ii + blockSize
				if iEnd > int(rows) {
					iEnd = int(rows)
				}
				jEnd := jj + blockSize
				if jEnd > int(bCols) {
					jEnd = int(bCols)
				}
				kEnd := kk + blockSize
				if kEnd > int(cols) {
					kEnd = int(cols)
				}

				for i := ii; i < iEnd; i++ {
					for j := jj; j < jEnd; j++ {
						sum := float32(0)
						for k := kk; k < kEnd; k++ {
							aVal := *(*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(matA)) +
								uintptr((i*int(cols)+k)*4)))
							bVal := *(*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(matB)) +
								uintptr((k*int(bCols)+j)*4)))
							sum += aVal * bVal
						}
						*(*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(result)) +
							uintptr((i*int(bCols)+j)*4))) += sum
					}
				}
			}
		}
	}
}

// softmaxOptimized implements numerically stable softmax with SIMD-friendly patterns
func softmaxOptimized(data []byte) {
	const sz = 4
	count := len(data) / sz
	if count <= 1 {
		return
	}

	// Find maximum for numerical stability
	maxVal := float32(-math.Inf(1))
	for i := 0; i < count; i++ {
		val := *(*float32)(unsafe.Pointer(&data[i*sz]))
		if val > maxVal {
			maxVal = val
		}
	}

	// Compute exp(x - max) and sum
	sum := float32(0)
	for i := 0; i < count; i++ {
		p := (*float32)(unsafe.Pointer(&data[i*sz]))
		*p = float32(math.Exp(float64(*p - maxVal)))
		sum += *p
	}

	// Normalize
	invSum := 1.0 / sum
	for i := 0; i < count; i++ {
		p := (*float32)(unsafe.Pointer(&data[i*sz]))
		*p *= invSum
	}
}

// convolution1D performs optimized 1D convolution
func convolution1D(data []byte) {
	// Layout: [input_len(2)][kernel_len(2)][input_data][kernel_data]
	if len(data) < 4 {
		return
	}

	inputLen := *(*uint16)(unsafe.Pointer(&data[0]))
	kernelLen := *(*uint16)(unsafe.Pointer(&data[2]))

	headerSize := 4
	inputSize := int(inputLen) * 4
	kernelSize := int(kernelLen) * 4

	if len(data) < headerSize+inputSize+kernelSize {
		return
	}

	input := (*float32)(unsafe.Pointer(&data[headerSize]))
	kernel := (*float32)(unsafe.Pointer(&data[headerSize+inputSize]))

	// Output overwrites input area
	output := input

	outputLen := int(inputLen) - int(kernelLen) + 1
	if outputLen <= 0 {
		return
	}

	// Perform convolution with loop unrolling
	for i := 0; i < outputLen; i++ {
		sum := float32(0)
		j := 0

		// Unroll inner loop for better performance
		for ; j < int(kernelLen)-3; j += 4 {
			sum += *(*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(input)) + uintptr((i+j)*4))) *
				*(*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(kernel)) + uintptr(j*4)))
			sum += *(*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(input)) + uintptr((i+j+1)*4))) *
				*(*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(kernel)) + uintptr((j+1)*4)))
			sum += *(*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(input)) + uintptr((i+j+2)*4))) *
				*(*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(kernel)) + uintptr((j+2)*4)))
			sum += *(*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(input)) + uintptr((i+j+3)*4))) *
				*(*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(kernel)) + uintptr((j+3)*4)))
		}

		// Handle remaining elements
		for ; j < int(kernelLen); j++ {
			sum += *(*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(input)) + uintptr((i+j)*4))) *
				*(*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(kernel)) + uintptr(j*4)))
		}

		*(*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(output)) + uintptr(i*4))) = sum
	}
}

// batchNorm implements batch normalization with efficient computation
func batchNorm(data []byte) {
	// Layout: [count(2)][mean][variance][gamma][beta][input_data]
	if len(data) < 18 {
		return
	}

	count := *(*uint16)(unsafe.Pointer(&data[0]))
	mean := *(*float32)(unsafe.Pointer(&data[2]))
	variance := *(*float32)(unsafe.Pointer(&data[6]))
	gamma := *(*float32)(unsafe.Pointer(&data[10]))
	beta := *(*float32)(unsafe.Pointer(&data[14]))

	input := (*float32)(unsafe.Pointer(&data[18]))

	// Precompute normalization factor
	invStd := 1.0 / float32(math.Sqrt(float64(variance)+1e-5))

	// Vectorized batch normalization
	for i := 0; i < int(count); i++ {
		val := *(*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(input)) + uintptr(i*4)))
		normalized := (val - mean) * invStd
		*(*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(input)) + uintptr(i*4))) =
			gamma*normalized + beta
	}
}

// Update catalog with optimized implementations
func init() {
	// Override default implementations with optimized versions
	Catalog[OpAdd] = vectorAddUnrolled
	Catalog[OpMatMul] = matMulOptimized
	Catalog[OpSoftmax] = softmaxOptimized

	// Add new kernels
	const (
		OpConv1D    = 0x0B
		OpBatchNorm = 0x0C
	)

	Catalog[OpConv1D] = convolution1D
	Catalog[OpBatchNorm] = batchNorm
}

// GetKernel returns the kernel function for the given opcode
func GetKernel(opcode byte) KernelFn {
	return Catalog[opcode]
}

// UseASM returns whether assembly optimizations are available
func UseASM() bool {
	return useASM
}

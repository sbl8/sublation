//go:build !amd64

package kernels

// useASM indicates whether to use assembly optimizations (disabled for non-AMD64)
const useASM = false

// Fallback implementations for non-AMD64 architectures

// VectorAddOptimized performs vectorized addition using pure Go
func VectorAddOptimized(a, b []float32) []float32 {
	if len(a) != len(b) {
		panic("vector length mismatch")
	}

	result := make([]float32, len(a))
	for i := range a {
		result[i] = a[i] + b[i]
	}
	return result
}

// VectorMulOptimized performs vectorized multiplication using pure Go
func VectorMulOptimized(a, b []float32) []float32 {
	if len(a) != len(b) {
		panic("vector length mismatch")
	}

	result := make([]float32, len(a))
	for i := range a {
		result[i] = a[i] * b[i]
	}
	return result
}

// VectorDotOptimized computes dot product using pure Go
func VectorDotOptimized(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("vector length mismatch")
	}

	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// MatMulOptimized performs matrix multiplication using pure Go
func MatMulOptimized(a []float32, aRows, aCols int, b []float32, bRows, bCols int) []float32 {
	if aCols != bRows {
		panic("matrix dimension mismatch")
	}
	if len(a) < aRows*aCols || len(b) < bRows*bCols {
		panic("matrix data insufficient")
	}

	result := make([]float32, aRows*bCols)

	// Cache-friendly matrix multiplication
	for i := 0; i < aRows; i++ {
		for j := 0; j < bCols; j++ {
			var sum float32
			for k := 0; k < aCols; k++ {
				sum += a[i*aCols+k] * b[k*bCols+j]
			}
			result[i*bCols+j] = sum
		}
	}

	return result
}

// VectorAddInPlace performs in-place vector addition
func VectorAddInPlace(a, b []float32) {
	if len(a) != len(b) {
		panic("vector length mismatch")
	}

	for i := range a {
		a[i] += b[i]
	}
}

// VectorMulInPlace performs in-place vector multiplication
func VectorMulInPlace(a, b []float32) {
	if len(a) != len(b) {
		panic("vector length mismatch")
	}

	for i := range a {
		a[i] *= b[i]
	}
}

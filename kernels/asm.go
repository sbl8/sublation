//go:build amd64

package kernels

// Assembly function declarations for AMD64
//
//go:noescape
func vectorAddASM(a, b, result []float32)

//go:noescape
func vectorMulASM(a, b, result []float32)

//go:noescape
func vectorDotASM(a, b []float32) float32

//go:noescape
func matMulASM(a []float32, aRows, aCols int, b []float32, bCols int, result []float32)

//go:noescape
func axpyASM(alpha float32, x, y []float32)

//go:noescape
func gemvASM(alpha float32, a []float32, rows, cols int, x []float32, beta float32, y []float32)

// useASM indicates whether to use assembly optimizations
const useASM = true

// High-level optimized kernel functions using assembly when available

// VectorAddOptimized performs vectorized addition with assembly acceleration
func VectorAddOptimized(a, b []float32) []float32 {
	if len(a) != len(b) {
		panic("vector length mismatch")
	}

	result := make([]float32, len(a))
	if useASM && len(a) > 0 {
		vectorAddASM(a, b, result)
	} else {
		// Fallback to pure Go
		for i := range a {
			result[i] = a[i] + b[i]
		}
	}
	return result
}

// VectorMulOptimized performs vectorized multiplication with assembly acceleration
func VectorMulOptimized(a, b []float32) []float32 {
	if len(a) != len(b) {
		panic("vector length mismatch")
	}

	result := make([]float32, len(a))
	if useASM && len(a) > 0 {
		vectorMulASM(a, b, result)
	} else {
		// Fallback to pure Go
		for i := range a {
			result[i] = a[i] * b[i]
		}
	}
	return result
}

// VectorDotOptimized computes dot product with assembly acceleration
func VectorDotOptimized(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("vector length mismatch")
	}

	if useASM && len(a) > 0 {
		return vectorDotASM(a, b)
	}

	// Fallback to pure Go
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// MatMulOptimized performs matrix multiplication with assembly acceleration
func MatMulOptimized(a []float32, aRows, aCols int, b []float32, bRows, bCols int) []float32 {
	if aCols != bRows {
		panic("matrix dimension mismatch")
	}
	if len(a) < aRows*aCols || len(b) < bRows*bCols {
		panic("matrix data insufficient")
	}

	result := make([]float32, aRows*bCols)

	if useASM {
		matMulASM(a, aRows, aCols, b, bCols, result)
	} else {
		// Fallback to pure Go with cache-friendly access
		for i := 0; i < aRows; i++ {
			for j := 0; j < bCols; j++ {
				var sum float32
				for k := 0; k < aCols; k++ {
					sum += a[i*aCols+k] * b[k*bCols+j]
				}
				result[i*bCols+j] = sum
			}
		}
	}

	return result
}

// In-place operations for zero-allocation patterns

// VectorAddInPlace performs in-place vector addition (a = a + b)
func VectorAddInPlace(a, b []float32) {
	if len(a) != len(b) {
		panic("vector length mismatch")
	}

	if useASM && len(a) > 0 {
		vectorAddASM(a, b, a) // Use a as both input and output
	} else {
		for i := range a {
			a[i] += b[i]
		}
	}
}

// VectorMulInPlace performs in-place vector multiplication (a = a * b)
func VectorMulInPlace(a, b []float32) {
	if len(a) != len(b) {
		panic("vector length mismatch")
	}

	if useASM && len(a) > 0 {
		vectorMulASM(a, b, a) // Use a as both input and output
	} else {
		for i := range a {
			a[i] *= b[i]
		}
	}
}

// AxpyOptimized performs y = alpha*x + y with assembly acceleration
func AxpyOptimized(alpha float32, x, y []float32) {
	if len(x) != len(y) {
		panic("vector length mismatch")
	}

	if useASM && len(x) > 0 {
		axpyASM(alpha, x, y)
	} else {
		// Fallback to pure Go
		for i := range x {
			y[i] = alpha*x[i] + y[i]
		}
	}
}

// GemvOptimized performs y = alpha*A*x + beta*y with assembly acceleration
func GemvOptimized(alpha float32, a []float32, rows, cols int, x []float32, beta float32, y []float32) {
	if len(a) < rows*cols {
		panic("matrix data insufficient")
	}
	if len(x) != cols {
		panic("vector x length mismatch")
	}
	if len(y) != rows {
		panic("vector y length mismatch")
	}

	if useASM {
		gemvASM(alpha, a, rows, cols, x, beta, y)
	} else {
		// Fallback to pure Go
		for i := 0; i < rows; i++ {
			sum := float32(0)
			for j := 0; j < cols; j++ {
				sum += a[i*cols+j] * x[j]
			}
			y[i] = alpha*sum + beta*y[i]
		}
	}
}

// Zero-allocation kernel wrappers for Sublate operations

// ApplyKernel applies an operation kernel directly to Sublate buffers
func ApplyKernel(kernel func([]float32), buf []float32, offset, length int) {
	if offset+length > len(buf) {
		panic("buffer bounds exceeded")
	}
	kernel(buf[offset : offset+length])
}

// ElementwiseOp performs element-wise operations between two buffers
func ElementwiseOp(op func([]float32, []float32), a, b []float32, aOffset, bOffset, length int) {
	if aOffset+length > len(a) || bOffset+length > len(b) {
		panic("buffer bounds exceeded")
	}
	op(a[aOffset:aOffset+length], b[bOffset:bOffset+length])
}

package kernels

import (
	"math"
	"math/rand"
	"testing"
	"time"
)

const floatTolerance = 1e-6

// Helper to generate a random float32 slice
func randomSlice(n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = rand.Float32()*2 - 1 // Random floats between -1 and 1
	}
	return s
}

// Helper to compare two float32 slices with tolerance
func slicesEqual(a, b []float32, tolerance float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Abs(float64(a[i]-b[i])) > float64(tolerance) {
			return false
		}
	}
	return true
}

// Helper to compare two float32 values with tolerance
func floatsEqual(a, b float32, tolerance float32) bool {
	return math.Abs(float64(a-b)) <= float64(tolerance)
}

// Go reference for vectorAdd
func vectorAddGo(a, b, result []float32) {
	for i := 0; i < len(a); i++ {
		result[i] = a[i] + b[i]
	}
}

// Go reference for vectorMul
func vectorMulGo(a, b, result []float32) {
	for i := 0; i < len(a); i++ {
		result[i] = a[i] * b[i]
	}
}

// Go reference for vectorDot
func vectorDotGo(a, b []float32) float32 {
	var sum float32
	for i := 0; i < len(a); i++ {
		sum += a[i] * b[i]
	}
	return sum
}

// Go reference for axpy
func axpyGo(alpha float32, x, y []float32) {
	for i := 0; i < len(x); i++ {
		y[i] = alpha*x[i] + y[i]
	}
}

// Go reference for matMul
func matMulGo(a []float32, aRows, aCols int, b []float32, bCols int, result []float32) {
	// result is M x N, a is M x K, b is K x N
	// M = aRows, K = aCols, N = bCols
	for i := 0; i < aRows; i++ { // M rows
		for j := 0; j < bCols; j++ { // N cols
			var sum float32
			for k := 0; k < aCols; k++ { // K inner
				sum += a[i*aCols+k] * b[k*bCols+j]
			}
			result[i*bCols+j] = sum
		}
	}
}

// Go reference for gemv
func gemvGo(alpha float32, a []float32, rows, cols int, x []float32, beta float32, y []float32) {
	// y = alpha * A * x + beta * y
	// A is M x N (rows x cols)
	// x is N x 1
	// y is M x 1
	for i := 0; i < rows; i++ { // M rows
		var dotProduct float32
		for j := 0; j < cols; j++ { // N cols
			dotProduct += a[i*cols+j] * x[j]
		}
		y[i] = alpha*dotProduct + beta*y[i]
	}
}

func TestMain(m *testing.M) {
	rand.Seed(time.Now().UnixNano())
	m.Run()
}

func TestVectorAddASM(t *testing.T) {
	sizes := []int{0, 1, 7, 8, 15, 16, 100}
	for _, n := range sizes {
		a := randomSlice(n)
		b := randomSlice(n)
		resultAsm := make([]float32, n)
		resultGo := make([]float32, n)

		vectorAddASM(a, b, resultAsm)
		vectorAddGo(a, b, resultGo)

		if !slicesEqual(resultAsm, resultGo, floatTolerance) {
			t.Errorf("VectorAddASM failed for n=%d. ASM: %v, Go: %v", n, resultAsm, resultGo)
		}
	}
}

func TestVectorMulASM(t *testing.T) {
	sizes := []int{0, 1, 7, 8, 15, 16, 100}
	for _, n := range sizes {
		a := randomSlice(n)
		b := randomSlice(n)
		resultAsm := make([]float32, n)
		resultGo := make([]float32, n)

		vectorMulASM(a, b, resultAsm)
		vectorMulGo(a, b, resultGo)

		if !slicesEqual(resultAsm, resultGo, floatTolerance) {
			t.Errorf("VectorMulASM failed for n=%d. ASM: %v, Go: %v", n, resultAsm, resultGo)
		}
	}
}

func TestVectorDotASM(t *testing.T) {
	sizes := []int{0, 1, 7, 8, 15, 16, 100}
	for _, n := range sizes {
		a := randomSlice(n)
		b := randomSlice(n)

		resultAsm := vectorDotASM(a, b)
		resultGo := vectorDotGo(a, b)

		if !floatsEqual(resultAsm, resultGo, floatTolerance*float32(n+1)) { // Tolerance might need to scale with n for dot products
			t.Errorf("VectorDotASM failed for n=%d. ASM: %f, Go: %f", n, resultAsm, resultGo)
		}
	}
}

func TestAxpyASM(t *testing.T) {
	sizes := []int{0, 1, 7, 8, 15, 16, 100}
	alpha := rand.Float32()*2 - 1
	for _, n := range sizes {
		x := randomSlice(n)
		yAsm := randomSlice(n)
		yGo := make([]float32, n)
		copy(yGo, yAsm)

		axpyASM(alpha, x, yAsm)
		axpyGo(alpha, x, yGo)

		if !slicesEqual(yAsm, yGo, floatTolerance) {
			t.Errorf("AxpyASM failed for n=%d, alpha=%f. ASM: %v, Go: %v", n, alpha, yAsm, yGo)
		}
	}
}

func TestMatMulASM(t *testing.T) {
	testCases := []struct {
		m, k, n int
	}{
		{1, 1, 1}, {2, 2, 2}, {3, 4, 5}, {8, 8, 8},
		{7, 7, 7}, {10, 1, 10}, {10, 10, 1}, {16, 16, 16},
		{15, 17, 13}, {0, 5, 5}, {5, 0, 5}, {5, 5, 0}, {0, 0, 0},
	}

	for _, tc := range testCases {
		if tc.m == 0 || tc.k == 0 || tc.n == 0 { // Handle zero dimensions
			a := make([]float32, 0)
			b := make([]float32, 0)
			resultAsm := make([]float32, 0)
			resultGo := make([]float32, 0)
			if tc.m*tc.n > 0 {
				resultAsm = make([]float32, tc.m*tc.n)
				resultGo = make([]float32, tc.m*tc.n)
			}
			if tc.m*tc.k > 0 {
				a = make([]float32, tc.m*tc.k)
			}
			if tc.k*tc.n > 0 {
				b = make([]float32, tc.k*tc.n)
			}

			matMulASM(a, tc.m, tc.k, b, tc.n, resultAsm)
			matMulGo(a, tc.m, tc.k, b, tc.n, resultGo)

			if !slicesEqual(resultAsm, resultGo, floatTolerance) {
				t.Errorf("MatMulASM failed for M=%d, K=%d, N=%d (zero case). ASM: %v, Go: %v", tc.m, tc.k, tc.n, resultAsm, resultGo)
			}
			continue
		}

		a := randomSlice(tc.m * tc.k)
		b := randomSlice(tc.k * tc.n)
		resultAsm := make([]float32, tc.m*tc.n)
		resultGo := make([]float32, tc.m*tc.n)

		matMulASM(a, tc.m, tc.k, b, tc.n, resultAsm)
		matMulGo(a, tc.m, tc.k, b, tc.n, resultGo)

		if !slicesEqual(resultAsm, resultGo, floatTolerance*float32(tc.k)) { // Tolerance might scale with K
			t.Errorf("MatMulASM failed for M=%d, K=%d, N=%d. \nASM: %v\n Go: %v\nDiff: %v", tc.m, tc.k, tc.n, resultAsm, resultGo, diffSlices(resultAsm, resultGo))
		}
	}
}

func TestGemvASM(t *testing.T) {
	testCases := []struct {
		rows, cols int
	}{
		{1, 1}, {2, 2}, {8, 8}, {7, 7}, {10, 1}, {1, 10}, {16, 16},
		{15, 17}, {0, 5}, {5, 0}, {0, 0},
	}
	alpha := rand.Float32()*2 - 1
	beta := rand.Float32()*2 - 1

	for _, tc := range testCases {
		if tc.rows == 0 || tc.cols == 0 { // Handle zero dimensions
			a := make([]float32, 0)
			x := make([]float32, 0)
			yAsm := make([]float32, 0)
			yGo := make([]float32, 0)

			if tc.rows*tc.cols > 0 {
				a = make([]float32, tc.rows*tc.cols)
			}
			if tc.cols > 0 {
				x = make([]float32, tc.cols)
			}
			if tc.rows > 0 {
				yAsm = make([]float32, tc.rows)
				yGo = make([]float32, tc.rows)
			}

			gemvASM(alpha, a, tc.rows, tc.cols, x, beta, yAsm)
			gemvGo(alpha, a, tc.rows, tc.cols, x, beta, yGo)

			if !slicesEqual(yAsm, yGo, floatTolerance) {
				t.Errorf("GemvASM failed for rows=%d, cols=%d (zero case). ASM: %v, Go: %v", tc.rows, tc.cols, yAsm, yGo)
			}
			continue
		}

		a := randomSlice(tc.rows * tc.cols)
		x := randomSlice(tc.cols)
		yAsm := randomSlice(tc.rows)
		yGo := make([]float32, tc.rows)
		copy(yGo, yAsm)

		gemvASM(alpha, a, tc.rows, tc.cols, x, beta, yAsm)
		gemvGo(alpha, a, tc.rows, tc.cols, x, beta, yGo)

		if !slicesEqual(yAsm, yGo, floatTolerance*float32(tc.cols+1)) { // Tolerance might scale with cols
			t.Errorf("GemvASM failed for rows=%d, cols=%d. Alpha=%f, Beta=%f. \nASM: %v\n Go: %v\nDiff: %v", tc.rows, tc.cols, alpha, beta, yAsm, yGo, diffSlices(yAsm, yGo))
		}
	}
}

// Helper to show differences for debugging
func diffSlices(a, b []float32) []float32 {
	if len(a) != len(b) {
		return nil // Or handle error appropriately
	}
	diff := make([]float32, len(a))
	for i := range a {
		diff[i] = a[i] - b[i]
	}
	return diff
}

// Ensure assembly functions are declared for the linker
// These are dummy calls, actual functions are in asm_amd64.s
var (
	_ = vectorAddASM
	_ = vectorMulASM
	_ = vectorDotASM
	_ = axpyASM
	_ = matMulASM
	_ = gemvASM
)

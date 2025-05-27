package kernels

import (
	"math/rand"
	"testing"
	"unsafe"
)

func init() {
	// No explicit seeding needed as of Go 1.20 - rand is automatically seeded
}

// Helper function to generate random float32 slices
func generateRandomFloat32(size int) []float32 {
	data := make([]float32, size)
	for i := range data {
		data[i] = rand.Float32()*200 - 100 // Range: -100 to 100
	}
	return data
}

// Benchmark vector addition
func BenchmarkVectorAdd_Pure_1K(b *testing.B) {
	a := generateRandomFloat32(1024)
	v := generateRandomFloat32(1024)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := range a {
			a[j] += v[j]
		}
	}
}

func BenchmarkVectorAdd_Optimized_1K(b *testing.B) {
	a := generateRandomFloat32(1024)
	v := generateRandomFloat32(1024)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		VectorAddInPlace(a, v)
	}
}

func BenchmarkVectorAdd_Pure_16K(b *testing.B) {
	a := generateRandomFloat32(16384)
	v := generateRandomFloat32(16384)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := range a {
			a[j] += v[j]
		}
	}
}

func BenchmarkVectorAdd_Optimized_16K(b *testing.B) {
	a := generateRandomFloat32(16384)
	v := generateRandomFloat32(16384)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		VectorAddInPlace(a, v)
	}
}

// Benchmark vector multiplication
func BenchmarkVectorMul_Pure_1K(b *testing.B) {
	a := generateRandomFloat32(1024)
	v := generateRandomFloat32(1024)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := range a {
			a[j] *= v[j]
		}
	}
}

func BenchmarkVectorMul_Optimized_1K(b *testing.B) {
	a := generateRandomFloat32(1024)
	v := generateRandomFloat32(1024)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		VectorMulInPlace(a, v)
	}
}

// Benchmark dot product
func BenchmarkDotProduct_Pure_1K(b *testing.B) {
	a := generateRandomFloat32(1024)
	v := generateRandomFloat32(1024)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var sum float32
		for j := range a {
			sum += a[j] * v[j]
		}
		_ = sum
	}
}

func BenchmarkDotProduct_Optimized_1K(b *testing.B) {
	a := generateRandomFloat32(1024)
	v := generateRandomFloat32(1024)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result := VectorDotOptimized(a, v)
		_ = result
	}
}

// Benchmark matrix multiplication
func BenchmarkMatMul_Pure_64x64(b *testing.B) {
	size := 64
	a := generateRandomFloat32(size * size)
	matrix := generateRandomFloat32(size * size)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result := make([]float32, size*size)
		for row := 0; row < size; row++ {
			for col := 0; col < size; col++ {
				var sum float32
				for k := 0; k < size; k++ {
					sum += a[row*size+k] * matrix[k*size+col]
				}
				result[row*size+col] = sum
			}
		}
		_ = result
	}
}

func BenchmarkMatMul_Optimized_64x64(b *testing.B) {
	size := 64
	a := generateRandomFloat32(size * size)
	matrix := generateRandomFloat32(size * size)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result := MatMulOptimized(a, size, size, matrix, size, size)
		_ = result
	}
}

func BenchmarkMatMul_Pure_128x128(b *testing.B) {
	size := 128
	a := generateRandomFloat32(size * size)
	matrix := generateRandomFloat32(size * size)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result := make([]float32, size*size)
		for row := 0; row < size; row++ {
			for col := 0; col < size; col++ {
				var sum float32
				for k := 0; k < size; k++ {
					sum += a[row*size+k] * matrix[k*size+col]
				}
				result[row*size+col] = sum
			}
		}
		_ = result
	}
}

func BenchmarkMatMul_Optimized_128x128(b *testing.B) {
	size := 128
	a := generateRandomFloat32(size * size)
	matrix := generateRandomFloat32(size * size)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result := MatMulOptimized(a, size, size, matrix, size, size)
		_ = result
	}
}

// Benchmark activation functions
func BenchmarkSigmoid_1K(b *testing.B) {
	// Create test data in byte format for existing kernel
	size := 1024
	data := make([]byte, size*4) // float32 array

	// Fill with random data
	for i := 0; i < size; i++ {
		val := rand.Float32()*20 - 10 // Range: -10 to 10
		bytes := (*[4]byte)(unsafe.Pointer(&val))[:]
		copy(data[i*4:(i+1)*4], bytes)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sigmoid(data)
	}
}

func BenchmarkTanh_1K(b *testing.B) {
	size := 1024
	data := make([]byte, size*4)

	for i := 0; i < size; i++ {
		val := rand.Float32()*20 - 10
		bytes := (*[4]byte)(unsafe.Pointer(&val))[:]
		copy(data[i*4:(i+1)*4], bytes)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tanh(data)
	}
}

func BenchmarkReLU_1K(b *testing.B) {
	size := 1024
	data := make([]byte, size*4)

	for i := 0; i < size; i++ {
		val := rand.Float32()*20 - 10
		bytes := (*[4]byte)(unsafe.Pointer(&val))[:]
		copy(data[i*4:(i+1)*4], bytes)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		relu(data)
	}
}

// Benchmark softmax
func BenchmarkSoftmax_1K(b *testing.B) {
	size := 1024
	data := make([]byte, size*4)

	for i := 0; i < size; i++ {
		val := rand.Float32()*20 - 10
		bytes := (*[4]byte)(unsafe.Pointer(&val))[:]
		copy(data[i*4:(i+1)*4], bytes)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		softmax(data)
	}
}

// Benchmark existing kernel operations for comparison
func BenchmarkExistingVectorAdd_1K(b *testing.B) {
	// Create test data in byte format for existing kernel
	size := 1024
	data := make([]byte, size*2*4) // Two vectors of float32

	// Fill with random data
	for i := 0; i < size*2; i++ {
		val := rand.Float32()*200 - 100
		bytes := (*[4]byte)(unsafe.Pointer(&val))[:]
		copy(data[i*4:(i+1)*4], bytes)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		vectorAddUnrolled(data)
	}
}

// Memory allocation benchmarks
func BenchmarkVectorAdd_Allocation(b *testing.B) {
	a := generateRandomFloat32(1024)
	v := generateRandomFloat32(1024)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result := VectorAddOptimized(a, v)
		_ = result
	}
}

func BenchmarkVectorAdd_InPlace(b *testing.B) {
	a := generateRandomFloat32(1024)
	v := generateRandomFloat32(1024)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		VectorAddInPlace(a, v)
	}
}

// Cache performance benchmarks
func BenchmarkCacheEfficiency_Sequential_1MB(b *testing.B) {
	size := 262144 // 1MB of float32s
	data := generateRandomFloat32(size)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var sum float32
		for j := range data {
			sum += data[j]
		}
		_ = sum
	}
}

func BenchmarkCacheEfficiency_Random_1MB(b *testing.B) {
	size := 262144 // 1MB of float32s
	data := generateRandomFloat32(size)
	indices := make([]int, size)
	for i := range indices {
		indices[i] = rand.Intn(size)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var sum float32
		for _, idx := range indices {
			sum += data[idx]
		}
		_ = sum
	}
}

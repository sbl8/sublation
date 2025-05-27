package kernels

import (
	"encoding/binary"
	"math"
	"testing"
	"unsafe"
)

func TestSqrPlusX(t *testing.T) {
	// Create properly encoded test data: [1.0, 2.0, 3.0, 4.0] as float32
	data := make([]byte, 16)
	binary.LittleEndian.PutUint32(data[0:4], *(*uint32)(unsafe.Pointer(&[]float32{1.0}[0])))
	binary.LittleEndian.PutUint32(data[4:8], *(*uint32)(unsafe.Pointer(&[]float32{2.0}[0])))
	binary.LittleEndian.PutUint32(data[8:12], *(*uint32)(unsafe.Pointer(&[]float32{3.0}[0])))
	binary.LittleEndian.PutUint32(data[12:16], *(*uint32)(unsafe.Pointer(&[]float32{4.0}[0])))

	sqrPlusX(data)

	// Expected: [2.0, 6.0, 12.0, 20.0]
	expected := []float32{2.0, 6.0, 12.0, 20.0}

	for i := 0; i < 4; i++ {
		result := *(*float32)(unsafe.Pointer(&data[i*4]))
		if math.Abs(float64(result-expected[i])) > 1e-6 {
			t.Errorf("Index %d: got %f, want %f", i, result, expected[i])
		}
	}
}

func TestReLU(t *testing.T) {
	// Create properly encoded test data: [-1.0, 2.0, -3.0, 4.0]
	data := make([]byte, 16)
	binary.LittleEndian.PutUint32(data[0:4], *(*uint32)(unsafe.Pointer(&[]float32{-1.0}[0])))
	binary.LittleEndian.PutUint32(data[4:8], *(*uint32)(unsafe.Pointer(&[]float32{2.0}[0])))
	binary.LittleEndian.PutUint32(data[8:12], *(*uint32)(unsafe.Pointer(&[]float32{-3.0}[0])))
	binary.LittleEndian.PutUint32(data[12:16], *(*uint32)(unsafe.Pointer(&[]float32{4.0}[0])))

	relu(data)

	// Expected: [0.0, 2.0, 0.0, 4.0]
	expected := []float32{0.0, 2.0, 0.0, 4.0}

	for i := 0; i < 4; i++ {
		result := *(*float32)(unsafe.Pointer(&data[i*4]))
		if math.Abs(float64(result-expected[i])) > 1e-6 {
			t.Errorf("Index %d: got %f, want %f", i, result, expected[i])
		}
	}
}

func TestVectorAdd(t *testing.T) {
	// Create properly encoded test data: [1.0, 2.0] + [3.0, 4.0]
	data := make([]byte, 16)
	binary.LittleEndian.PutUint32(data[0:4], *(*uint32)(unsafe.Pointer(&[]float32{1.0}[0])))
	binary.LittleEndian.PutUint32(data[4:8], *(*uint32)(unsafe.Pointer(&[]float32{2.0}[0])))
	binary.LittleEndian.PutUint32(data[8:12], *(*uint32)(unsafe.Pointer(&[]float32{3.0}[0])))
	binary.LittleEndian.PutUint32(data[12:16], *(*uint32)(unsafe.Pointer(&[]float32{4.0}[0])))

	vectorAdd(data)

	// Expected: [4.0, 6.0]
	expected := []float32{4.0, 6.0}

	for i := 0; i < 2; i++ {
		result := *(*float32)(unsafe.Pointer(&data[i*4]))
		if math.Abs(float64(result-expected[i])) > 1e-6 {
			t.Errorf("Index %d: got %f, want %f", i, result, expected[i])
		}
	}
}

func TestSoftmax(t *testing.T) {
	// Create properly encoded test data: [1.0, 2.0, 3.0]
	data := make([]byte, 12)
	binary.LittleEndian.PutUint32(data[0:4], *(*uint32)(unsafe.Pointer(&[]float32{1.0}[0])))
	binary.LittleEndian.PutUint32(data[4:8], *(*uint32)(unsafe.Pointer(&[]float32{2.0}[0])))
	binary.LittleEndian.PutUint32(data[8:12], *(*uint32)(unsafe.Pointer(&[]float32{3.0}[0])))

	softmax(data)

	// Check that values sum to 1.0
	var sum float32
	for i := 0; i < 3; i++ {
		val := *(*float32)(unsafe.Pointer(&data[i*4]))
		sum += val
		if val <= 0 {
			t.Errorf("Softmax output should be positive, got %f at index %d", val, i)
		}
	}

	if math.Abs(float64(sum-1.0)) > 1e-6 {
		t.Errorf("Softmax should sum to 1.0, got %f", sum)
	}
}

func BenchmarkSqrPlusX(b *testing.B) {
	data := make([]byte, 1024*4) // 1024 float32s

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sqrPlusX(data)
	}
}

func BenchmarkReLU(b *testing.B) {
	data := make([]byte, 1024*4) // 1024 float32s

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		relu(data)
	}
}

func BenchmarkVectorAdd(b *testing.B) {
	data := make([]byte, 1024*4) // 512 + 512 float32s

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		vectorAdd(data)
	}
}

func BenchmarkSoftmax(b *testing.B) {
	data := make([]byte, 1024*4) // 1024 float32s

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		softmax(data)
	}
}

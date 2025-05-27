package main

import (
	"flag"
	"fmt"
	"math/rand"
	"os"
	"runtime"
	"time"
	"unsafe"

	"github.com/sbl8/sublation/kernels"
)

var (
	testType = flag.String("test", "all", "Test type: all, vector, matrix, activation")
	size     = flag.Int("size", 1024, "Test data size")
	iter     = flag.Int("iter", 1000, "Number of iterations")
	verbose  = flag.Bool("verbose", false, "Verbose output")
)

func main() {
	flag.Parse()

	fmt.Printf("Sublation Performance Analysis Tool\n")
	fmt.Printf("===================================\n")
	fmt.Printf("Go Version: %s\n", runtime.Version())
	fmt.Printf("OS/Arch: %s/%s\n", runtime.GOOS, runtime.GOARCH)
	fmt.Printf("CPUs: %d\n", runtime.NumCPU())
	fmt.Printf("Test Size: %d elements\n", *size)
	fmt.Printf("Iterations: %d\n", *iter)
	fmt.Printf("Assembly Support: %t\n", kernels.UseASM())
	fmt.Printf("\n")

	switch *testType {
	case "all":
		runAllTests()
	case "vector":
		runVectorTests()
	case "matrix":
		runMatrixTests()
	case "activation":
		runActivationTests()
	default:
		fmt.Printf("Unknown test type: %s\n", *testType)
		os.Exit(1)
	}
}

func runAllTests() {
	fmt.Printf("Running comprehensive performance tests...\n\n")
	runVectorTests()
	runMatrixTests()
	runActivationTests()
}

func runVectorTests() {
	fmt.Printf("Vector Operations Performance\n")
	fmt.Printf("----------------------------\n")

	// Generate test data
	a := generateFloat32(*size)
	b := generateFloat32(*size)

	// Test vector addition
	start := time.Now()
	for i := 0; i < *iter; i++ {
		result := kernels.VectorAddOptimized(a, b)
		_ = result
	}
	vectorAddTime := time.Since(start)

	// Test in-place addition
	aCopy := make([]float32, len(a))
	copy(aCopy, a)

	start = time.Now()
	for i := 0; i < *iter; i++ {
		copy(aCopy, a) // Reset for fair comparison
		kernels.VectorAddInPlace(aCopy, b)
	}
	vectorAddInPlaceTime := time.Since(start)

	// Test vector multiplication
	start = time.Now()
	for i := 0; i < *iter; i++ {
		result := kernels.VectorMulOptimized(a, b)
		_ = result
	}
	vectorMulTime := time.Since(start)

	// Test dot product
	start = time.Now()
	for i := 0; i < *iter; i++ {
		result := kernels.VectorDotOptimized(a, b)
		_ = result
	}
	dotProductTime := time.Since(start)

	// Calculate throughput
	elementsPerSecond := func(duration time.Duration) float64 {
		return float64(*size*(*iter)) / duration.Seconds()
	}

	fmt.Printf("Vector Add (allocating):     %v (%.2f Mops/s)\n",
		vectorAddTime, elementsPerSecond(vectorAddTime)/1e6)
	fmt.Printf("Vector Add (in-place):       %v (%.2f Mops/s)\n",
		vectorAddInPlaceTime, elementsPerSecond(vectorAddInPlaceTime)/1e6)
	fmt.Printf("Vector Multiply:             %v (%.2f Mops/s)\n",
		vectorMulTime, elementsPerSecond(vectorMulTime)/1e6)
	fmt.Printf("Dot Product:                 %v (%.2f Mops/s)\n",
		dotProductTime, elementsPerSecond(dotProductTime)/1e6)

	if *verbose {
		fmt.Printf("  Memory allocation overhead: %.2fx\n",
			float64(vectorAddTime)/float64(vectorAddInPlaceTime))
	}

	fmt.Printf("\n")
}

func runMatrixTests() {
	fmt.Printf("Matrix Operations Performance\n")
	fmt.Printf("----------------------------\n")

	sizes := []int{32, 64, 128}
	if *size < 128 {
		sizes = []int{16, 32, 64}
	}

	for _, matSize := range sizes {
		a := generateFloat32(matSize * matSize)
		b := generateFloat32(matSize * matSize)

		start := time.Now()
		for i := 0; i < *iter/10; i++ { // Fewer iterations for larger matrices
			result := kernels.MatMulOptimized(a, matSize, matSize, b, matSize, matSize)
			_ = result
		}
		matMulTime := time.Since(start)

		operations := int64(matSize) * int64(matSize) * int64(matSize) * 2 * int64(*iter/10) // 2 ops per multiply-add
		gflops := float64(operations) / matMulTime.Seconds() / 1e9

		fmt.Printf("Matrix Multiply %dx%d:       %v (%.2f GFLOPS)\n",
			matSize, matSize, matMulTime, gflops)
	}

	fmt.Printf("\n")
}

func runActivationTests() {
	fmt.Printf("Activation Functions Performance\n")
	fmt.Printf("-------------------------------\n")

	// Create test data in byte format for kernel functions
	data := make([]byte, *size*4)
	for i := 0; i < *size; i++ {
		val := rand.Float32()*20 - 10 // Range: -10 to 10
		bytes := (*[4]byte)(unsafe.Pointer(&val))[:]
		copy(data[i*4:(i+1)*4], bytes)
	}

	tests := []struct {
		name string
		fn   func([]byte)
	}{
		{"ReLU", kernels.GetKernel(kernels.OpReLU)},
		{"Sigmoid", kernels.GetKernel(kernels.OpSigmoid)},
		{"Tanh", kernels.GetKernel(kernels.OpTanh)},
		{"Softmax", kernels.GetKernel(kernels.OpSoftmax)},
	}

	for _, test := range tests {
		dataCopy := make([]byte, len(data))

		start := time.Now()
		for i := 0; i < *iter; i++ {
			copy(dataCopy, data) // Reset data
			test.fn(dataCopy)
		}
		duration := time.Since(start)

		elementsPerSecond := float64(*size*(*iter)) / duration.Seconds()

		fmt.Printf("%-15s:             %v (%.2f Mops/s)\n",
			test.name, duration, elementsPerSecond/1e6)
	}

	fmt.Printf("\n")
}

func generateFloat32(size int) []float32 {
	data := make([]float32, size)
	for i := range data {
		data[i] = rand.Float32()*200 - 100 // Range: -100 to 100
	}
	return data
}

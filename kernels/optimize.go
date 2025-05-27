package kernels

import (
	"runtime"
	"unsafe"
)

// BatchSize determines optimal vectorization width based on architecture
func BatchSize() int {
	// Detect CPU capabilities and return optimal batch size
	switch runtime.GOARCH {
	case "amd64":
		return 8 // AVX2 can process 8 float32s per instruction
	case "arm64":
		return 4 // NEON can process 4 float32s per instruction
	default:
		return 4 // Conservative default
	}
}

// VectorizedKernel provides auto-vectorization for simple operations
type VectorizedKernel struct {
	scalar func(float32) float32
	batch  int
}

// NewVectorizedKernel creates a kernel that automatically batches operations
func NewVectorizedKernel(scalar func(float32) float32) *VectorizedKernel {
	return &VectorizedKernel{
		scalar: scalar,
		batch:  BatchSize(),
	}
}

// Execute runs the vectorized kernel on the data
func (vk *VectorizedKernel) Execute(data []byte) {
	const sz = 4
	count := len(data) / sz

	// Process in batches for better cache usage
	for i := 0; i < count; i += vk.batch {
		end := i + vk.batch
		if end > count {
			end = count
		}

		// Process batch
		for j := i; j < end; j++ {
			p := (*float32)(unsafe.Pointer(&data[j*sz]))
			*p = vk.scalar(*p)
		}
	}
}

// Cache-aligned memory operations for optimal performance
const CacheLineSize = 64

// AlignedCopy performs cache-aligned memory operations
func AlignedCopy(dst, src []byte) {
	// Ensure we don't exceed bounds
	if len(dst) != len(src) {
		return
	}

	// Process cache-line sized chunks
	for i := 0; i < len(src); i += CacheLineSize {
		end := i + CacheLineSize
		if end > len(src) {
			end = len(src)
		}
		copy(dst[i:end], src[i:end])
	}
}

// PrefetchData hints to the CPU to preload data into cache
func PrefetchData(data []byte) {
	// Prefetch each cache line
	for i := 0; i < len(data); i += CacheLineSize {
		if i < len(data) {
			// Simple read to trigger prefetch
			_ = data[i]
		}
	}
}

// Memory pool for temporary allocations during kernel execution
type KernelPool struct {
	buffers chan []byte
	size    int
}

// NewKernelPool creates a pool of reusable byte slices
func NewKernelPool(bufferSize, poolSize int) *KernelPool {
	kp := &KernelPool{
		buffers: make(chan []byte, poolSize),
		size:    bufferSize,
	}

	// Pre-allocate buffers
	for i := 0; i < poolSize; i++ {
		kp.buffers <- make([]byte, bufferSize)
	}

	return kp
}

// Get retrieves a buffer from the pool
func (kp *KernelPool) Get() []byte {
	select {
	case buf := <-kp.buffers:
		return buf[:kp.size] // Reset length
	default:
		// Pool empty, allocate new buffer
		return make([]byte, kp.size)
	}
}

// Put returns a buffer to the pool
func (kp *KernelPool) Put(buf []byte) {
	if cap(buf) >= kp.size {
		select {
		case kp.buffers <- buf:
		default:
			// Pool full, let GC handle it
		}
	}
}

// Global kernel pool for temporary allocations
var globalPool = NewKernelPool(4096, runtime.NumCPU()*2)

// GetTempBuffer gets a temporary buffer for kernel operations
func GetTempBuffer() []byte {
	return globalPool.Get()
}

// PutTempBuffer returns a temporary buffer
func PutTempBuffer(buf []byte) {
	globalPool.Put(buf)
}

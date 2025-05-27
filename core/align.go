package core

import "unsafe"

const (
	// CacheLineSize is a common cache line size, typically 64 bytes.
	// Adjust if targeting specific architectures with different cache line sizes.
	CacheLineSize = 64
)

// IsAligned checks if a pointer (represented as a uintptr) is aligned to a cache line boundary.
// addr is the memory address to check.
// Returns true if aligned, false otherwise.
func IsAligned(addr uintptr) bool {
	return addr%CacheLineSize == 0
}

// AlignedSize calculates the size rounded up to the nearest cache line multiple.
// size is the original size.
// Returns the aligned size.
func AlignedSize(size uintptr) uintptr {
	return (size + uintptr(CacheLineSize-1)) & ^uintptr(CacheLineSize-1)
}

// AlignedBytes allocates a byte slice with its underlying array aligned to CacheLineSize.
// size is the desired size of the slice.
// Returns the aligned byte slice.
// This is the recommended way to get an aligned slice in Go.
func AlignedBytes(size int) []byte {
	if size == 0 {
		return nil
	}
	// Allocate extra space to allow for alignment.
	// The extra space needed is at most CacheLineSize - 1.
	buf := make([]byte, size+CacheLineSize-1)

	// Get the address of the first element of the backing array.
	ptr := uintptr(unsafe.Pointer(&buf[0]))

	// Calculate the offset to the next cache line boundary.
	// If ptr is already aligned, offset will be 0.
	// Otherwise, it's CacheLineSize minus the remainder.
	offset := uintptr(0)
	if mod := ptr % CacheLineSize; mod != 0 {
		offset = CacheLineSize - mod
	}

	// Slice the buffer to the aligned portion.
	alignedBuf := buf[offset : offset+uintptr(size)]
	return alignedBuf
}

// Align32 rounds n up to the nearest 32‑byte boundary.
// This is kept for specific 32-byte alignment needs, separate from cache line alignment.
func Align32(n int) int { return (n + 31) &^ 31 }

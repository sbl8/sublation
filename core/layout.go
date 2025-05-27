package core

import "unsafe"

// Memory layout constants for optimal performance
const (
	PageSize     = 4096
	SublateAlign = 32 // SIMD-friendly alignment
)

// AlignSize rounds size up to the specified alignment boundary
func AlignSize(size, align int) int {
	return (size + align - 1) &^ (align - 1)
}

// AlignCacheLine rounds size up to cache line boundary
func AlignCacheLine(size int) int {
	return AlignSize(size, CacheLineSize) // Uses CacheLineSize from core package (implicitly, will be align.CacheLineSize or core.CacheLineSize)
}

// AlignPage rounds size up to page boundary
func AlignPage(size int) int {
	return AlignSize(size, PageSize)
}

// SublateSize calculates the exact memory footprint of a Sublate
func SublateSize(s *Sublate) int {
	return int(unsafe.Sizeof(*s)) + len(s.PayloadPrev) + len(s.PayloadProp) + len(s.Topology)*2
}

// SublateAlignedSize calculates the aligned memory footprint
func SublateAlignedSize(s *Sublate) int {
	base := SublateSize(s)
	return AlignSize(base, SublateAlign)
}

// OptimalBatchSize calculates optimal batch size for SIMD operations
func OptimalBatchSize(elementSize int) int {
	// Target: fill one cache line with elements
	elementsPerLine := CacheLineSize / elementSize // Uses CacheLineSize from core package
	if elementsPerLine < 1 {
		return 1
	}
	return elementsPerLine
}

// PadToAlignment adds padding bytes to reach alignment
func PadToAlignment(data []byte, align int) []byte {
	currentLen := len(data)
	alignedLen := AlignSize(currentLen, align)
	if alignedLen == currentLen {
		return data
	}

	padded := make([]byte, alignedLen)
	copy(padded, data)
	return padded
}

// Package core provides fundamental primitives for the Sublation dialectical AI engine.
//
// This package implements the core Sublate data structure, which is the fundamental
// compute unit in Sublation's architecture. Each Sublate contains dual cache-aligned
// buffers (PayloadPrev and PayloadProp) for lock-free computation, topology information
// for neighbor connections, and runtime flags for lineage tracking and optimization.
//
// Key components:
//   - Sublate: Cache-aligned struct with dual buffers for zero-allocation compute
//   - Memory alignment utilities for optimal SIMD performance
//   - Layout optimization functions for cache efficiency
//   - Serialization support for model persistence
//
// The design follows dialectical principles where each compute step preserves
// previous state while proposing new state, enabling both stability and change
// in the computation graph.
package core

import (
	"errors"
	"sync"
	"unsafe"
)

// Sublate is the fundamental compute block with dual buffers for cache efficiency.
// Every Sublate is cache-aligned with dual buffers as per architectural requirements.
type Sublate struct {
	PayloadPrev []byte   // previous step data (aligned to cache boundary)
	PayloadProp []byte   // propagation data (aligned to cache boundary)
	Topology    []uint16 // neighbor indices for message passing
	KernelID    uint8    // opcode for data transform
	Flags       uint32   // runtime flags including lineage tracking

	// Internal fields for memory management
	arena    []byte // backing memory arena
	offset   int    // offset within arena
	capacity int    // total allocated capacity
}

// Flags bit definitions for runtime behavior
const (
	FlagLineageTracked = 1 << 0 // Set when lineage has been updated
	FlagFused          = 1 << 1 // Set when sublate has been fused
	FlagDirty          = 1 << 2 // Set when data needs propagation
	FlagReadOnly       = 1 << 3 // Set for immutable sublates
)

// Size returns the total size of the sublate data
func (s *Sublate) Size() int {
	return len(s.PayloadPrev) + len(s.PayloadProp)
}

// Validate checks the integrity of a Sublate
func (s *Sublate) Validate() error {
	if s == nil {
		return errors.New("sublate is nil")
	}
	if len(s.PayloadPrev) == 0 && len(s.PayloadProp) == 0 {
		return errors.New("sublate payload is empty")
	}
	if len(s.PayloadPrev)%4 != 0 || len(s.PayloadProp)%4 != 0 {
		return errors.New("sublate payload not aligned to 4-byte boundary")
	}
	for _, idx := range s.Topology {
		if idx == 0xFFFF {
			return errors.New("invalid topology index")
		}
	}
	return nil
}

// AsFloat32Prev safely casts PayloadPrev to []float32 with bounds checking
func (s *Sublate) AsFloat32Prev() []float32 {
	if len(s.PayloadPrev)%4 != 0 {
		return nil
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(&s.PayloadPrev[0])), len(s.PayloadPrev)/4)
}

// AsFloat32Prop safely casts PayloadProp to []float32 with bounds checking
func (s *Sublate) AsFloat32Prop() []float32 {
	if len(s.PayloadProp)%4 != 0 {
		return nil
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(&s.PayloadProp[0])), len(s.PayloadProp)/4)
}

// AsUint32Prev safely casts PayloadPrev to []uint32 with bounds checking
func (s *Sublate) AsUint32Prev() []uint32 {
	if len(s.PayloadPrev)%4 != 0 {
		return nil
	}
	return unsafe.Slice((*uint32)(unsafe.Pointer(&s.PayloadPrev[0])), len(s.PayloadPrev)/4)
}

// AsUint32Prop safely casts PayloadProp to []uint32 with bounds checking
func (s *Sublate) AsUint32Prop() []uint32 {
	if len(s.PayloadProp)%4 != 0 {
		return nil
	}
	return unsafe.Slice((*uint32)(unsafe.Pointer(&s.PayloadProp[0])), len(s.PayloadProp)/4)
}

// SwapBuffers swaps prev and prop for double buffering
func (s *Sublate) SwapBuffers() {
	s.PayloadPrev, s.PayloadProp = s.PayloadProp, s.PayloadPrev
}

// SetFlag sets a runtime flag
func (s *Sublate) SetFlag(flag uint32) {
	s.Flags |= flag
}

// ClearFlag clears a runtime flag
func (s *Sublate) ClearFlag(flag uint32) {
	s.Flags &^= flag
}

// HasFlag checks if a runtime flag is set
func (s *Sublate) HasFlag(flag uint32) bool {
	return s.Flags&flag != 0
}

// Clone creates a deep copy of the Sublate
func (s *Sublate) Clone() *Sublate {
	clone := &Sublate{
		KernelID:    s.KernelID,
		Flags:       s.Flags,
		PayloadPrev: make([]byte, len(s.PayloadPrev)),
		PayloadProp: make([]byte, len(s.PayloadProp)),
		Topology:    make([]uint16, len(s.Topology)),
		capacity:    s.capacity,
	}
	copy(clone.PayloadPrev, s.PayloadPrev)
	copy(clone.PayloadProp, s.PayloadProp)
	copy(clone.Topology, s.Topology)
	return clone
}

// SublatePool provides zero-allocation Sublate management
type SublatePool struct {
	sublates sync.Pool
	buffers  sync.Pool
}

// NewSublatePool creates a new memory pool for Sublates
func NewSublatePool(maxDataSize int) *SublatePool {
	return &SublatePool{
		sublates: sync.Pool{
			New: func() interface{} {
				return &Sublate{}
			},
		},
		buffers: sync.Pool{
			New: func() interface{} {
				return make([]byte, 0, maxDataSize)
			},
		},
	}
}

// Get retrieves a Sublate from the pool
func (p *SublatePool) Get() *Sublate {
	return p.sublates.Get().(*Sublate)
}

// Put returns a Sublate to the pool after resetting it
func (p *SublatePool) Put(s *Sublate) {
	if s != nil {
		s.KernelID = 0
		s.Flags = 0
		s.Topology = s.Topology[:0]
		if cap(s.PayloadPrev) > 0 {
			buf := s.PayloadPrev[:0]
			p.buffers.Put(buf)
		}
		if cap(s.PayloadProp) > 0 {
			buf := s.PayloadProp[:0]
			p.buffers.Put(buf)
		}
		s.PayloadPrev = nil
		s.PayloadProp = nil
		p.sublates.Put(s)
	}
}

// GetBuffer retrieves a byte buffer from the pool
func (p *SublatePool) GetBuffer() []byte {
	return p.buffers.Get().([]byte)
}

// PutBuffer returns a byte buffer to the pool
func (p *SublatePool) PutBuffer(buf []byte) {
	if buf != nil {
		buf = buf[:0]
		p.buffers.Put(buf)
	}
}

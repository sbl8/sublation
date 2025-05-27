package core

import (
	"testing"
	"unsafe"
)

func TestSublateValidation(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name    string
		sublate *Sublate
		wantErr bool
	}{
		{
			name:    "nil sublate",
			sublate: nil,
			wantErr: true,
		},
		{
			name: "empty payload",
			sublate: &Sublate{
				PayloadPrev: []byte{},
				PayloadProp: []byte{},
				KernelID:    1,
			},
			wantErr: true,
		},
		{
			name: "unaligned payload",
			sublate: &Sublate{
				PayloadPrev: []byte{1, 2, 3}, // 3 bytes, not 4-byte aligned
				KernelID:    1,
			},
			wantErr: true,
		},
		{
			name: "valid sublate",
			sublate: &Sublate{
				PayloadPrev: []byte{1, 2, 3, 4}, // 4 bytes, aligned
				Topology:    []uint16{1, 2},
				KernelID:    1,
			},
			wantErr: false,
		},
		{
			name: "invalid topology",
			sublate: &Sublate{
				PayloadPrev: []byte{1, 2, 3, 4},
				Topology:    []uint16{0xFFFF}, // invalid index
				KernelID:    1,
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.sublate.Validate()
			if (err != nil) != tt.wantErr {
				t.Errorf("Sublate.Validate() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestSublateAsFloat32(t *testing.T) {
	t.Parallel()
	data := []byte{0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40} // 1.0, 2.0 in little-endian float32
	s := &Sublate{PayloadPrev: data}

	floats := s.AsFloat32Prev()
	if len(floats) != 2 {
		t.Errorf("Expected 2 floats, got %d", len(floats))
	}

	if floats[0] != 1.0 {
		t.Errorf("Expected first float to be 1.0, got %f", floats[0])
	}

	if floats[1] != 2.0 {
		t.Errorf("Expected second float to be 2.0, got %f", floats[1])
	}
}

func TestSublateAsFloat32Unaligned(t *testing.T) {
	t.Parallel()
	data := []byte{1, 2, 3} // Not 4-byte aligned
	s := &Sublate{PayloadPrev: data}

	floats := s.AsFloat32Prev()
	if floats != nil {
		t.Errorf("Expected nil for unaligned data, got %v", floats)
	}
}

func TestSublateClone(t *testing.T) {
	t.Parallel()
	original := &Sublate{
		PayloadPrev: []byte{1, 2, 3, 4},
		PayloadProp: []byte{5, 6, 7, 8},
		Topology:    []uint16{1, 2, 3},
		KernelID:    5,
		Flags:       FlagDirty,
	}

	clone := original.Clone()

	// Verify fields are copied
	if clone.KernelID != original.KernelID {
		t.Errorf("KernelID not copied: got %d, want %d", clone.KernelID, original.KernelID)
	}

	if clone.Flags != original.Flags {
		t.Errorf("Flags not copied: got %d, want %d", clone.Flags, original.Flags)
	}

	if len(clone.PayloadPrev) != len(original.PayloadPrev) {
		t.Errorf("PayloadPrev length mismatch: got %d, want %d", len(clone.PayloadPrev), len(original.PayloadPrev))
	}

	if len(clone.PayloadProp) != len(original.PayloadProp) {
		t.Errorf("PayloadProp length mismatch: got %d, want %d", len(clone.PayloadProp), len(original.PayloadProp))
	}

	if len(clone.Topology) != len(original.Topology) {
		t.Errorf("Topology length mismatch: got %d, want %d", len(clone.Topology), len(original.Topology))
	}

	// Verify independence (modifying clone doesn't affect original)
	clone.PayloadPrev[0] = 99
	if original.PayloadPrev[0] == 99 {
		t.Error("Clone and original share PayloadPrev slice")
	}

	clone.Topology[0] = 99
	if original.Topology[0] == 99 {
		t.Error("Clone and original share Topology slice")
	}
}

func TestSublateFlags(t *testing.T) {
	t.Parallel()
	s := &Sublate{}

	// Test setting flags
	s.SetFlag(FlagDirty)
	if !s.HasFlag(FlagDirty) {
		t.Error("Flag should be set")
	}

	s.SetFlag(FlagLineageTracked)
	if !s.HasFlag(FlagLineageTracked) {
		t.Error("Second flag should be set")
	}

	// Test clearing flags
	s.ClearFlag(FlagDirty)
	if s.HasFlag(FlagDirty) {
		t.Error("Flag should be cleared")
	}

	if !s.HasFlag(FlagLineageTracked) {
		t.Error("Other flag should still be set")
	}
}

func TestSublateSwapBuffers(t *testing.T) {
	t.Parallel()
	s := &Sublate{
		PayloadPrev: []byte{1, 2, 3, 4},
		PayloadProp: []byte{5, 6, 7, 8},
	}

	origPrev := s.PayloadPrev
	origProp := s.PayloadProp

	s.SwapBuffers()

	if &s.PayloadPrev[0] != &origProp[0] {
		t.Error("PayloadPrev should now point to original PayloadProp")
	}

	if &s.PayloadProp[0] != &origPrev[0] {
		t.Error("PayloadProp should now point to original PayloadPrev")
	}
}

func TestSublatePool(t *testing.T) {
	t.Parallel()
	pool := NewSublatePool(1024)

	// Get a sublate from pool
	s1 := pool.Get()
	if s1 == nil {
		t.Fatal("Pool should return a sublate")
	}

	// Modify the sublate
	s1.KernelID = 42
	s1.PayloadPrev = pool.GetBuffer()
	s1.PayloadPrev = append(s1.PayloadPrev, 1, 2, 3, 4)
	s1.Topology = []uint16{1, 2, 3}

	// Return to pool
	pool.Put(s1)

	// Get another sublate (should be the same instance, but reset)
	s2 := pool.Get()
	if s2.KernelID != 0 {
		t.Errorf("Sublate not reset: KernelID = %d, want 0", s2.KernelID)
	}
	if len(s2.Topology) != 0 {
		t.Errorf("Sublate not reset: Topology len = %d, want 0", len(s2.Topology))
	}
	if s2.PayloadPrev != nil {
		t.Error("Sublate not reset: PayloadPrev should be nil")
	}
}

func BenchmarkSublateValidation(b *testing.B) {
	s := &Sublate{
		PayloadPrev: make([]byte, 1024),
		Topology:    []uint16{1, 2, 3, 4, 5},
		KernelID:    1,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = s.Validate()
	}
}

func BenchmarkSublateAsFloat32(b *testing.B) {
	data := make([]byte, 1024) // 256 float32 values
	s := &Sublate{PayloadPrev: data}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = s.AsFloat32Prev()
	}
}

func BenchmarkSublateClone(b *testing.B) {
	s := &Sublate{
		PayloadPrev: make([]byte, 1024),
		PayloadProp: make([]byte, 1024),
		Topology:    []uint16{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
		KernelID:    1,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		clone := s.Clone()
		_ = clone
	}
}

func BenchmarkSublatePool(b *testing.B) {
	pool := NewSublatePool(1024)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s := pool.Get()
		s.PayloadPrev = pool.GetBuffer()
		pool.PutBuffer(s.PayloadPrev)
		pool.Put(s)
	}
}

func TestAlignment(t *testing.T) {
	t.Parallel()
	// Test that our Sublate struct is properly aligned
	s := &Sublate{}
	addr := uintptr(unsafe.Pointer(s))

	if addr%8 != 0 {
		t.Errorf("Sublate not 8-byte aligned: address = 0x%x", addr)
	}
}

func TestSublateSize(t *testing.T) {
	t.Parallel()
	s := &Sublate{
		PayloadPrev: make([]byte, 100),
		PayloadProp: make([]byte, 200),
	}

	expectedSize := 300
	actualSize := s.Size()

	if actualSize != expectedSize {
		t.Errorf("Size() = %d, want %d", actualSize, expectedSize)
	}
}

func TestSublateAsUint32(t *testing.T) {
	t.Parallel()
	data := make([]byte, 8) // 2 uint32 values
	// Write values in little-endian
	data[0] = 0x01
	data[1] = 0x02
	data[2] = 0x03
	data[3] = 0x04
	data[4] = 0x05
	data[5] = 0x06
	data[6] = 0x07
	data[7] = 0x08

	s := &Sublate{PayloadPrev: data}

	uints := s.AsUint32Prev()
	if len(uints) != 2 {
		t.Errorf("Expected 2 uint32 values, got %d", len(uints))
	}

	// Test unaligned case
	s2 := &Sublate{PayloadPrev: []byte{1, 2, 3}}
	uints2 := s2.AsUint32Prev()
	if uints2 != nil {
		t.Error("Expected nil for unaligned data")
	}
}

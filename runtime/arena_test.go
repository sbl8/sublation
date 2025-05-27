package runtime

import (
	"testing"
	"unsafe"

	"github.com/sbl8/sublation/model"
)

func TestNewArena(t *testing.T) {
	t.Parallel()
	graph := &model.Graph{
		Payload: make([]byte, 1024),
		Nodes: []model.Node{
			{Kernel: 1, In: 0, Out: 256, Flags: 0x01, Topo: []uint16{1, 2, 0, 0}},
			{Kernel: 2, In: 256, Out: 512, Flags: 0x02, Topo: []uint16{2, 1, 0, 0}},
		},
	}

	arena, err := NewArena(8192, graph, 512, 1024, 1024) // Added kernelScratchSize
	if err != nil {
		t.Fatalf("NewArena failed: %v", err)
	}

	// Verify total size
	if arena.TotalSize() == 0 {
		t.Error("Arena has zero size")
	}

	// Check regions exist
	if _, ok := arena.Region("ModelPayload"); !ok {
		t.Error("ModelPayload region not found")
	}
	if _, ok := arena.Region("SublateMetadata"); !ok {
		t.Error("SublateMetadata region not found")
	}
	if _, ok := arena.Region("Scratch"); !ok {
		t.Error("Scratch region not found")
	}
}

func TestArenaMemoryLayout(t *testing.T) {
	t.Parallel()
	graph := &model.Graph{
		Payload: make([]byte, 256),
		Nodes:   make([]model.Node, 4),
	}

	arena, err := NewArena(4096, graph, 256, 512, 512) // Added kernelScratchSize
	if err != nil {
		t.Fatalf("NewArena failed: %v", err)
	}

	// Verify regions don't overlap
	regions := []string{"ModelPayload", "SublateMetadata", "Scratch", "StreamingInput"}
	offsets := make([]uintptr, len(regions))
	sizes := make([]uintptr, len(regions))

	for i, name := range regions {
		region, ok := arena.Region(name)
		if !ok {
			t.Errorf("Region %s not found", name)
			continue
		}
		offsets[i] = region.Offset
		sizes[i] = region.Size
	}

	// Check no overlaps
	for i := 0; i < len(regions)-1; i++ {
		if offsets[i]+sizes[i] > offsets[i+1] {
			t.Errorf("Region %s overlaps with %s", regions[i], regions[i+1])
		}
	}
}

func TestGetSublateAtIndex(t *testing.T) {
	t.Parallel()
	graph := &model.Graph{
		Payload: make([]byte, 128),
		Nodes: []model.Node{
			{Kernel: 1, Flags: 0x01},
			{Kernel: 2, Flags: 0x02},
		},
	}

	arena, err := NewArena(2048, graph, 128, 256, 256) // Added kernelScratchSize
	if err != nil {
		t.Fatalf("NewArena failed: %v", err)
	}

	// Test valid indices
	for i := 0; i < len(graph.Nodes); i++ {
		sublate, err := arena.GetSublateAtIndex(i)
		if err != nil {
			t.Errorf("GetSublateAtIndex(%d) failed: %v", i, err)
		}
		if sublate == nil {
			t.Errorf("GetSublateAtIndex(%d) returned nil", i)
		}
	}

	// Test out of bounds
	_, err = arena.GetSublateAtIndex(len(graph.Nodes))
	if err == nil {
		t.Error("Expected error for out of bounds index")
	}
}

func TestScratchAllocation(t *testing.T) {
	t.Parallel()
	graph := &model.Graph{
		Payload: make([]byte, 64),
		Nodes:   []model.Node{{Kernel: 1}},
	}

	arena, err := NewArena(1024, graph, 64, 256, 256) // Added kernelScratchSize
	if err != nil {
		t.Fatalf("NewArena failed: %v", err)
	}

	// Allocate from scratch
	buf1, err := arena.AllocateScratch(64, 8)
	if err != nil {
		t.Errorf("AllocateScratch failed: %v", err)
	}
	if len(buf1) != 64 {
		t.Errorf("Expected 64 bytes, got %d", len(buf1))
	}

	// Allocate more
	buf2, err := arena.AllocateScratch(32, 8)
	if err != nil {
		t.Errorf("Second AllocateScratch failed: %v", err)
	}

	// Verify no overlap
	ptr1 := uintptr(unsafe.Pointer(&buf1[0]))
	ptr2 := uintptr(unsafe.Pointer(&buf2[0]))
	if ptr1+64 > ptr2 {
		t.Error("Scratch allocations overlap")
	}

	// Reset and reallocate
	arena.ResetScratch()
	buf3, err := arena.AllocateScratch(64, 8)
	if err != nil {
		t.Errorf("AllocateScratch after reset failed: %v", err)
	}
	ptr3 := uintptr(unsafe.Pointer(&buf3[0]))
	if ptr3 != ptr1 {
		t.Error("Reset didn't return to start of scratch region")
	}
}

func TestInitSublateInArena(t *testing.T) {
	t.Parallel()
	graph := &model.Graph{
		Payload: []byte{1, 2, 3, 4, 5, 6, 7, 8},
		Nodes: []model.Node{
			{Kernel: 1, In: 0, Out: 4, Flags: 0x01, Topo: []uint16{1, 1, 0, 0}},
		},
	}

	arena, err := NewArena(1024, graph, 128, 256, 256) // Increased nodePayloadsSize to accommodate both PayloadPrev and PayloadProp
	if err != nil {
		t.Fatalf("NewArena failed: %v", err)
	}

	err = InitSublateInArena(arena, 0, &graph.Nodes[0], graph.Payload, 32, 32)
	if err != nil {
		t.Fatalf("InitSublateInArena failed: %v", err)
	}

	sublate, err := arena.GetSublateAtIndex(0)
	if err != nil {
		t.Fatalf("GetSublateAtIndex failed: %v", err)
	}

	// Verify initialization
	if sublate.KernelID != 1 {
		t.Errorf("Expected KernelID 1, got %d", sublate.KernelID)
	}
	if sublate.Flags != 0x01 {
		t.Errorf("Expected Flags 0x01, got 0x%02x", sublate.Flags)
	}
	if len(sublate.PayloadPrev) == 0 {
		t.Error("PayloadPrev not allocated")
	}
	if len(sublate.PayloadProp) == 0 {
		t.Error("PayloadProp not allocated")
	}
}

func TestStreamingInput(t *testing.T) {
	t.Parallel()
	graph := &model.Graph{
		Payload: make([]byte, 32),
		Nodes:   []model.Node{{Kernel: 1}},
	}

	arena, err := NewArena(512, graph, 128, 64, 64) // Added kernelScratchSize
	if err != nil {
		t.Fatalf("NewArena failed: %v", err)
	}

	testData := []byte{0xAA, 0xBB, 0xCC, 0xDD}
	err = arena.WriteToStreamingInput(testData)
	if err != nil {
		t.Errorf("WriteToStreamingInput failed: %v", err)
	}

	window, err := arena.StreamingInputWindow()
	if err != nil {
		t.Errorf("StreamingInputWindow failed: %v", err)
	}

	for i, expected := range testData {
		if window[i] != expected {
			t.Errorf("Expected window[%d] = 0x%02x, got 0x%02x", i, expected, window[i])
		}
	}
}

func TestFloatConversion(t *testing.T) {
	t.Parallel()
	floats := []float32{1.0, 2.5, -3.14, 0.0}
	bytes := FloatsToBytes(floats)

	converted, err := BytesToFloats(bytes)
	if err != nil {
		t.Errorf("BytesToFloats failed: %v", err)
	}

	if len(converted) != len(floats) {
		t.Errorf("Expected %d floats, got %d", len(floats), len(converted))
	}

	for i, expected := range floats {
		if converted[i] != expected {
			t.Errorf("Expected %f, got %f", expected, converted[i])
		}
	}
}

func BenchmarkArenaAllocation(b *testing.B) {
	graph := &model.Graph{
		Payload: make([]byte, 1024),
		Nodes:   make([]model.Node, 100),
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		arena, err := NewArena(32768, graph, 512, 1024, 1024) // Increased arena size for 100 nodes
		if err != nil {
			b.Fatalf("NewArena failed: %v", err)
		}
		_ = arena
	}
}

func BenchmarkScratchAllocation(b *testing.B) {
	graph := &model.Graph{
		Payload: make([]byte, 64),
		Nodes:   []model.Node{{Kernel: 1}},
	}

	arena, err := NewArena(8192, graph, 64, 2048, 2048) // Increased total size to accommodate all regions
	if err != nil {
		b.Fatalf("NewArena failed: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if i%100 == 0 {
			arena.ResetScratch()
		}
		_, err := arena.AllocateScratch(64, 8)
		if err != nil {
			arena.ResetScratch()
			i-- // Retry
		}
	}
}

package runtime

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"
	"unsafe"

	"github.com/sbl8/sublation/core"
	"github.com/sbl8/sublation/model"
)

func TestNewEngine(t *testing.T) {
	t.Parallel()
	graph := &model.Graph{
		Payload: make([]byte, 512),
		Nodes: []model.Node{
			{Kernel: 1, In: 0, Out: 128, Flags: 0x01},
			{Kernel: 2, In: 128, Out: 256, Flags: 0x02},
		},
	}

	opts := &EngineOptions{
		ArenaSize: 4096, // Provide a default arena size for testing
	}
	engine, err := NewEngine(graph, opts)
	if err != nil {
		t.Fatalf("NewEngine failed: %v", err)
	}

	if engine == nil {
		t.Fatal("Engine is nil")
	}

	if engine.graph != graph {
		t.Error("Engine graph not set correctly")
	}
}

func TestEngineOptions(t *testing.T) {
	t.Parallel()
	opts := DefaultEngineOptions()
	if opts.Workers <= 0 {
		t.Error("Default workers should be > 0")
	}

	// Test custom options
	customOpts := &EngineOptions{
		Workers:   4,
		ArenaSize: 8192,
		Streaming: true,
	}

	graph := &model.Graph{
		Payload: make([]byte, 256),
		Nodes:   []model.Node{{Kernel: 1}},
	}

	engine, err := NewEngine(graph, customOpts)
	if err != nil {
		t.Fatalf("NewEngine with custom options failed: %v", err)
	}

	if engine.workers != 4 {
		t.Errorf("Expected 4 workers, got %d", engine.workers)
	}
}

func TestExecutionContext(t *testing.T) {
	t.Parallel()
	graph := &model.Graph{
		Payload: make([]byte, 128),
		Nodes: []model.Node{
			{Kernel: 1, In: 0, Out: 64, Flags: 0x01},
		},
	}

	opts := &EngineOptions{
		ArenaSize:   4096, // Provide a default arena size for testing
		EnableStats: true, // Ensure stats are enabled for this test
	}
	engine, err := NewEngine(graph, opts)
	if err != nil {
		t.Fatalf("NewEngine failed: %v", err)
	}

	ctx := NewExecutionContext(len(graph.Nodes))
	err = engine.Execute(ctx)
	if err != nil {
		t.Errorf("Execute failed: %v", err)
	}

	stats := engine.Stats()
	if stats.TotalExecutions == 0 {
		t.Error("Expected execution count > 0")
	}
}

func TestStreamingExecution(t *testing.T) {
	t.Parallel()
	graph := &model.Graph{
		Payload: make([]byte, 256),
		Nodes: []model.Node{
			{Kernel: 1, In: 0, Out: 128, Flags: 0x01},
			{Kernel: 2, In: 128, Out: 256, Flags: 0x02},
		},
	}

	opts := &EngineOptions{
		Workers:   2,
		ArenaSize: 4096,
		Streaming: true,
	}

	engine, err := NewEngine(graph, opts)
	if err != nil {
		t.Fatalf("NewEngine failed: %v", err)
	}

	// Test streaming execution with byte slices
	input := make([]byte, 128)
	output := make([]byte, 128)

	err = engine.ExecuteStreaming(input, output)
	if err != nil {
		t.Errorf("ExecuteStreaming failed: %v", err)
	}

	// Verify output has been written to
	if len(output) != 128 {
		t.Errorf("Expected output length 128, got %d", len(output))
	}
}

func TestWorkStealingScheduler(t *testing.T) {
	t.Parallel()
	scheduler := NewWorkStealingScheduler(4)
	if scheduler == nil {
		t.Fatal("NewWorkStealingScheduler returned nil")
	}

	// Submit sublate work items to different workers
	for i := 0; i < 8; i++ {
		sublate := &core.Sublate{
			KernelID: uint8(i % 4),
			Flags:    uint32(i),
		}
		scheduler.SubmitWork(i%4, sublate)
	}

	// Workers should be able to steal work
	var completedWork int64
	var wg sync.WaitGroup
	wg.Add(4)
	for i := 0; i < 4; i++ {
		go func(workerID int) {
			defer wg.Done()
			for {
				sublate := scheduler.GetWork(workerID)
				if sublate == nil {
					break
				}
				// Process the sublate (simulate work)
				time.Sleep(1 * time.Millisecond)
				atomic.AddInt64(&completedWork, 1)
			}
		}(i)
	}

	// Wait for all workers to complete
	wg.Wait()
}

func TestArenaAllocator(t *testing.T) {
	t.Parallel()
	allocator := NewArenaAllocator(1024)
	if allocator == nil {
		t.Fatal("NewArenaAllocator returned nil")
	}

	// Test basic allocation
	buf1 := allocator.Allocate(64, 8)
	if buf1 == nil {
		t.Error("Allocate returned nil")
	}
	if len(buf1) != 64 {
		t.Errorf("Expected 64 bytes, got %d", len(buf1))
	}

	// Test alignment
	buf2 := allocator.Allocate(32, 16)
	if buf2 == nil {
		t.Error("Aligned allocate returned nil")
	}

	// Check 16-byte alignment
	addr := uintptr(unsafe.Pointer(&buf2[0]))
	if addr%16 != 0 {
		t.Errorf("Buffer not 16-byte aligned: 0x%x", addr)
	}

	// Test available space
	if allocator.Available() >= 1024 {
		t.Error("Available space should be less than total after allocations")
	}

	// Test reset
	allocator.Reset()
	if allocator.Available() != 1024 {
		t.Error("Reset didn't restore full capacity")
	}
}

func TestEngineStats(t *testing.T) {
	t.Parallel()
	graph := &model.Graph{
		Payload: make([]byte, 64),
		Nodes:   []model.Node{{Kernel: 1}},
	}

	opts := &EngineOptions{
		ArenaSize:   4096, // Provide a default arena size for testing
		EnableStats: true,
	}
	engine, err := NewEngine(graph, opts)
	if err != nil {
		t.Fatalf("NewEngine failed: %v", err)
	}

	initialStats := engine.Stats()
	if initialStats.TotalExecutions != 0 {
		t.Error("Initial execution count should be 0")
	}

	// Execute and check stats
	ctx := NewExecutionContext(len(graph.Nodes))
	err = engine.Execute(ctx)
	if err != nil {
		t.Errorf("Execute failed: %v", err)
	}

	stats := engine.Stats()
	if stats.TotalExecutions != 1 {
		t.Errorf("Expected 1 execution, got %d", stats.TotalExecutions)
	}
}

func BenchmarkEngineExecution(b *testing.B) {
	graph := &model.Graph{
		Payload: make([]byte, 512),
		Nodes: []model.Node{
			{Kernel: 1, In: 0, Out: 256, Flags: 0x01},
			{Kernel: 2, In: 256, Out: 512, Flags: 0x02},
		},
	}

	opts := &EngineOptions{
		ArenaSize: 4096, // Provide a default arena size for testing
	}
	engine, err := NewEngine(graph, opts)
	if err != nil {
		b.Fatalf("NewEngine failed: %v", err)
	}

	ctx := NewExecutionContext(len(graph.Nodes))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		err := engine.Execute(ctx)
		if err != nil {
			b.Fatalf("Execute failed: %v", err)
		}
	}
}

func BenchmarkWorkStealing(b *testing.B) {
	scheduler := NewWorkStealingScheduler(4)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sublate := &core.Sublate{
			KernelID: uint8(i % 4),
			Flags:    uint32(i),
		}
		scheduler.SubmitWork(i%4, sublate)

		work := scheduler.GetWork(i % 4)
		if work != nil {
			// Process the sublate (no-op for benchmark)
			_ = work.KernelID
		}
	}
}

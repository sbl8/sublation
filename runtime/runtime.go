// Package runtime implements the Sublation execution engine and memory management.
//
// This package provides the core runtime system that executes compiled Sublation
// models (.subl files) using streaming dataflow scheduling. The runtime manages
// a pre-allocated memory arena, coordinates worker goroutines, and handles
// dependency resolution for optimal parallelism.
//
// Key components:
//   - Engine: Main execution coordinator with immutable model graph
//   - Arena: Zero-allocation memory management with cache-aligned regions
//   - StreamScheduler: Dependency-aware task scheduling with work stealing
//   - ExecutionContext: Per-execution state tracking and metrics
//
// The runtime follows a strict zero-allocation policy during execution - all
// memory is pre-planned at startup, and computation proceeds through lock-free
// buffer swapping and in-place kernel operations.
//
// Execution model:
//  1. Load compiled model into arena memory
//  2. Build dependency graph for scheduling
//  3. Stream input data through computation pipeline
//  4. Coordinate parallel kernel execution across worker goroutines
//  5. Collect results and execution statistics
package runtime

import (
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"runtime"
	"sync"
	"time"
	"unsafe"

	"github.com/sbl8/sublation/core"
	"github.com/sbl8/sublation/kernels"
	"github.com/sbl8/sublation/model"
)

// KernelFn operates in‑place on a Sublate payload with zero allocations
type KernelFn func(data []byte)

// Basic kernel catalog - will be populated with actual kernels
var kernelCatalog = [256]KernelFn{
	// Initialize with noop kernels to avoid nil panics
}

func init() {
	// Simple noop kernel as fallback
	noop := func(data []byte) {
		// Do nothing - placeholder kernel
	}

	// Initialize all kernels with noop to avoid nil pointer issues
	for i := range kernelCatalog {
		kernelCatalog[i] = noop
	}
}

// NewArenaCompat creates a new Arena using the arena.go constructor for backward compatibility
func NewArenaCompat(totalSize int) *Arena {
	arena, err := NewArena(uintptr(totalSize), nil, 0, uintptr(totalSize/4), uintptr(totalSize/4)) // Added kernelScratchSize
	if err != nil {
		// Fallback to minimal arena
		arena, _ = NewArena(uintptr(totalSize), nil, 0, 0, 0) // Added kernelScratchSize
	}
	return arena
}

// TaskGroup represents a group of nodes that can execute concurrently
type TaskGroup struct {
	nodes    []model.Node
	priority int
}

// StreamScheduler manages dependency-aware execution of graph nodes
type StreamScheduler struct {
	ready     chan *TaskGroup
	completed chan uint16
	deps      map[uint16][]uint16
	waiting   map[uint16]*TaskGroup
	workers   int
}

// Engine manages the execution of a Sublation graph with worker pools and arena management
type Engine struct {
	graph     *model.Graph
	arena     *Arena
	workers   int
	sublates  []*core.Sublate
	scheduler *StreamScheduler
	opts      EngineOptions
	stats     ExecutionStats
	mu        sync.RWMutex
}

// Graph returns the engine's underlying graph.
func (e *Engine) Graph() *model.Graph {
	return e.graph
}

// EngineOptions configures engine behavior
type EngineOptions struct {
	Workers     int
	ArenaSize   uintptr
	EnableStats bool
	Streaming   bool
}

// ExecutionStats tracks runtime performance metrics
type ExecutionStats struct {
	TotalExecutions  int64
	AverageLatency   time.Duration
	KernelExecutions map[uint8]int64
	ArenaUtilization float64
}

// DefaultEngineOptions provides sensible runtime defaults
func DefaultEngineOptions() EngineOptions {
	return EngineOptions{
		Workers:     runtime.NumCPU(),
		ArenaSize:   0, // Auto-calculate
		EnableStats: false,
		Streaming:   true,
	}
}

// NewStreamScheduler creates a scheduler with dependency analysis
func NewStreamScheduler(graph *model.Graph, workers int) *StreamScheduler {
	s := &StreamScheduler{
		ready:     make(chan *TaskGroup, len(graph.Nodes)), // Buffered channel
		completed: make(chan uint16, len(graph.Nodes)),     // Buffered channel
		deps:      make(map[uint16][]uint16),
		waiting:   make(map[uint16]*TaskGroup),
		workers:   workers,
	}
	s.buildDependencies(graph)
	s.createTaskGroups(graph) // This will populate s.waiting
	return s
}

// buildDependencies analyzes the graph to build execution dependencies
func (s *StreamScheduler) buildDependencies(graph *model.Graph) {
	// Build dependency mapping from topology connections
	// For each node, its dependencies are the nodes listed in its Topo field.
	// The deps map should store: nodeID -> list of nodes that depend on nodeID.
	// However, the current scheduler logic seems to expect deps[nodeID] to be
	// the list of nodes that *nodeID depends on*. Let's clarify.
	// Based on scheduleReady: `for _, depID := range e.scheduler.deps[node.ID]`
	// This means deps[node.ID] are the nodes that node.ID depends on (its inputs).

	// Initialize deps for all nodes
	for _, node := range graph.Nodes {
		s.deps[node.ID] = []uint16{} // Initialize with empty slice
	}

	// Populate dependencies. A node depends on the nodes in its `Topo` field if `Topo` represents inputs.
	// Or, if `Topo` represents outputs, then other nodes depend on it.
	// Assuming `node.Topo` are the IDs of nodes that *this node sends output to* (i.e., nodes that depend on this node).
	// The `graph.topologicalSort` and `model.Node.Topo` usage suggests `Topo` might be outgoing connections.
	// Let's assume `node.Topo` are nodes that *depend* on the current node.
	// The scheduler logic `e.scheduler.deps[node.ID]` expects a list of prerequisites for `node.ID`.

	// Let's re-evaluate: `model.Node.Topo` are "neighbor indices for message passing".
	// If a node A has node B in its Topo, does A depend on B, or B depend on A?
	// The `graph.topologicalSort` uses `inDegree` and `adj` where `adj[dep]` gets `node.ID`.
	// This implies `dep` is a prerequisite for `node.ID`. So, `node.Topo` might be outgoing connections.
	// For now, let's assume `node.Topo` are nodes that *depend* on the current node.

	// Correct interpretation: `node.Topo` lists the IDs of nodes that are *inputs* to the current node.
	// So, `s.deps[node.ID]` should be `node.Topo`.
	for _, node := range graph.Nodes {
		// Ensure node.ID exists as a key
		if _, ok := s.deps[node.ID]; !ok {
			s.deps[node.ID] = []uint16{}
		}
		// Add dependencies from node.Topo
		// Defensive copy, though node.Topo should be correct from graph construction.
		s.deps[node.ID] = append(s.deps[node.ID], node.Topo...)
	}
}

// createTaskGroups organizes nodes into concurrent execution groups
// This populates the s.waiting map.
func (s *StreamScheduler) createTaskGroups(graph *model.Graph) {
	// Group nodes by dependency level for parallel execution.
	// Nodes with no dependencies are level 0.
	// Nodes that depend only on level 0 nodes are level 1, and so on.

	// This is a simplified way to create initial task groups.
	// A more sophisticated approach might consider actual parallelism and priorities.
	// For now, let's group by "readiness" based on dependencies.
	// We can iterate and find all nodes whose dependencies are met.

	// For simplicity, we can initially put all nodes into a single "level"
	// and let `scheduleReady` figure out which ones can run.
	// Or, group them by some heuristic.
	// Let's try to group them by the number of dependencies as a proxy for levels.

	levels := make(map[int][]*model.Node)
	maxLevel := 0

	nodeMap := make(map[uint16]model.Node)
	for _, n := range graph.Nodes {
		nodeMap[n.ID] = n
	}

	// Calculate levels (this is a simple depth-first search style level assignment)
	// This is not a perfect level calculation for parallel execution groups but a starting point.
	// A true level-based approach (like Coffman-Graham or similar) is more complex.
	// For now, we'll use a placeholder: group all nodes into one task group at level 0.
	// `scheduleReady` will then pick those that are actually ready.

	// A more direct approach for `s.waiting`:
	// Create task groups where each group corresponds to a "level" of execution.
	// Level 0: nodes with no dependencies.
	// Level 1: nodes whose dependencies are all in Level 0.
	// ... and so on.

	nodeLevels := make(map[uint16]int)
	visited := make(map[uint16]bool)

	var calculateLevel func(nodeID uint16) int
	calculateLevel = func(nodeID uint16) int {
		if level, ok := nodeLevels[nodeID]; ok {
			return level
		}
		if visited[nodeID] { // Cycle detection or already processed
			return 0 // Or handle error
		}
		visited[nodeID] = true

		maxDepLevel := -1
		for _, depID := range s.deps[nodeID] {
			depLevel := calculateLevel(depID)
			if depLevel > maxDepLevel {
				maxDepLevel = depLevel
			}
		}

		currentLevel := maxDepLevel + 1
		nodeLevels[nodeID] = currentLevel
		delete(visited, nodeID) // Allow re-calculation if part of different paths in complex graphs
		return currentLevel
	}

	for _, node := range graph.Nodes {
		level := calculateLevel(node.ID)
		if _, ok := levels[level]; !ok {
			levels[level] = []*model.Node{}
		}
		// Store pointer to node from graph.Nodes to avoid copying large structs
		// Find the original node pointer
		var originalNode model.Node
		for _, n := range graph.Nodes {
			if n.ID == node.ID {
				originalNode = n
				break
			}
		}
		levels[level] = append(levels[level], &originalNode)
		if level > maxLevel {
			maxLevel = level
		}
	}

	for i := 0; i <= maxLevel; i++ {
		if nodesInLevel, ok := levels[i]; ok && len(nodesInLevel) > 0 {
			// Convert []*model.Node to []model.Node for TaskGroup
			taskGroupNodes := make([]model.Node, len(nodesInLevel))
			for j, nodePtr := range nodesInLevel {
				taskGroupNodes[j] = *nodePtr
			}
			s.waiting[uint16(i)] = &TaskGroup{nodes: taskGroupNodes, priority: i}
		}
	}
}

// NewEngine creates a new runtime engine with optimal configuration
func NewEngine(graph *model.Graph, opts *EngineOptions) (*Engine, error) {
	if graph == nil {
		return nil, errors.New("graph cannot be nil")
	}

	engine, err := createBaseEngine(graph, opts)
	if err != nil {
		return nil, err
	}

	if err := setupEngineArena(engine); err != nil {
		return nil, err
	}

	if err := initializeEngineComponents(engine); err != nil {
		return nil, err
	}

	return engine, nil
}

// createBaseEngine creates the basic engine structure
func createBaseEngine(graph *model.Graph, opts *EngineOptions) (*Engine, error) {
	engineOpts := DefaultEngineOptions()
	if opts != nil {
		engineOpts = *opts
		if opts.Workers <= 0 {
			engineOpts.Workers = DefaultEngineOptions().Workers
		}
	}

	arenaSize := engineOpts.ArenaSize
	if arenaSize == 0 {
		arenaSize = calculateArenaSize(graph)
		if arenaSize == 0 && len(graph.Nodes) > 0 {
			return nil, errors.New("calculated arena size is zero for a non-empty graph")
		}
	}
	engineOpts.ArenaSize = arenaSize

	return &Engine{
		graph:    graph,
		workers:  engineOpts.Workers,
		opts:     engineOpts,
		stats:    ExecutionStats{KernelExecutions: make(map[uint8]int64)},
		sublates: make([]*core.Sublate, len(graph.Nodes)),
	}, nil
}

// setupEngineArena creates and configures the engine's arena
func setupEngineArena(engine *Engine) error {
	arenaSize := engine.opts.ArenaSize
	if arenaSize == 0 {
		engine.arena = nil
		return nil
	}

	arenaSizes, err := calculateArenaSizes(arenaSize, engine.opts.Streaming, engine.graph)
	if err != nil {
		return err
	}

	arena, err := createArenaWithFallback(arenaSize, engine.graph, arenaSizes)
	if err != nil {
		return fmt.Errorf("failed to create arena: %w", err)
	}

	engine.arena = arena
	return nil
}

// calculateArenaSizes computes scratch, streaming, and node payloads sizes
func calculateArenaSizes(totalSize uintptr, streaming bool, graph *model.Graph) (struct{ scratch, streaming, nodePayloads uintptr }, error) {
	var sizes struct{ scratch, streaming, nodePayloads uintptr }

	// Calculate node payloads size based on graph nodes
	if len(graph.Nodes) > 0 {
		totalNodeDataSize := uintptr(0)
		for i := range graph.Nodes {
			node := graph.Nodes[i]
			nodePayload := uintptr(calculateNodePayloadSize(&node, graph))
			totalNodeDataSize += nodePayload * 2 // For Prev and Prop
		}
		sizes.nodePayloads = core.AlignedSize(totalNodeDataSize)
	}

	// Calculate remaining size after node payloads
	var remainingSize uintptr
	if sizes.nodePayloads < totalSize {
		remainingSize = totalSize - sizes.nodePayloads
	} else {
		// If node payloads exceed total size, use minimal allocation
		sizes.nodePayloads = totalSize / 2
		remainingSize = totalSize - sizes.nodePayloads
	}

	if streaming {
		sizes.streaming = remainingSize / 4 // 25% of remaining for streaming
		sizes.scratch = remainingSize / 4   // 25% of remaining for scratch
	} else {
		sizes.streaming = 0
		sizes.scratch = remainingSize / 2 // 50% of remaining for scratch
	}

	// Validate total doesn't exceed arena size
	total := sizes.nodePayloads + sizes.streaming + sizes.scratch
	if total > totalSize {
		// Scale down proportionally
		scale := float64(totalSize) / float64(total)
		sizes.nodePayloads = uintptr(float64(sizes.nodePayloads) * scale)
		sizes.streaming = uintptr(float64(sizes.streaming) * scale)
		sizes.scratch = uintptr(float64(sizes.scratch) * scale)
	}

	return sizes, nil
}

// createArenaWithFallback attempts arena creation with fallback
func createArenaWithFallback(totalSize uintptr, graph *model.Graph, sizes struct{ scratch, streaming, nodePayloads uintptr }) (*Arena, error) {
	arena, err := NewArena(totalSize, graph, sizes.nodePayloads, sizes.streaming, sizes.scratch)
	if err != nil {
		// Fallback with minimal scratch/streaming
		arena, err = NewArena(totalSize, graph, 0, 0, 0)
		if err != nil {
			return nil, err
		}
	}
	return arena, nil
}

// initializeEngineComponents sets up sublates and scheduler
func initializeEngineComponents(engine *Engine) error {
	if err := initializeSublatesIfNeeded(engine); err != nil {
		return err
	}

	if err := initializeSchedulerIfNeeded(engine); err != nil {
		return err
	}

	return nil
}

// initializeSublatesIfNeeded initializes sublates in arena if required
func initializeSublatesIfNeeded(engine *Engine) error {
	if engine.arena != nil && len(engine.graph.Nodes) > 0 {
		if err := engine.initializeSublates(engine.graph, engine.arena); err != nil {
			return fmt.Errorf("failed to initialize sublates: %w", err)
		}
	} else if len(engine.graph.Nodes) > 0 && engine.arena == nil {
		return errors.New("cannot initialize sublates without an arena for a non-empty graph")
	}
	return nil
}

// initializeSchedulerIfNeeded sets up scheduler for streaming mode
func initializeSchedulerIfNeeded(engine *Engine) error {
	if engine.opts.Streaming && engine.workers > 0 {
		engine.scheduler = NewStreamScheduler(engine.graph, engine.workers)
	}
	return nil
}

// calculateArenaSize estimates required arena size based on graph
func calculateArenaSize(graph *model.Graph) uintptr {
	// Base size: graph payload
	size := uintptr(len(graph.Payload))

	// Add space for sublate structs (metadata)
	if len(graph.Nodes) > 0 {
		sublateStructSize := unsafe.Sizeof(core.Sublate{})
		alignedSublateStructSize := core.AlignedSize(sublateStructSize)
		size += uintptr(len(graph.Nodes)) * alignedSublateStructSize
	}

	// Add space for actual sublate payloads (Prev and Prop data)
	// This requires iterating nodes and summing their estimated payload sizes.
	totalNodeDataSize := uintptr(0)
	for i := range graph.Nodes {
		// Use a temporary node variable to pass to calculateNodePayloadSize if it expects a pointer
		node := graph.Nodes[i]
		// Each sublate has PayloadPrev and PayloadProp.
		// calculateNodePayloadSize should return the size for *one* such buffer.
		nodePayload := uintptr(calculateNodePayloadSize(&node, graph))
		totalNodeDataSize += nodePayload * 2 // For Prev and Prop
	}
	size += core.AlignedSize(totalNodeDataSize)

	// Add scratch space (e.g., 25% of the sum of graph payload, sublate metadata, and sublate data)
	// This is a heuristic and might need refinement.
	if size > 0 { // Avoid division by zero or negative scratch if size is 0
		scratch := size / 4
		size += scratch
	} else if len(graph.Nodes) > 0 { // If graph has nodes but calculated size is 0, add some default scratch
		size += 1024 // Default minimum scratch for non-empty graphs with no other size indicators
	}

	// Add streaming buffer if enabled (e.g., 12.5% of current total)
	// This should ideally be based on engine options if available here, or a graph property.
	// For now, let's assume it's a general overhead if streaming might be used.
	// if engineOpts.Streaming { // engineOpts not available here, assume general provision
	//  streamingBuf := size / 8
	// 	size += streamingBuf
	// }

	// Ensure a minimum arena size if there are nodes but calculated size is too small
	if len(graph.Nodes) > 0 && size < 1024 { // Arbitrary minimum
		size = 1024
	}
	if len(graph.Nodes) == 0 && len(graph.Payload) == 0 && size < 256 { // Min for empty graph
		size = 256
	}

	return core.AlignedSize(size)
}

// Run executes the graph using the engine's default arena and pre-initialized sublates.
func (e *Engine) Run() error { // Parameter arena removed
	if e.arena == nil && len(e.sublates) > 0 { // Check if sublates exist but arena doesn't
		return errors.New("engine arena is nil but sublates exist, inconsistent state")
	}
	// If e.sublates are nil or empty, this loop is a no-op.
	// If e.arena is nil but there are no sublates, it might be fine (e.g. empty graph).

	start := time.Now()

	// Execute each sublate in topological order
	for i, sublate := range e.sublates {
		if sublate == nil {
			continue
		}

		kernelFn := kernels.GetKernel(sublate.KernelID)
		if kernelFn == nil {
			return fmt.Errorf("unknown kernel ID: %d for sublate %d", sublate.KernelID, i)
		}

		// Execute kernel on PayloadProp
		kernelFn(sublate.PayloadProp)

		// Update stats
		if e.opts.EnableStats {
			e.mu.Lock()
			e.stats.KernelExecutions[sublate.KernelID]++
			e.mu.Unlock()
		}

		// Swap buffers for next iteration
		sublate.SwapBuffers()
	}

	// Update execution stats
	if e.opts.EnableStats {
		e.mu.Lock()
		e.stats.TotalExecutions++
		duration := time.Since(start)
		e.stats.AverageLatency = time.Duration(
			(int64(e.stats.AverageLatency)*e.stats.TotalExecutions + int64(duration)) /
				(e.stats.TotalExecutions + 1),
		)
		e.mu.Unlock()
	}

	return nil
}

// ExecuteStreaming processes streaming input data
func (e *Engine) ExecuteStreaming(input, output []byte) error {
	if !e.opts.Streaming {
		return fmt.Errorf("engine not configured for streaming")
	}

	// Write input to streaming window
	if err := e.arena.WriteToStreamingInput(input); err != nil {
		return fmt.Errorf("failed to write streaming input: %w", err)
	}

	// Execute the graph
	if err := e.Run(); err != nil { // Changed from e.Run(nil)
		return err
	}

	// Read output from first sublate's PayloadProp
	if len(e.sublates) > 0 && e.sublates[0] != nil {
		outputSize := len(e.sublates[0].PayloadProp)
		if outputSize > len(output) {
			outputSize = len(output)
		}
		copy(output[:outputSize], e.sublates[0].PayloadProp[:outputSize])
	}

	return nil
}

// ArenaBytes returns the arena size in bytes
func (e *Engine) ArenaBytes() int {
	return int(e.arena.TotalSize())
}

// Stats returns current execution statistics
func (e *Engine) Stats() ExecutionStats {
	e.mu.RLock()
	defer e.mu.RUnlock()

	// Return a copy to avoid races
	stats := e.stats
	stats.KernelExecutions = make(map[uint8]int64)
	for k, v := range e.stats.KernelExecutions {
		stats.KernelExecutions[k] = v
	}

	return stats
}

// nodesAsBytes converts a slice of nodes to a byte slice for direct memory operations.
// It assumes model.NodeSize() returns the size of a single model.Node in bytes.
func nodesAsBytes(nodes []model.Node) []byte {
	if len(nodes) == 0 {
		return nil
	}
	// Calculate the total size in bytes. model.NodeSize() is expected to return int.
	// This size must match the expected size for memory operations like copy.
	totalBytes := len(nodes) * model.NodeSize()

	// Get a pointer to the first element of the slice.
	ptr := unsafe.Pointer(&nodes[0])

	// Use unsafe.Slice to create a []byte view over the []model.Node data.
	// This requires Go 1.17+.
	// The returned slice shares the underlying memory with the original nodes slice.
	// Modifications to this byteSlice will modify the original nodes slice and vice-versa.
	byteSlice := unsafe.Slice((*byte)(ptr), totalBytes)
	return byteSlice
}

// Load reads a .subl file and constructs an Engine
func Load(path string) (*Engine, error) {
	buf, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	read := 0
	if len(buf) < 8 {
		return nil, errors.New("invalid model file: too small")
	}

	nodeCnt := int(binary.LittleEndian.Uint32(buf[read:]))
	read += 4
	payloadLen := int(binary.LittleEndian.Uint32(buf[read:]))
	read += 4

	nodes := make([]model.Node, nodeCnt)
	copySize := nodeCnt * model.NodeSize()
	if len(buf) < read+copySize+payloadLen {
		return nil, errors.New("invalid model file: inconsistent sizes")
	}

	copy(nodesAsBytes(nodes), buf[read:read+copySize])
	read += copySize

	payload := make([]byte, payloadLen)
	copy(payload, buf[read:read+payloadLen])

	graph := &model.Graph{Nodes: nodes, Payload: payload}
	opts := DefaultEngineOptions()
	// Ensure NewEngine calculates arena size based on the full graph structure,
	// not just payload length. calculateArenaSize considers node data, metadata, and scratch.
	opts.ArenaSize = 0 // Force auto-calculation in NewEngine

	return NewEngine(graph, &opts)
}

// LoadFromFile reads a .subl file and constructs a Graph (alias for Load for compatibility)
func LoadFromFile(path string) (*model.Graph, error) {
	engine, err := Load(path)
	if err != nil {
		return nil, err
	}
	return engine.graph, nil
}

// SetWorkers configures the number of worker goroutines for parallel execution
func (e *Engine) SetWorkers(n int) {
	if n > 0 {
		e.workers = n
	}
}

// runStreaming executes using the dependency-aware scheduler
func (e *Engine) runStreaming(arena *Arena) {
	var wg sync.WaitGroup

	// Start worker goroutines
	for i := 0; i < e.workers; i++ {
		wg.Add(1)
		go e.worker(arena, &wg)
	}

	// Schedule initial ready tasks
	e.scheduleReady()

	// Wait for completion
	wg.Wait()
}

// worker processes tasks from the ready queue
func (e *Engine) worker(arena *Arena, wg *sync.WaitGroup) {
	defer wg.Done()
	buffer := arena.Buffer()

	for taskGroup := range e.scheduler.ready {
		// Process all nodes in the task group concurrently
		var groupWg sync.WaitGroup

		for _, node := range taskGroup.nodes {
			groupWg.Add(1)

			go func(n model.Node) {
				defer groupWg.Done()

				kernel := kernelCatalog[n.Kernel]
				if kernel == nil {
					return
				}

				offset := int(n.Out)
				if offset < len(buffer) {
					kernel(buffer[offset:])
				}
			}(node)
		}

		// Wait for all nodes in group to complete
		groupWg.Wait()

		// Signal completion to scheduler
		for _, node := range taskGroup.nodes {
			e.scheduler.completed <- node.ID
		}
	}
}

// scheduleReady moves ready task groups to the execution queue
func (e *Engine) scheduleReady() {
	scheduled := make(map[uint16]bool)

	e.scheduleInitialReady(scheduled)
	e.startCompletionHandler(scheduled)
}

// scheduleInitialReady schedules tasks with no dependencies
func (e *Engine) scheduleInitialReady(scheduled map[uint16]bool) {
	for level, taskGroup := range e.scheduler.waiting {
		if len(taskGroup.nodes) == 0 {
			continue
		}

		if e.isTaskGroupReady(taskGroup, scheduled) {
			e.scheduleTaskGroup(level, taskGroup, scheduled)
		}
	}
}

// isTaskGroupReady checks if all dependencies for a task group are satisfied
func (e *Engine) isTaskGroupReady(taskGroup *TaskGroup, scheduled map[uint16]bool) bool {
	for _, node := range taskGroup.nodes {
		for _, depID := range e.scheduler.deps[node.ID] {
			if !scheduled[depID] {
				return false
			}
		}
	}
	return true
}

// scheduleTaskGroup moves a task group to ready queue and marks nodes as scheduled
func (e *Engine) scheduleTaskGroup(level uint16, taskGroup *TaskGroup, scheduled map[uint16]bool) {
	e.scheduler.ready <- taskGroup
	for _, node := range taskGroup.nodes {
		scheduled[node.ID] = true
	}
	delete(e.scheduler.waiting, level)
}

// startCompletionHandler manages task completion and schedules new ready tasks
func (e *Engine) startCompletionHandler(scheduled map[uint16]bool) {
	go func() {
		defer close(e.scheduler.ready)

		for len(e.scheduler.waiting) > 0 {
			nodeID := <-e.scheduler.completed
			scheduled[nodeID] = true

			e.checkAndScheduleNewReady(scheduled)
		}
	}()
}

// checkAndScheduleNewReady checks for newly ready tasks after completion
func (e *Engine) checkAndScheduleNewReady(scheduled map[uint16]bool) {
	for level, taskGroup := range e.scheduler.waiting {
		if e.isTaskGroupReady(taskGroup, scheduled) {
			e.scheduleTaskGroup(level, taskGroup, scheduled)
			break // Process one at a time to avoid concurrent map modification
		}
	}
}

// Execute runs the model with enhanced execution context
func (e *Engine) Execute(ctx *ExecutionContext) error {
	arena, err := e.setupExecutionArena()
	if err != nil {
		return err
	}

	if err := e.prepareExecution(arena); err != nil {
		return err
	}

	start := time.Now()

	if err := e.runExecution(arena); err != nil {
		return err
	}

	return e.updateExecutionStats(start)
}

// setupExecutionArena creates and configures arena for execution
func (e *Engine) setupExecutionArena() (*Arena, error) {
	arenaTotalSize := e.opts.ArenaSize

	sizes, err := calculateArenaSizes(arenaTotalSize, e.opts.Streaming, e.graph)
	if err != nil {
		return nil, err
	}

	arena, err := NewArena(arenaTotalSize, e.graph, sizes.nodePayloads, sizes.streaming, sizes.scratch)
	if err != nil {
		return nil, fmt.Errorf("failed to create arena for execution: %w", err)
	}

	if arena == nil {
		return nil, errors.New("failed to create arena for execution (arena is nil despite no error)")
	}

	return arena, nil
}

// prepareExecution initializes sublates and copies payload
func (e *Engine) prepareExecution(arena *Arena) error {
	if err := e.initializeSublates(e.graph, arena); err != nil {
		return fmt.Errorf("failed to initialize sublates for execution: %w", err)
	}

	// Copy payload to model payload region
	if len(e.graph.Payload) > 0 {
		if modelPayload, err := arena.ModelPayload(uintptr(len(e.graph.Payload))); err == nil && modelPayload != nil {
			copy(modelPayload, e.graph.Payload)
		}
	}

	return nil
}

// runExecution executes the model using streaming or sequential mode
func (e *Engine) runExecution(arena *Arena) error {
	if e.opts.Streaming {
		return e.runStreamingExecution(arena)
	}
	return e.runSequentialExecution()
}

// runStreamingExecution handles streaming mode execution
func (e *Engine) runStreamingExecution(arena *Arena) error {
	if e.scheduler == nil {
		return fmt.Errorf("engine is configured for streaming but scheduler is not initialized (workers: %d)", e.workers)
	}
	e.runStreaming(arena)
	return nil
}

// runSequentialExecution handles non-streaming sequential execution
func (e *Engine) runSequentialExecution() error {
	for i, sublate := range e.sublates {
		if sublate == nil {
			continue
		}

		if err := e.executeSublate(i, sublate); err != nil {
			return err
		}

		sublate.SwapBuffers()
	}
	return nil
}

// executeSublate runs a single sublate's kernel
func (e *Engine) executeSublate(index int, sublate *core.Sublate) error {
	kernelFn := kernels.GetKernel(sublate.KernelID)
	if kernelFn == nil {
		return fmt.Errorf("unknown kernel ID: %d for sublate %d", sublate.KernelID, index)
	}

	kernelFn(sublate.PayloadProp)

	if e.opts.EnableStats {
		e.updateKernelStats(sublate.KernelID)
	}

	return nil
}

// updateKernelStats safely updates kernel execution statistics
func (e *Engine) updateKernelStats(kernelID uint8) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.stats.KernelExecutions == nil {
		e.stats.KernelExecutions = make(map[uint8]int64)
	}
	e.stats.KernelExecutions[kernelID]++
}

// updateExecutionStats updates total executions and average latency
func (e *Engine) updateExecutionStats(start time.Time) error {
	if !e.opts.EnableStats {
		return nil
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	e.stats.TotalExecutions++
	duration := time.Since(start)

	if e.stats.TotalExecutions == 1 {
		e.stats.AverageLatency = duration
	} else {
		oldTotal := e.stats.TotalExecutions - 1
		if oldTotal < 0 {
			oldTotal = 0
		}
		e.stats.AverageLatency = time.Duration((int64(e.stats.AverageLatency)*oldTotal + int64(duration)) / e.stats.TotalExecutions)
	}

	return nil
}

// initializeSublates creates sublates from the graph model
// Assuming the signature is (e *Engine) initializeSublates(graph *model.Graph, arena *Arena) error
func (e *Engine) initializeSublates(graph *model.Graph, arena *Arena) error {
	if arena == nil {
		return errors.New("arena is nil in initializeSublates")
	}
	if e.sublates == nil || len(e.sublates) != len(graph.Nodes) {
		return fmt.Errorf("engine sublates slice not correctly initialized (len: %d, expected: %d)", len(e.sublates), len(graph.Nodes))
	}

	modelPayloadBytes, err := arena.ModelPayload(uintptr(len(graph.Payload)))
	if err != nil && len(graph.Payload) > 0 {
		return fmt.Errorf("could not get model payload view from arena: %w", err)
	}

	for i, node := range graph.Nodes {
		sublatePtr, err := arena.GetSublateAtIndex(i)
		if err != nil {
			return fmt.Errorf("failed to get sublate struct %d from arena: %w", i, err)
		}
		e.sublates[i] = sublatePtr

		if err := e.initializeSublateFields(sublatePtr, &node, graph, modelPayloadBytes, arena); err != nil {
			return fmt.Errorf("failed to initialize fields for sublate %d: %w", i, err)
		}
	}
	return nil
}

func (e *Engine) initializeSublateFields(sublatePtr *core.Sublate, node *model.Node, graph *model.Graph, modelPayloadBytes []byte, arena *Arena) error {
	sublatePtr.KernelID = node.Kernel
	sublatePtr.Flags = node.Flags
	if len(node.Topo) > 0 {
		sublatePtr.Topology = make([]uint16, len(node.Topo))
		copy(sublatePtr.Topology, node.Topo)
	} else {
		sublatePtr.Topology = nil
	}

	if err := e.allocateSublatePayloads(sublatePtr, node, graph, arena); err != nil {
		return err
	}

	return e.copyInitialPayloadData(sublatePtr, node, modelPayloadBytes)
}

func (e *Engine) allocateSublatePayloads(sublatePtr *core.Sublate, node *model.Node, graph *model.Graph, arena *Arena) error {
	payloadSize := uintptr(calculateNodePayloadSize(node, graph))
	alignedPayloadSize := core.AlignedSize(payloadSize)

	if alignedPayloadSize > 0 {
		prevPayload, err := arena.AllocateNodePayload(alignedPayloadSize, core.CacheLineSize)
		if err != nil {
			return fmt.Errorf("failed to allocate PayloadPrev from arena node payloads: %w", err)
		}
		sublatePtr.PayloadPrev = prevPayload

		propPayload, err := arena.AllocateNodePayload(alignedPayloadSize, core.CacheLineSize)
		if err != nil {
			return fmt.Errorf("failed to allocate PayloadProp from arena node payloads: %w", err)
		}
		sublatePtr.PayloadProp = propPayload
	} else {
		sublatePtr.PayloadPrev = nil
		sublatePtr.PayloadProp = nil
	}
	return nil
}

func (e *Engine) copyInitialPayloadData(sublatePtr *core.Sublate, node *model.Node, modelPayloadBytes []byte) error {
	if len(modelPayloadBytes) > 0 && node.Out > node.In {
		sourceDataLen := int(node.Out - node.In)
		if sourceDataLen > 0 {
			sourceSlice := modelPayloadBytes[node.In:node.Out]
			if len(sublatePtr.PayloadPrev) >= sourceDataLen {
				copy(sublatePtr.PayloadPrev, sourceSlice)
			} else if len(sublatePtr.PayloadPrev) > 0 {
				copy(sublatePtr.PayloadPrev, sourceSlice[:len(sublatePtr.PayloadPrev)])
				// Consider logging a warning here about truncation
			}
		}
	}
	return nil
}

// calculateNodePayloadSize determines buffer size needed for a node
// This is a placeholder function. The actual logic would be more complex.
func calculateNodePayloadSize(node *model.Node, _ *model.Graph) int { // graph parameter marked as unused
	// A more robust implementation would:
	// 1. Look up kernel properties associated with node.Kernel. Some kernels might have fixed input/output sizes.
	// 2. Consider node.Flags if they influence size.
	// 3. If node.In and node.Out define a segment in graph.Payload, that's a strong hint for input size.
	//    The output size might be the same or different depending on the kernel.
	// 4. For now, use a simple heuristic: if In/Out are valid, use that length. Otherwise, a default.

	if node.Out > node.In {
		// This assumes In/Out are byte offsets for this node's primary data segment.
		// The Sublate architecture implies dual buffers, usually of the same size.
		size := int(node.Out - node.In)
		if size > 0 {
			return size
		}
	}

	// Fallback to a default size if In/Out don't yield a positive size.
	// This default should be large enough for typical operations or based on some global config.
	// As per instructions, memory is statically planned, so this should ideally not be a "guess".
	// For testing, a small default might work.
	// The original placeholder in the prompt used 256.
	// Let's use a value that might be common, e.g., if nodes operate on vectors of a certain length.
	// If a node represents a vector of 64 float32s: 64 * 4 = 256 bytes.
	return 256 // Default fallback size in bytes.
}

// WorkStealingScheduler implements work-stealing for load balancing
type WorkStealingScheduler struct {
	localQueues []chan *core.Sublate
	globalQueue chan *core.Sublate
	workers     int
}

// NewWorkStealingScheduler creates a work-stealing scheduler for fine-grained tasks
func NewWorkStealingScheduler(workers int) *WorkStealingScheduler {
	ws := &WorkStealingScheduler{
		localQueues: make([]chan *core.Sublate, workers),
		globalQueue: make(chan *core.Sublate, workers*4),
		workers:     workers,
	}

	for i := range ws.localQueues {
		ws.localQueues[i] = make(chan *core.Sublate, 16)
	}

	return ws
}

// SubmitWork adds work to a worker's local queue
func (ws *WorkStealingScheduler) SubmitWork(workerID int, sublate *core.Sublate) {
	select {
	case ws.localQueues[workerID] <- sublate:
	default:
		// Local queue full, submit to global queue
		ws.globalQueue <- sublate
	}
}

// GetWork attempts to get work from local queue, then steals from others
func (ws *WorkStealingScheduler) GetWork(workerID int) *core.Sublate {
	// Try local queue first
	select {
	case work := <-ws.localQueues[workerID]:
		return work
	default:
	}

	// Try global queue
	select {
	case work := <-ws.globalQueue:
		return work
	default:
	}

	// Try to steal from other workers
	for i := 0; i < ws.workers; i++ {
		if i == workerID {
			continue
		}

		select {
		case work := <-ws.localQueues[i]:
			return work
		default:
		}
	}

	return nil
}

// ArenaAllocator manages memory allocation within a fixed arena
type ArenaAllocator struct {
	buf    []byte
	offset int
	mutex  sync.Mutex
}

// NewArenaAllocator creates a memory arena allocator
func NewArenaAllocator(size int) *ArenaAllocator {
	return &ArenaAllocator{
		buf: make([]byte, size),
	}
}

// Allocate returns a slice from the arena with specified size and alignment
func (a *ArenaAllocator) Allocate(size, align int) []byte {
	a.mutex.Lock()
	defer a.mutex.Unlock()

	// Align offset
	aligned := (a.offset + align - 1) &^ (align - 1)

	if aligned+size > len(a.buf) {
		return nil // Out of memory
	}

	result := a.buf[aligned : aligned+size]
	a.offset = aligned + size

	return result
}

// Reset clears the arena for reuse
func (a *ArenaAllocator) Reset() {
	a.mutex.Lock()
	a.offset = 0
	a.mutex.Unlock()
}

// Available returns remaining space in arena
func (a *ArenaAllocator) Available() int {
	a.mutex.Lock()
	defer a.mutex.Unlock()
	return len(a.buf) - a.offset
}

// BufferPool manages reusable byte slices for sublate payloads
type BufferPool struct {
	buffers chan []byte
	size    int
}

// NewBufferPool creates a pool of reusable buffers
func NewBufferPool(poolSize, bufferSize int) *BufferPool {
	bp := &BufferPool{
		buffers: make(chan []byte, poolSize),
		size:    bufferSize,
	}

	// Pre-allocate buffers
	for i := 0; i < poolSize; i++ {
		bp.buffers <- make([]byte, bufferSize)
	}

	return bp
}

// GetBuffer returns a buffer from the pool or creates a new one
func (bp *BufferPool) GetBuffer() []byte {
	select {
	case buf := <-bp.buffers:
		return buf
	default:
		return make([]byte, bp.size)
	}
}

// PutBuffer returns a buffer to the pool
func (bp *BufferPool) PutBuffer(buf []byte) {
	if len(buf) == bp.size {
		select {
		case bp.buffers <- buf:
		default:
			// Pool full, let GC handle it
		}
	}
}

// SublatePool manages reusable Sublate instances
type SublatePool struct {
	sublates chan *core.Sublate
	size     int
}

// NewSublatePool creates a pool of reusable sublates
func NewSublatePool(poolSize int) *SublatePool {
	sp := &SublatePool{
		sublates: make(chan *core.Sublate, poolSize),
		size:     poolSize,
	}

	// Pre-allocate sublates
	for i := 0; i < poolSize; i++ {
		sp.sublates <- &core.Sublate{}
	}

	return sp
}

// Get returns a sublate from the pool or creates a new one
func (sp *SublatePool) Get() *core.Sublate {
	select {
	case sublate := <-sp.sublates:
		return sublate
	default:
		return &core.Sublate{}
	}
}

// Put returns a sublate to the pool
func (sp *SublatePool) Put(sublate *core.Sublate) {
	// Reset sublate state
	sublate.KernelID = 0
	sublate.Flags = 0
	sublate.Topology = nil
	sublate.PayloadPrev = nil
	sublate.PayloadProp = nil

	select {
	case sp.sublates <- sublate:
	default:
		// Pool full, let GC handle it
	}
}

// ExecutionContext provides execution state and resource pools
type ExecutionContext struct {
	sublates []*core.Sublate
	pool     *SublatePool
	bufPool  *BufferPool
}

// NewExecutionContext creates a new execution context with resource pools
func NewExecutionContext(maxSublates int) *ExecutionContext {
	return &ExecutionContext{
		pool:     NewSublatePool(maxSublates),
		bufPool:  NewBufferPool(maxSublates*2, 1024), // 2 buffers per sublate, 1KB each
		sublates: make([]*core.Sublate, 0, maxSublates),
	}
}

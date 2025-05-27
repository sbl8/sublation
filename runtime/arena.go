package runtime

import (
	"encoding/binary"
	"errors"
	"fmt"
	"unsafe"

	"github.com/sbl8/sublation/core"
	"github.com/sbl8/sublation/model"
)

// ArenaRegion represents a distinct memory region within the Arena.
type ArenaRegion struct {
	Offset uintptr
	Size   uintptr
	Name   string
}

// Arena manages a single pre-allocated byte slice for all runtime data.
// It is structured according to the memory model in architecture_overview.md:
// 1. Model Payload (immutable parameters)
// 2. Sublate Metadata (per-node offsets, flags, topology)
// 3. Scratch Buffers (reused temp slices)
// 4. Streaming Input Window (active batch)
// 5. Free Tail (head-room for growth/hot-swap)
type Arena struct {
	buffer       []byte // The underlying raw memory buffer
	regions      map[string]ArenaRegion
	modelPayload ArenaRegion // Immutable parameters
	sublateMeta  ArenaRegion // Per-node offsets, flags, topology for core.Sublate structs
	// Note: The actual data for Sublate.PayloadPrev, Sublate.PayloadProp, Sublate.Topology
	// might be stored in dedicated sections within ModelPayload or Scratch, or directly after each Sublate struct.
	// This design assumes Sublate structs are in sublateMeta, and their slice fields point to other arena locations.

	nodePayloads   ArenaRegion // Region for Sublate.PayloadPrev and Sublate.PayloadProp data
	scratch        ArenaRegion // Reused temp slices
	streamingInput ArenaRegion // Active batch
	freeTail       ArenaRegion // Head-room for growth / hot-swap

	currentNodePayloadOffset uintptr // Bump allocator for nodePayloads region
	currentScratchOffset     uintptr // Bump allocator for scratch region
}

const (
	// DefaultAlignment is used for general allocations within the arena.
	DefaultAlignment = 8 // 8-byte alignment (could be core.CacheLineSize for stricter alignment)
)

// NewArena initializes a new Arena with a given total size and graph definition.
func NewArena(totalSize uintptr, graph *model.Graph, nodePayloadsSize uintptr, streamingInputSize uintptr, kernelScratchSize uintptr) (*Arena, error) {
	if err := validateArenaInputs(totalSize, graph, nodePayloadsSize, streamingInputSize, kernelScratchSize); err != nil {
		return nil, err
	}

	effectiveTotalSize, err := calculateEffectiveSize(totalSize, graph, nodePayloadsSize, streamingInputSize, kernelScratchSize)
	if err != nil {
		return nil, err
	}

	arena, err := createArenaBuffer(effectiveTotalSize)
	if err != nil {
		return nil, err
	}

	return layoutArenaRegions(arena, graph, nodePayloadsSize, streamingInputSize, kernelScratchSize, effectiveTotalSize)
}

// validateArenaInputs validates the input parameters for arena creation
func validateArenaInputs(totalSize uintptr, graph *model.Graph, nodePayloadsSize uintptr, streamingInputSize uintptr, kernelScratchSize uintptr) error {
	if totalSize == 0 && (graph == nil || uintptr(len(graph.Payload)) == 0) && nodePayloadsSize == 0 && streamingInputSize == 0 && kernelScratchSize == 0 {
		return fmt.Errorf("cannot create zero-size arena with all component sizes zero")
	}
	if graph == nil {
		return fmt.Errorf("graph cannot be nil")
	}
	return nil
}

// calculateEffectiveSize computes the actual arena size needed
func calculateEffectiveSize(totalSize uintptr, graph *model.Graph, nodePayloadsSize uintptr, streamingInputSize uintptr, kernelScratchSize uintptr) (uintptr, error) {
	minRequiredSize := calculateMinRequiredSize(graph, nodePayloadsSize, streamingInputSize, kernelScratchSize)

	if totalSize == 0 {
		return minRequiredSize, nil
	}

	alignedUserTotalSize := core.AlignedSize(totalSize)
	if alignedUserTotalSize < minRequiredSize {
		return 0, fmt.Errorf("user-provided totalSize %d (aligned to %d) is less than minimum required size %d", totalSize, alignedUserTotalSize, minRequiredSize)
	}

	return alignedUserTotalSize, nil
}

// calculateMinRequiredSize computes the minimum size needed for all components
func calculateMinRequiredSize(graph *model.Graph, nodePayloadsSize uintptr, streamingInputSize uintptr, kernelScratchSize uintptr) uintptr {
	minRequiredSize := uintptr(0)

	// Model payload
	actualPayloadSize := uintptr(len(graph.Payload))
	if actualPayloadSize > 0 {
		minRequiredSize += core.AlignedSize(actualPayloadSize)
	}

	// Sublate metadata
	numNodes := uintptr(len(graph.Nodes))
	if numNodes > 0 {
		sublateStructSize := unsafe.Sizeof(core.Sublate{})
		alignedSublateStructSize := core.AlignedSize(sublateStructSize)
		minRequiredSize += core.AlignedSize(numNodes * alignedSublateStructSize)
	}

	// Optional regions
	if nodePayloadsSize > 0 {
		minRequiredSize += core.AlignedSize(nodePayloadsSize)
	}
	if kernelScratchSize > 0 {
		minRequiredSize += core.AlignedSize(kernelScratchSize)
	}
	if streamingInputSize > 0 {
		minRequiredSize += core.AlignedSize(streamingInputSize)
	}

	return core.AlignedSize(minRequiredSize)
}

// createArenaBuffer allocates the arena buffer
func createArenaBuffer(effectiveTotalSize uintptr) (*Arena, error) {
	arena := &Arena{
		buffer:  core.AlignedBytes(int(effectiveTotalSize)),
		regions: make(map[string]ArenaRegion),
	}

	if arena.buffer == nil && effectiveTotalSize > 0 {
		return nil, fmt.Errorf("failed to allocate arena buffer of size %d", effectiveTotalSize)
	}

	return arena, nil
}

// layoutArenaRegions partitions the arena into regions
func layoutArenaRegions(arena *Arena, graph *model.Graph, nodePayloadsSize uintptr, streamingInputSize uintptr, kernelScratchSize uintptr, effectiveTotalSize uintptr) (*Arena, error) {
	currentOffset := uintptr(0)

	// Layout each region in order
	currentOffset = layoutModelPayload(arena, graph, currentOffset)
	currentOffset = layoutSublateMetadata(arena, graph, currentOffset)
	currentOffset = layoutNodePayloads(arena, nodePayloadsSize, currentOffset)
	currentOffset = layoutScratchBuffers(arena, kernelScratchSize, currentOffset)
	currentOffset = layoutStreamingInput(arena, streamingInputSize, currentOffset)
	currentOffset = layoutFreeTail(arena, effectiveTotalSize, currentOffset)

	if currentOffset > effectiveTotalSize {
		return nil, fmt.Errorf("arena layout exceeds total size: %d > %d", currentOffset, effectiveTotalSize)
	}

	return arena, nil
}

// layoutModelPayload sets up the model payload region
func layoutModelPayload(arena *Arena, graph *model.Graph, currentOffset uintptr) uintptr {
	actualPayloadSize := uintptr(len(graph.Payload))
	if actualPayloadSize == 0 {
		return currentOffset
	}

	currentOffset = core.AlignedSize(currentOffset)
	regionSize := core.AlignedSize(actualPayloadSize)
	arena.modelPayload = ArenaRegion{Offset: currentOffset, Size: regionSize, Name: "ModelPayload"}
	arena.regions["ModelPayload"] = arena.modelPayload
	copy(arena.buffer[currentOffset:currentOffset+actualPayloadSize], graph.Payload)

	return currentOffset + regionSize
}

// layoutSublateMetadata sets up the sublate metadata region
func layoutSublateMetadata(arena *Arena, graph *model.Graph, currentOffset uintptr) uintptr {
	numNodes := uintptr(len(graph.Nodes))
	if numNodes == 0 {
		return currentOffset
	}

	sublateStructSize := unsafe.Sizeof(core.Sublate{})
	alignedSublateStructSize := core.AlignedSize(sublateStructSize)
	totalSublateMetaSize := numNodes * alignedSublateStructSize

	currentOffset = core.AlignedSize(currentOffset)
	arena.sublateMeta = ArenaRegion{Offset: currentOffset, Size: totalSublateMetaSize, Name: "SublateMetadata"}
	arena.regions["SublateMetadata"] = arena.sublateMeta

	return currentOffset + totalSublateMetaSize
}

// layoutNodePayloads sets up the node payloads region
func layoutNodePayloads(arena *Arena, nodePayloadsSize uintptr, currentOffset uintptr) uintptr {
	if nodePayloadsSize == 0 {
		return currentOffset
	}

	currentOffset = core.AlignedSize(currentOffset)
	arena.nodePayloads = ArenaRegion{Offset: currentOffset, Size: nodePayloadsSize, Name: "NodePayloads"}
	arena.regions["NodePayloads"] = arena.nodePayloads
	arena.currentNodePayloadOffset = currentOffset

	return currentOffset + nodePayloadsSize
}

// layoutScratchBuffers sets up the scratch buffer region
func layoutScratchBuffers(arena *Arena, kernelScratchSize uintptr, currentOffset uintptr) uintptr {
	if kernelScratchSize == 0 {
		return currentOffset
	}

	currentOffset = core.AlignedSize(currentOffset)
	arena.scratch = ArenaRegion{Offset: currentOffset, Size: kernelScratchSize, Name: "Scratch"}
	arena.regions["Scratch"] = arena.scratch
	arena.currentScratchOffset = currentOffset

	return currentOffset + kernelScratchSize
}

// layoutStreamingInput sets up the streaming input region
func layoutStreamingInput(arena *Arena, streamingInputSize uintptr, currentOffset uintptr) uintptr {
	if streamingInputSize == 0 {
		return currentOffset
	}

	currentOffset = core.AlignedSize(currentOffset)
	arena.streamingInput = ArenaRegion{Offset: currentOffset, Size: streamingInputSize, Name: "StreamingInput"}
	arena.regions["StreamingInput"] = arena.streamingInput

	return currentOffset + streamingInputSize
}

// layoutFreeTail sets up the remaining free space
func layoutFreeTail(arena *Arena, effectiveTotalSize uintptr, currentOffset uintptr) uintptr {
	currentOffset = core.AlignedSize(currentOffset)
	freeTailSize := uintptr(0)
	if effectiveTotalSize > currentOffset {
		freeTailSize = effectiveTotalSize - currentOffset
	}

	arena.freeTail = ArenaRegion{Offset: currentOffset, Size: freeTailSize, Name: "FreeTail"}
	arena.regions["FreeTail"] = arena.freeTail

	return currentOffset + freeTailSize
}

// Buffer returns the raw byte buffer of the arena.
func (a *Arena) Buffer() []byte {
	return a.buffer
}

// Region returns the specified ArenaRegion.
func (a *Arena) Region(name string) (ArenaRegion, bool) {
	region, ok := a.regions[name]
	return region, ok
}

// ModelPayload returns a slice to the model payload region.
// The returned slice covers the actual payload size, not the aligned size.
func (a *Arena) ModelPayload(graphPayloadLen uintptr) ([]byte, error) {
	if a.modelPayload.Size == 0 {
		return nil, errors.New("no payload region defined")
	}
	if graphPayloadLen > a.modelPayload.Size {
		return nil, fmt.Errorf("requested payload size %d exceeds region size %d", graphPayloadLen, a.modelPayload.Size)
	}
	return a.buffer[a.modelPayload.Offset : a.modelPayload.Offset+graphPayloadLen], nil
}

// SublateMetadataRaw returns a slice to the raw sublate metadata region.
// This region contains the core.Sublate structs.
func (a *Arena) SublateMetadataRaw() ([]byte, error) {
	if a.sublateMeta.Size == 0 {
		return nil, errors.New("no sublate metadata region defined")
	}
	return a.buffer[a.sublateMeta.Offset : a.sublateMeta.Offset+a.sublateMeta.Size], nil
}

// GetSublateAtIndex returns a pointer to the core.Sublate struct at a given index
// within the SublateMeta region.
func (a *Arena) GetSublateAtIndex(index int) (*core.Sublate, error) {
	if a.sublateMeta.Size == 0 {
		return nil, errors.New("sublate metadata region is not initialized or is empty")
	}

	sublateStructSize := unsafe.Sizeof(core.Sublate{})
	alignedSublateStructSize := core.AlignedSize(sublateStructSize)

	offsetInRegion := uintptr(index) * alignedSublateStructSize

	if offsetInRegion >= a.sublateMeta.Size || offsetInRegion+alignedSublateStructSize > a.sublateMeta.Size {
		return nil, fmt.Errorf("index %d out of bounds for sublate metadata region (size %d, sublate aligned size %d, requested offset %d, end %d)", index, a.sublateMeta.Size, alignedSublateStructSize, offsetInRegion, offsetInRegion+alignedSublateStructSize)
	}

	absOffset := a.sublateMeta.Offset + offsetInRegion
	// Final check against main buffer length
	if absOffset >= uintptr(len(a.buffer)) || absOffset+alignedSublateStructSize > uintptr(len(a.buffer)) {
		return nil, fmt.Errorf("absolute offset %d for sublate %d (size %d) is out of arena buffer bounds (%d)", absOffset, index, alignedSublateStructSize, len(a.buffer))
	}
	// Ensure that the pointer conversion is safe by checking if the buffer is large enough.
	// This check is technically covered by the absOffset+alignedSublateStructSize check above,
	// but an explicit check against len(a.buffer) for the start of the slice is good practice.
	if int(absOffset) >= len(a.buffer) {
		return nil, fmt.Errorf("absolute offset %d for sublate %d is at or beyond arena buffer end (%d)", absOffset, index, len(a.buffer))
	}
	return (*core.Sublate)(unsafe.Pointer(&a.buffer[absOffset])), nil
}

// AllocateNodePayload allocates a slice from the node payloads region using a bump allocator.
// Not thread-safe without external locking.
func (a *Arena) AllocateNodePayload(size uintptr, alignment uintptr) ([]byte, error) {
	if a.nodePayloads.Size == 0 {
		return nil, errors.New("no node payloads region defined")
	}
	if alignment == 0 {
		alignment = DefaultAlignment
	}

	alignedOffset := (a.currentNodePayloadOffset + alignment - 1) &^ (alignment - 1)
	if alignedOffset+size > a.nodePayloads.Offset+a.nodePayloads.Size {
		return nil, fmt.Errorf("node payloads region exhausted: requested %d, available approx %d from current offset %d in region size %d", size, (a.nodePayloads.Offset+a.nodePayloads.Size)-alignedOffset, a.currentNodePayloadOffset, a.nodePayloads.Size)
	}

	result := a.buffer[alignedOffset : alignedOffset+size]
	a.currentNodePayloadOffset = alignedOffset + size
	return result, nil
}

// ResetNodePayloads resets the bump allocator for the node payloads region.
func (a *Arena) ResetNodePayloads() {
	a.currentNodePayloadOffset = a.nodePayloads.Offset
}

// AllocateScratch allocates a slice from the scratch buffer region using a bump allocator.
// Not thread-safe without external locking.
func (a *Arena) AllocateScratch(size uintptr, alignment uintptr) ([]byte, error) {
	if a.scratch.Size == 0 {
		return nil, errors.New("no scratch region defined")
	}
	if alignment == 0 {
		alignment = DefaultAlignment
	}

	alignedOffset := (a.currentScratchOffset + alignment - 1) &^ (alignment - 1)
	if alignedOffset+size > a.scratch.Offset+a.scratch.Size {
		return nil, errors.New("scratch region exhausted")
	}

	result := a.buffer[alignedOffset : alignedOffset+size]
	a.currentScratchOffset = alignedOffset + size
	return result, nil
}

// ResetScratch resets the bump allocator for the scratch region.
func (a *Arena) ResetScratch() {
	a.currentScratchOffset = a.scratch.Offset
}

// StreamingInputWindow returns a slice to the streaming input window.
func (a *Arena) StreamingInputWindow() ([]byte, error) {
	if a.streamingInput.Size == 0 {
		return nil, errors.New("no streaming input region defined")
	}
	return a.buffer[a.streamingInput.Offset : a.streamingInput.Offset+a.streamingInput.Size], nil
}

// WriteToStreamingInput copies data into the streaming input window.
func (a *Arena) WriteToStreamingInput(data []byte) error {
	window, err := a.StreamingInputWindow()
	if err != nil {
		return err
	}
	if uintptr(len(data)) > a.streamingInput.Size {
		return fmt.Errorf("data size %d exceeds streaming input size %d", len(data), a.streamingInput.Size)
	}
	copy(window, data)
	return nil
}

// TotalSize returns the total capacity of the arena's buffer.
func (a *Arena) TotalSize() uintptr {
	return uintptr(len(a.buffer))
}

// UsedSize calculates the currently "committed" size of the arena,
// up to the start of the FreeTail.
func (a *Arena) UsedSize() uintptr {
	return a.freeTail.Offset
}

// RemainingSize returns the size of the FreeTail.
func (a *Arena) RemainingSize() uintptr {
	return a.freeTail.Size
}

// WriteAt writes data to the arena at a specific offset.
func (a *Arena) WriteAt(offset uintptr, data []byte) error {
	if offset+uintptr(len(data)) > uintptr(len(a.buffer)) {
		return fmt.Errorf("write exceeds buffer bounds")
	}
	copy(a.buffer[offset:offset+uintptr(len(data))], data)
	return nil
}

// ReadAt reads data from the arena at a specific offset.
func (a *Arena) ReadAt(offset uintptr, size uintptr) ([]byte, error) {
	if offset+size > uintptr(len(a.buffer)) {
		return nil, fmt.Errorf("read exceeds buffer bounds")
	}
	return a.buffer[offset : offset+size], nil
}

// ZeroRegion sets all bytes in a given region to zero.
func (a *Arena) ZeroRegion(regionName string) error {
	region, ok := a.regions[regionName]
	if !ok {
		return fmt.Errorf("region %s not found", regionName)
	}
	for i := region.Offset; i < region.Offset+region.Size; i++ {
		a.buffer[i] = 0
	}
	return nil
}

// FloatsToBytes converts a slice of float32 to a byte slice using LittleEndian encoding.
func FloatsToBytes(f []float32) []byte {
	result := make([]byte, len(f)*4)
	for i, val := range f {
		binary.LittleEndian.PutUint32(result[i*4:(i+1)*4], *(*uint32)(unsafe.Pointer(&val)))
	}
	return result
}

// BytesToFloats converts a byte slice to a float32 slice using LittleEndian encoding.
// Returns an error if the byte slice length is not a multiple of 4.
func BytesToFloats(b []byte) ([]float32, error) {
	if len(b)%4 != 0 {
		return nil, fmt.Errorf("byte slice length %d not multiple of 4", len(b))
	}
	result := make([]float32, len(b)/4)
	for i := 0; i < len(result); i++ {
		val := binary.LittleEndian.Uint32(b[i*4 : (i+1)*4])
		result[i] = *(*float32)(unsafe.Pointer(&val))
	}
	return result, nil
}

// InitSublateInArena initializes a core.Sublate struct at the given index within the arena's
// SublateMeta region. It sets the Sublate's fields based on the model.Node and resolves
// PayloadPrev and PayloadProp to point to appropriate locations within the arena
// (e.g., ModelPayload for weights/inputs, Scratch for activations/outputs).
//
// Parameters:
//   - arena: The Arena instance.
//   - sublateIndex: The index of the Sublate in the SublateMeta region.
//   - modelNode: The corresponding model.Node from the graph.
//   - graphPayloadData: The raw byte slice of the original graph payload (used to calculate content for PayloadPrev).
//   - defaultPayloadSize: A default size for PayloadPrev/PayloadProp if not derivable from modelNode.
//
// This function is critical for bridging the gap between the static graph model and the live runtime Sublates.
// The logic for determining payload sizes and locations (especially for dual buffering) is complex
// and depends on the specific conventions of the Sublation model.
// NOTE: This function seems to be a helper and might be superseded or used by engine.initializeSublates.
// If used, it needs to be updated to use AllocateNodePayload instead of AllocateScratch for PayloadPrev/Prop.
func InitSublateInArena(
	arena *Arena,
	sublateIndex int,
	modelNode *model.Node,
	graphPayloadData []byte,
	defaultPayloadPrevSize uintptr,
	defaultPayloadPropSize uintptr,
) error {
	sublatePtr, err := arena.GetSublateAtIndex(sublateIndex)
	if err != nil {
		return err
	}

	// Initialize basic fields
	sublatePtr.KernelID = modelNode.Kernel
	sublatePtr.Flags = modelNode.Flags
	sublatePtr.Topology = modelNode.Topo

	// Allocate PayloadPrev from model payload or scratch
	prevSize := defaultPayloadPrevSize
	if modelNode.In < uint16(len(graphPayloadData)) {
		// Calculate size based on model structure or use default
		remaining := uintptr(len(graphPayloadData)) - uintptr(modelNode.In)
		if remaining < prevSize {
			prevSize = remaining
		}
	}

	if prevSize > 0 {
		// prevBuf, err := arena.AllocateScratch(prevSize, 8) // Old
		prevBuf, err := arena.AllocateNodePayload(prevSize, core.CacheLineSize) // Changed
		if err != nil {
			return fmt.Errorf("failed to allocate PayloadPrev: %w", err)
		}
		sublatePtr.PayloadPrev = prevBuf

		// Copy initial data if available
		if modelNode.In < uint16(len(graphPayloadData)) {
			copySize := prevSize
			if uintptr(modelNode.In)+copySize > uintptr(len(graphPayloadData)) {
				copySize = uintptr(len(graphPayloadData)) - uintptr(modelNode.In)
			}
			copy(sublatePtr.PayloadPrev[:copySize], graphPayloadData[modelNode.In:modelNode.In+uint16(copySize)])
		}
	}

	// Allocate PayloadProp for outputs
	propSize := defaultPayloadPropSize
	if propSize > 0 {
		// propBuf, err := arena.AllocateScratch(propSize, 8) // Old
		propBuf, err := arena.AllocateNodePayload(propSize, core.CacheLineSize) // Changed
		if err != nil {
			return fmt.Errorf("failed to allocate PayloadProp: %w", err)
		}
		sublatePtr.PayloadProp = propBuf
		// Initialize to zero
		for i := range sublatePtr.PayloadProp {
			sublatePtr.PayloadProp[i] = 0
		}
	}

	return nil
}

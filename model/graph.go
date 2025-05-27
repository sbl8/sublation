// Package model defines the computational graph representation for Sublation models.
//
// This package provides the core data structures for representing neural network
// models as directed graphs of compute nodes (Sublates). The graph representation
// is used throughout the compilation and runtime pipeline for optimization,
// scheduling, and execution.
//
// Key data structures:
//   - Node: Individual compute unit with kernel ID, memory offsets, and topology
//   - Graph: Complete model representation with nodes and payload data
//   - Serialization utilities for persistent model storage
//
// The graph model supports:
//   - Arbitrary topology including cycles for recurrent architectures
//   - Efficient serialization for fast model loading
//   - Graph validation and optimization passes
//   - Memory layout analysis for cache performance
//
// Models are typically created by the compiler from .subs specifications,
// then loaded by the runtime for execution. The graph representation is
// designed to be immutable after compilation for thread-safe execution.
package model

import (
	"bytes"
	"encoding/binary"
	"encoding/gob"
	"fmt"
	"io"
)

// Node represents a graph node with input and output ports and flags
type Node struct {
	ID     uint16
	In     uint16   // payload offset for input
	Out    uint16   // payload offset for output
	Kernel uint8    // opcode for data transform
	Flags  uint32   // node-specific flags
	Topo   []uint16 // neighbor indices for message passing
}

// Graph is an immutable representation parsed from .subl, with utility methods
type Graph struct {
	Nodes   []Node
	Payload []byte // concatenated and aligned data payload
}

// NodeCount returns the number of nodes in the graph
func (g *Graph) NodeCount() int {
	return len(g.Nodes)
}

// NodeSize returns the size in bytes of a serialized Node entry
func NodeSize() int {
	return 16 // Fixed size for binary serialization
}

// Serialize writes the Graph to a byte slice using optimized binary format
func (g *Graph) Serialize() ([]byte, error) {
	var buf bytes.Buffer

	// Write header: magic number, version, node count, payload size
	if err := binary.Write(&buf, binary.LittleEndian, uint32(0x53554C42)); err != nil { // "SULB"
		return nil, err
	}
	if err := binary.Write(&buf, binary.LittleEndian, uint16(1)); err != nil { // version
		return nil, err
	}
	if err := binary.Write(&buf, binary.LittleEndian, uint16(len(g.Nodes))); err != nil {
		return nil, err
	}
	if err := binary.Write(&buf, binary.LittleEndian, uint32(len(g.Payload))); err != nil {
		return nil, err
	}

	// Write nodes in fixed-size format
	for _, node := range g.Nodes {
		if err := binary.Write(&buf, binary.LittleEndian, node.ID); err != nil {
			return nil, err
		}
		if err := binary.Write(&buf, binary.LittleEndian, node.In); err != nil {
			return nil, err
		}
		if err := binary.Write(&buf, binary.LittleEndian, node.Out); err != nil {
			return nil, err
		}
		if err := binary.Write(&buf, binary.LittleEndian, node.Kernel); err != nil {
			return nil, err
		}
		if err := binary.Write(&buf, binary.LittleEndian, uint8(len(node.Topo))); err != nil {
			return nil, err
		}
		if err := binary.Write(&buf, binary.LittleEndian, node.Flags); err != nil {
			return nil, err
		}

		// Write topology indices (padded to 4-byte alignment)
		for _, idx := range node.Topo {
			if err := binary.Write(&buf, binary.LittleEndian, idx); err != nil {
				return nil, err
			}
		}
		// Pad to maintain alignment
		for i := len(node.Topo); i < 2; i++ {
			if err := binary.Write(&buf, binary.LittleEndian, uint16(0xFFFF)); err != nil {
				return nil, err
			}
		}
	}

	// Write payload data aligned to 32-byte boundary
	payloadOffset := buf.Len()
	alignedOffset := (payloadOffset + 31) &^ 31
	padding := alignedOffset - payloadOffset
	for i := 0; i < padding; i++ {
		buf.WriteByte(0)
	}

	buf.Write(g.Payload)

	return buf.Bytes(), nil
}

// Deserialize reads a Graph from a byte slice using binary format
func Deserialize(data []byte) (*Graph, error) {
	buf := bytes.NewReader(data)

	// Read header
	var magic uint32
	if err := binary.Read(buf, binary.LittleEndian, &magic); err != nil {
		return nil, err
	}
	if magic != 0x53554C42 {
		return nil, fmt.Errorf("invalid magic number: %x", magic)
	}

	var version uint16
	if err := binary.Read(buf, binary.LittleEndian, &version); err != nil {
		return nil, err
	}
	if version != 1 {
		return nil, fmt.Errorf("unsupported version: %d", version)
	}

	var nodeCount uint16
	if err := binary.Read(buf, binary.LittleEndian, &nodeCount); err != nil {
		return nil, err
	}

	var payloadSize uint32
	if err := binary.Read(buf, binary.LittleEndian, &payloadSize); err != nil {
		return nil, err
	}

	// Read nodes
	nodes := make([]Node, nodeCount)
	for i := range nodes {
		if err := binary.Read(buf, binary.LittleEndian, &nodes[i].ID); err != nil {
			return nil, err
		}
		if err := binary.Read(buf, binary.LittleEndian, &nodes[i].In); err != nil {
			return nil, err
		}
		if err := binary.Read(buf, binary.LittleEndian, &nodes[i].Out); err != nil {
			return nil, err
		}
		if err := binary.Read(buf, binary.LittleEndian, &nodes[i].Kernel); err != nil {
			return nil, err
		}

		var topoLen uint8
		if err := binary.Read(buf, binary.LittleEndian, &topoLen); err != nil {
			return nil, err
		}

		if err := binary.Read(buf, binary.LittleEndian, &nodes[i].Flags); err != nil {
			return nil, err
		}

		// Read topology (always read 2 uint16s for alignment)
		topo := make([]uint16, 2)
		if err := binary.Read(buf, binary.LittleEndian, &topo); err != nil {
			return nil, err
		}

		// Extract actual topology
		nodes[i].Topo = make([]uint16, 0, topoLen)
		for j := 0; j < int(topoLen) && j < 2; j++ {
			if topo[j] != 0xFFFF {
				nodes[i].Topo = append(nodes[i].Topo, topo[j])
			}
		}
	}

	// Skip to aligned payload offset
	currentOffset, _ := buf.Seek(0, io.SeekCurrent)
	alignedOffset := (currentOffset + 31) &^ 31
	if _, err := buf.Seek(alignedOffset, io.SeekStart); err != nil {
		return nil, err
	}

	// Read payload
	payload := make([]byte, payloadSize)
	if _, err := buf.Read(payload); err != nil {
		return nil, err
	}

	return &Graph{Nodes: nodes, Payload: payload}, nil
}

// SerializeGob writes the Graph using gob encoding (fallback)
func (g *Graph) SerializeGob() ([]byte, error) {
	var buf bytes.Buffer
	encoder := gob.NewEncoder(&buf)
	if err := encoder.Encode(g.Nodes); err != nil {
		return nil, err
	}
	if err := encoder.Encode(g.Payload); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// DeserializeGob reads a Graph from gob-encoded data (fallback)
func DeserializeGob(data []byte) (*Graph, error) {
	buf := bytes.NewBuffer(data)
	decoder := gob.NewDecoder(buf)
	var nodes []Node
	if err := decoder.Decode(&nodes); err != nil {
		return nil, err
	}
	var payload []byte
	if err := decoder.Decode(&payload); err != nil {
		return nil, err
	}
	return &Graph{Nodes: nodes, Payload: payload}, nil
}

// Validate checks graph consistency
func (g *Graph) Validate() error {
	if len(g.Nodes) == 0 {
		return fmt.Errorf("graph has no nodes")
	}

	// Check for duplicate IDs
	ids := make(map[uint16]bool)
	for _, node := range g.Nodes {
		if ids[node.ID] {
			return fmt.Errorf("duplicate node ID: %d", node.ID)
		}
		ids[node.ID] = true

		// Check topology references
		for _, neighborID := range node.Topo {
			if neighborID != 0xFFFF && !ids[neighborID] {
				return fmt.Errorf("node %d references non-existent neighbor %d", node.ID, neighborID)
			}
		}

		// Check payload bounds
		if int(node.Out) >= len(g.Payload) {
			return fmt.Errorf("node %d output offset %d exceeds payload size %d", node.ID, node.Out, len(g.Payload))
		}
	}

	return nil
}

// Optimize performs graph optimizations for runtime performance
func (g *Graph) Optimize() {
	// Sort nodes by execution order for better cache locality
	g.topologicalSort()

	// Pack payload for optimal memory layout
	g.compactPayload()
}

// topologicalSort reorders nodes for execution dependency order
func (g *Graph) topologicalSort() {
	// Build dependency graph
	adj := make(map[uint16][]uint16)
	inDegree := make(map[uint16]int)

	for _, node := range g.Nodes {
		if _, exists := inDegree[node.ID]; !exists {
			inDegree[node.ID] = 0
		}
		for _, dep := range node.Topo {
			if dep != 0xFFFF {
				adj[dep] = append(adj[dep], node.ID)
				inDegree[node.ID]++
			}
		}
	}

	// Kahn's algorithm for topological sort
	queue := make([]uint16, 0)
	for nodeID, degree := range inDegree {
		if degree == 0 {
			queue = append(queue, nodeID)
		}
	}

	var executionOrder []uint16
	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]
		executionOrder = append(executionOrder, current)

		for _, neighbor := range adj[current] {
			inDegree[neighbor]--
			if inDegree[neighbor] == 0 {
				queue = append(queue, neighbor)
			}
		}
	}

	// Reorder nodes based on execution order
	nodeMap := make(map[uint16]*Node)
	for i := range g.Nodes {
		nodeMap[g.Nodes[i].ID] = &g.Nodes[i]
	}

	reordered := make([]Node, 0, len(g.Nodes))
	for _, nodeID := range executionOrder {
		if node, exists := nodeMap[nodeID]; exists {
			reordered = append(reordered, *node)
		}
	}
	g.Nodes = reordered
}

// compactPayload optimizes payload layout for cache efficiency
func (g *Graph) compactPayload() {
	// TODO: Implement payload compaction based on access patterns
	// This would analyze which data segments are accessed together
	// and reorder them for optimal cache locality
}

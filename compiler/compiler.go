// Package compiler transforms Sublation model specifications into optimized binary format.
//
// This package implements the Sublation compiler (sublc) that converts human-readable
// .subs model specifications into efficient .subl binary files for runtime execution.
// The compilation process includes parsing, validation, optimization, and binary emission.
//
// Compilation pipeline:
//  1. Parse .subs DSL into internal graph representation
//  2. Validate graph structure and detect cycles
//  3. Apply layout optimizations for cache locality
//  4. Emit cache-aligned binary format with metadata
//
// Supported optimizations:
//   - Topological reordering for execution efficiency
//   - Memory layout optimization for cache performance
//   - Dead code elimination and kernel fusion analysis
//   - Payload compaction and alignment
//
// The compiler produces self-contained .subl files that include all model data,
// topology information, and execution metadata required by the runtime engine.
//
// DSL features:
//   - Node declarations with kernel opcodes and memory offsets
//   - Hexadecimal payload data for weights and parameters
//   - Iteration constructs for batch processing
//   - Flexible topology specification for complex architectures
package compiler

import (
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/sbl8/sublation/core"
	"github.com/sbl8/sublation/model"
)

// Compile turns a .subs text spec into a binary .subl file.
func Compile(src, out string) error {
	g, err := loadAndParseSpec(src)
	if err != nil {
		return err
	}

	return writeSimpleGraph(&g, out)
}

// loadAndParseSpec reads and parses a source file
func loadAndParseSpec(src string) (model.Graph, error) {
	spec, err := os.ReadFile(src)
	if err != nil {
		return model.Graph{}, err
	}

	return parseSpec(spec)
}

// writeSimpleGraph writes a graph in the simple binary format
func writeSimpleGraph(g *model.Graph, out string) error {
	f, err := os.Create(out)
	if err != nil {
		return err
	}
	defer f.Close()

	writer := &simpleWriter{f: f}
	return writer.writeGraph(g)
}

// simpleWriter handles the original simple binary format
type simpleWriter struct {
	f *os.File
}

// writeGraph writes graph in simple format
func (w *simpleWriter) writeGraph(g *model.Graph) error {
	if err := w.writeSimpleHeader(g); err != nil {
		return err
	}

	if err := w.writeSimpleNodes(g.Nodes); err != nil {
		return err
	}

	return w.writeSimplePayload(g.Payload)
}

// writeSimpleHeader writes the simple format header
func (w *simpleWriter) writeSimpleHeader(g *model.Graph) error {
	// Header: node count (uint32), payload length (uint32)
	headers := []uint32{
		uint32(len(g.Nodes)),
		uint32(len(g.Payload)),
	}

	for _, header := range headers {
		if err := binary.Write(w.f, binary.LittleEndian, header); err != nil {
			return err
		}
	}

	return nil
}

// writeSimpleNodes writes nodes in simple format
func (w *simpleWriter) writeSimpleNodes(nodes []model.Node) error {
	for _, node := range nodes {
		if err := w.writeSimpleNode(node); err != nil {
			return err
		}
	}
	return nil
}

// writeSimpleNode writes a single node in simple format
func (w *simpleWriter) writeSimpleNode(node model.Node) error {
	// Write basic fields
	fields := []interface{}{
		node.ID,
		node.Kernel,
		node.In,
		node.Out,
		node.Flags,
	}

	for _, field := range fields {
		if err := binary.Write(w.f, binary.LittleEndian, field); err != nil {
			return err
		}
	}

	// Apply padding to NodeSize
	return w.writeSimpleNodePadding()
}

// writeSimpleNodePadding pads node to fixed size
func (w *simpleWriter) writeSimpleNodePadding() error {
	pad := model.NodeSize() - (2 + 1 + 2 + 2 + 4) // ID+Kernel+In+Out+Flags
	if pad > 0 {
		padBytes := make([]byte, pad)
		_, err := w.f.Write(padBytes)
		return err
	}
	return nil
}

// writeSimplePayload writes aligned payload
func (w *simpleWriter) writeSimplePayload(payload []byte) error {
	pad := core.Align32(len(payload)) - len(payload)

	if _, err := w.f.Write(payload); err != nil {
		return err
	}

	if pad > 0 {
		padBytes := make([]byte, pad)
		_, err := w.f.Write(padBytes)
		return err
	}

	return nil
}

// --- DSL parser with support for node, payload, and iterate blocks ---
// parseSpec parses the DSL and returns a Graph or an error on invalid syntax
func parseSpec(src []byte) (model.Graph, error) {
	lines := strings.Split(string(src), "\n")
	var nodes []model.Node
	var payload []byte

	parser := &dslParser{nodes: &nodes, payload: &payload}

	for i := 0; i < len(lines); i++ {
		line := strings.TrimSpace(lines[i])
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		var err error
		i, err = parser.parseLine(lines, i)
		if err != nil {
			return model.Graph{}, fmt.Errorf("line %d: %v", i+1, err)
		}
	}

	// align payload
	payload = alignPayload(payload)
	return model.Graph{Nodes: nodes, Payload: payload}, nil
}

// dslParser handles DSL parsing state
type dslParser struct {
	nodes   *[]model.Node
	payload *[]byte
}

// parseLine processes a single line and returns the next line index
func (p *dslParser) parseLine(lines []string, idx int) (int, error) {
	line := strings.TrimSpace(lines[idx])
	fields := strings.Fields(line)

	switch fields[0] {
	case "iterate":
		return p.parseIterateBlock(lines, idx, fields)
	default:
		return idx, p.processSimpleLine(line, fields)
	}
}

// parseIterateBlock handles iterate constructs
func (p *dslParser) parseIterateBlock(lines []string, idx int, fields []string) (int, error) {
	if len(fields) < 4 {
		return idx, fmt.Errorf("invalid iterate spec: %s", strings.Join(fields, " "))
	}

	varName, start, end, err := parseIterateParams(fields)
	if err != nil {
		return idx, err
	}

	// Find opening brace and collect block
	blockStart := idx
	if !strings.HasSuffix(strings.Join(fields, " "), "{") {
		blockStart++
		for blockStart < len(lines) && strings.TrimSpace(lines[blockStart]) == "" {
			blockStart++
		}
		if blockStart >= len(lines) || strings.TrimSpace(lines[blockStart]) != "{" {
			return idx, fmt.Errorf("missing '{' after iterate")
		}
	}

	block, blockEnd, err := collectBlockLines(lines, blockStart)
	if err != nil {
		return idx, err
	}

	// Expand and process block
	if err := p.expandIterateBlock(block, varName, start, end); err != nil {
		return idx, err
	}

	return blockEnd, nil
}

// processSimpleLine handles node and payload directives
func (p *dslParser) processSimpleLine(line string, fields []string) error {
	switch fields[0] {
	case "node":
		return p.parseNodeLine(fields)
	case "payload":
		return p.parsePayloadLine(fields)
	default:
		return fmt.Errorf("unknown directive: %s", fields[0])
	}
}

// parseNodeLine parses a node directive
func (p *dslParser) parseNodeLine(fields []string) error {
	if len(fields) < 5 {
		return fmt.Errorf("invalid node spec: needs at least 5 fields")
	}

	node, err := parseNodeFields(fields)
	if err != nil {
		return err
	}

	*p.nodes = append(*p.nodes, node)
	return nil
}

// parsePayloadLine parses a payload directive
func (p *dslParser) parsePayloadLine(fields []string) error {
	if len(fields) < 2 {
		return fmt.Errorf("invalid payload spec: missing data")
	}

	data, err := parsePayloadData(fields[1])
	if err != nil {
		return err
	}

	*p.payload = append(*p.payload, data...)
	return nil
}

// parseIterateParams extracts iterate parameters
func parseIterateParams(fields []string) (varName string, start, end int, err error) {
	varName = fields[1]
	start, err = strconv.Atoi(fields[2])
	if err != nil {
		return "", 0, 0, fmt.Errorf("invalid iterate start %q: %v", fields[2], err)
	}
	end, err = strconv.Atoi(fields[3])
	if err != nil {
		return "", 0, 0, fmt.Errorf("invalid iterate end %q: %v", fields[3], err)
	}
	return varName, start, end, nil
}

// collectBlockLines gathers lines within braces
func collectBlockLines(lines []string, startIdx int) ([]string, int, error) {
	var block []string
	i := startIdx + 1

	for i < len(lines) {
		line := strings.TrimSpace(lines[i])
		if line == "}" {
			return block, i, nil
		}
		if line != "" && !strings.HasPrefix(line, "#") {
			block = append(block, line)
		}
		i++
	}

	return nil, i, fmt.Errorf("unterminated iterate block")
}

// expandIterateBlock processes iterate expansion
func (p *dslParser) expandIterateBlock(block []string, varName string, start, end int) error {
	for v := start; v <= end; v++ {
		for _, line := range block {
			expanded := expandVariable(line, varName, v)
			fields := strings.Fields(expanded)
			if err := p.processSimpleLine(expanded, fields); err != nil {
				return fmt.Errorf("iterate expansion error: %v", err)
			}
		}
	}
	return nil
}

// expandVariable replaces variable with value in line
func expandVariable(line, varName string, value int) string {
	fields := strings.Fields(line)
	for i, field := range fields {
		if field == varName {
			fields[i] = strconv.Itoa(value)
		}
	}
	return strings.Join(fields, " ")
}

// parseNodeFields extracts node from field tokens
func parseNodeFields(fields []string) (model.Node, error) {
	id, err := strconv.Atoi(fields[1])
	if err != nil {
		return model.Node{}, fmt.Errorf("invalid node id %q: %v", fields[1], err)
	}
	kernel, err := strconv.ParseUint(fields[2], 0, 8)
	if err != nil {
		return model.Node{}, fmt.Errorf("invalid kernel %q: %v", fields[2], err)
	}
	in, err := strconv.ParseUint(fields[3], 0, 16)
	if err != nil {
		return model.Node{}, fmt.Errorf("invalid in %q: %v", fields[3], err)
	}
	out, err := strconv.ParseUint(fields[4], 0, 16)
	if err != nil {
		return model.Node{}, fmt.Errorf("invalid out %q: %v", fields[4], err)
	}

	var flags uint32
	if len(fields) > 5 {
		f, err := strconv.ParseUint(fields[5], 0, 32)
		if err != nil {
			return model.Node{}, fmt.Errorf("invalid flags %q: %v", fields[5], err)
		}
		flags = uint32(f)
	}

	return model.Node{
		ID:     uint16(id),
		Kernel: uint8(kernel),
		In:     uint16(in),
		Out:    uint16(out),
		Flags:  flags,
	}, nil
}

// parsePayloadData decodes hex or literal payload data
func parsePayloadData(data string) ([]byte, error) {
	// Try hex decode first
	if decoded, err := hex.DecodeString(data); err == nil {
		return decoded, nil
	}
	// Fallback to raw literal
	return []byte(data), nil
}

// alignPayload pads payload to 32-byte alignment
func alignPayload(payload []byte) []byte {
	pad := core.Align32(len(payload)) - len(payload)
	if pad > 0 {
		payload = append(payload, make([]byte, pad)...)
	}
	return payload
}

// CompileOptions configures the compilation process
type CompileOptions struct {
	OptimizeLayout bool // Reorder nodes for cache efficiency
	ValidateGraph  bool // Check for cycles, unreachable nodes
	DebugOutput    bool // Include debug symbols
	Verbose        bool // Enable verbose output
}

// DefaultOptions provides sensible compilation defaults
func DefaultOptions() CompileOptions {
	return CompileOptions{
		OptimizeLayout: true,
		ValidateGraph:  true,
		DebugOutput:    false,
		Verbose:        false,
	}
}

// CompileWithOptions provides advanced compilation features
func CompileWithOptions(src, out string, opts CompileOptions) error {
	if opts.Verbose {
		fmt.Printf("Compiling %s -> %s\n", src, out)
	}

	spec, err := os.ReadFile(src)
	if err != nil {
		return fmt.Errorf("failed to read source: %w", err)
	}

	// Parse the specification
	g, err := parseSpec(spec)
	if err != nil {
		return fmt.Errorf("parse error: %w", err)
	}

	if opts.Verbose {
		fmt.Printf("Parsed %d nodes with %d bytes payload\n", len(g.Nodes), len(g.Payload))
	}

	// Validate graph structure
	if opts.ValidateGraph {
		if err := validateGraph(&g); err != nil {
			return fmt.Errorf("validation error: %w", err)
		}
		if opts.Verbose {
			fmt.Println("Graph validation passed")
		}
	}

	// Optimize node layout
	if opts.OptimizeLayout {
		optimizeNodeLayout(&g)
		if opts.Verbose {
			fmt.Println("Applied layout optimizations")
		}
	}

	// Write output file
	if err := writeCompiledGraph(&g, out, opts); err != nil {
		return fmt.Errorf("failed to write output: %w", err)
	}

	if opts.Verbose {
		fmt.Printf("Successfully compiled to %s\n", out)
	}

	return nil
}

// validateGraph checks for common graph issues
func validateGraph(g *model.Graph) error {
	if len(g.Nodes) == 0 {
		return fmt.Errorf("empty graph")
	}

	// Check for duplicate node IDs
	seen := make(map[uint16]bool)
	for i, node := range g.Nodes {
		if seen[node.ID] {
			return fmt.Errorf("duplicate node ID %d at index %d", node.ID, i)
		}
		seen[node.ID] = true

		// Check payload bounds
		if node.In >= uint16(len(g.Payload)) {
			return fmt.Errorf("node %d input offset %d exceeds payload size %d", node.ID, node.In, len(g.Payload))
		}
		if node.Out >= uint16(len(g.Payload)) {
			return fmt.Errorf("node %d output offset %d exceeds payload size %d", node.ID, node.Out, len(g.Payload))
		}

		// Check topology references
		for _, ref := range node.Topo {
			if !seen[ref] && ref != 0xFFFF { // 0xFFFF is sentinel for unused
				fmt.Printf("Warning: node %d references undefined node %d\n", node.ID, ref)
			}
		}
	}

	// Check for cycles (simplified DFS-based detection)
	return detectCycles(g)
}

// detectCycles performs topological sort to detect cycles
func detectCycles(g *model.Graph) error {
	// Build adjacency list
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

	// Kahn's algorithm
	queue := make([]uint16, 0)
	for nodeID, degree := range inDegree {
		if degree == 0 {
			queue = append(queue, nodeID)
		}
	}

	processed := 0
	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]
		processed++

		for _, neighbor := range adj[current] {
			inDegree[neighbor]--
			if inDegree[neighbor] == 0 {
				queue = append(queue, neighbor)
			}
		}
	}

	if processed != len(g.Nodes) {
		return fmt.Errorf("cycle detected in graph")
	}

	return nil
}

// optimizeNodeLayout reorders nodes for better cache locality
func optimizeNodeLayout(g *model.Graph) {
	// Simple optimization: sort nodes by execution order based on dependencies
	// This puts dependent nodes closer together in memory

	// Build execution order using topological sort
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

	// Execute topological sort
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

	// Reorder nodes according to execution order
	nodeMap := make(map[uint16]model.Node)
	for _, node := range g.Nodes {
		nodeMap[node.ID] = node
	}

	newNodes := make([]model.Node, 0, len(g.Nodes))
	for _, nodeID := range executionOrder {
		if node, exists := nodeMap[nodeID]; exists {
			newNodes = append(newNodes, node)
			delete(nodeMap, nodeID)
		}
	}

	// Append any remaining nodes (shouldn't happen if graph is valid)
	for _, node := range nodeMap {
		newNodes = append(newNodes, node)
	}

	g.Nodes = newNodes
}

// writeCompiledGraph writes the optimized graph to a binary file
func writeCompiledGraph(g *model.Graph, output string, opts CompileOptions) error {
	f, err := os.Create(output)
	if err != nil {
		return err
	}
	defer f.Close()

	writer := &binaryWriter{f: f}

	if err := writer.writeHeader(g, opts); err != nil {
		return err
	}

	if err := writer.writeNodes(g.Nodes); err != nil {
		return err
	}

	return writer.writePayload(g.Payload)
}

// binaryWriter handles binary file output
type binaryWriter struct {
	f *os.File
}

// writeHeader writes file version and metadata
func (w *binaryWriter) writeHeader(g *model.Graph, opts CompileOptions) error {
	// File format version
	if err := binary.Write(w.f, binary.LittleEndian, uint32(1)); err != nil {
		return err
	}

	// Compute flags
	flags := uint32(0)
	if opts.DebugOutput {
		flags |= 0x01
	}

	// Write header fields
	headers := []uint32{
		uint32(len(g.Nodes)),
		uint32(len(g.Payload)),
		flags,
	}

	for _, header := range headers {
		if err := binary.Write(w.f, binary.LittleEndian, header); err != nil {
			return err
		}
	}

	return nil
}

// writeNodes writes all nodes with proper alignment
func (w *binaryWriter) writeNodes(nodes []model.Node) error {
	for _, node := range nodes {
		if err := w.writeNode(node); err != nil {
			return err
		}
	}
	return nil
}

// writeNode writes a single node with alignment
func (w *binaryWriter) writeNode(node model.Node) error {
	// Write basic fields
	if err := w.writeNodeFields(node); err != nil {
		return err
	}

	// Write topology
	if err := w.writeNodeTopology(node.Topo); err != nil {
		return err
	}

	// Apply padding for alignment
	return w.writeNodePadding(node)
}

// writeNodeFields writes basic node fields
func (w *binaryWriter) writeNodeFields(node model.Node) error {
	fields := []interface{}{
		node.ID,
		node.Kernel,
		node.In,
		node.Out,
		node.Flags,
	}

	for _, field := range fields {
		if err := binary.Write(w.f, binary.LittleEndian, field); err != nil {
			return err
		}
	}

	return nil
}

// writeNodeTopology writes topology data with length prefix
func (w *binaryWriter) writeNodeTopology(topo []uint16) error {
	// Write length prefix
	if err := binary.Write(w.f, binary.LittleEndian, uint16(len(topo))); err != nil {
		return err
	}

	// Write topology entries
	for _, entry := range topo {
		if err := binary.Write(w.f, binary.LittleEndian, entry); err != nil {
			return err
		}
	}

	return nil
}

// writeNodePadding applies alignment padding
func (w *binaryWriter) writeNodePadding(node model.Node) error {
	baseSize := 16 + 2 + len(node.Topo)*2 // ID+Kernel+In+Out+Flags+TopoLen+Topo
	padding := core.AlignSize(baseSize, 8) - baseSize

	if padding > 0 {
		padBytes := make([]byte, padding)
		_, err := w.f.Write(padBytes)
		return err
	}

	return nil
}

// writePayload writes aligned payload data
func (w *binaryWriter) writePayload(payload []byte) error {
	alignedPayload := core.PadToAlignment(payload, 32)
	_, err := w.f.Write(alignedPayload)
	return err
}

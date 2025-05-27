package core

import (
	"bytes"
	"encoding/binary"
	"errors"
)

// SerializeSublate writes a Sublate to a byte slice in binary form.
// Layout: [KernelID(1)][Flags(4)][len(Topology)(2)][Topology elems(2*len)][len(PayloadPrev)(4)][PayloadPrev bytes][len(PayloadProp)(4)][PayloadProp bytes]
func SerializeSublate(s *Sublate) ([]byte, error) {
	buf := &bytes.Buffer{}

	// KernelID
	if err := buf.WriteByte(s.KernelID); err != nil {
		return nil, err
	}

	// Flags
	if err := binary.Write(buf, binary.LittleEndian, s.Flags); err != nil {
		return nil, err
	}

	// Topology length
	topoLen := uint16(len(s.Topology))
	if err := binary.Write(buf, binary.LittleEndian, topoLen); err != nil {
		return nil, err
	}

	// Topology elements
	for _, idx := range s.Topology {
		if err := binary.Write(buf, binary.LittleEndian, idx); err != nil {
			return nil, err
		}
	}

	// PayloadPrev length and data
	prevLen := uint32(len(s.PayloadPrev))
	if err := binary.Write(buf, binary.LittleEndian, prevLen); err != nil {
		return nil, err
	}
	if prevLen > 0 {
		if n, err := buf.Write(s.PayloadPrev); err != nil || n != int(prevLen) {
			return nil, errors.New("failed to write PayloadPrev")
		}
	}

	// PayloadProp length and data
	propLen := uint32(len(s.PayloadProp))
	if err := binary.Write(buf, binary.LittleEndian, propLen); err != nil {
		return nil, err
	}
	if propLen > 0 {
		if n, err := buf.Write(s.PayloadProp); err != nil || n != int(propLen) {
			return nil, errors.New("failed to write PayloadProp")
		}
	}

	return buf.Bytes(), nil
}

// DeserializeSublate reads a Sublate from a byte slice.
func DeserializeSublate(b []byte) (*Sublate, error) {
	buf := bytes.NewReader(b)
	s := &Sublate{}

	// KernelID
	kernelID, err := buf.ReadByte()
	if err != nil {
		return nil, err
	}
	s.KernelID = kernelID

	// Flags
	if err := binary.Read(buf, binary.LittleEndian, &s.Flags); err != nil {
		return nil, err
	}

	// Topology length
	var topoLen uint16
	if err := binary.Read(buf, binary.LittleEndian, &topoLen); err != nil {
		return nil, err
	}

	// Topology elements
	s.Topology = make([]uint16, topoLen)
	for i := uint16(0); i < topoLen; i++ {
		if err := binary.Read(buf, binary.LittleEndian, &s.Topology[i]); err != nil {
			return nil, err
		}
	}

	// PayloadPrev length and data
	var prevLen uint32
	if err := binary.Read(buf, binary.LittleEndian, &prevLen); err != nil {
		return nil, err
	}
	if prevLen > 0 {
		s.PayloadPrev = make([]byte, prevLen)
		if n, err := buf.Read(s.PayloadPrev); err != nil || n != int(prevLen) {
			return nil, errors.New("failed to read PayloadPrev")
		}
	}

	// PayloadProp length and data
	var propLen uint32
	if err := binary.Read(buf, binary.LittleEndian, &propLen); err != nil {
		return nil, err
	}
	if propLen > 0 {
		s.PayloadProp = make([]byte, propLen)
		if n, err := buf.Read(s.PayloadProp); err != nil || n != int(propLen) {
			return nil, errors.New("failed to read PayloadProp")
		}
	}

	return s, nil
}

// BatchSerializeSublates serializes multiple sublates with optimal memory layout
func BatchSerializeSublates(sublates []*Sublate) ([]byte, error) {
	if len(sublates) == 0 {
		return nil, nil
	}

	// Pre-calculate total size for single allocation
	totalSize := 0
	for _, s := range sublates {
		totalSize += 1 + 4 + 2 + len(s.Topology)*2 + 4 + len(s.PayloadPrev) + 4 + len(s.PayloadProp)
	}

	buf := make([]byte, 0, totalSize)
	buffer := bytes.NewBuffer(buf)

	for _, s := range sublates {
		data, err := SerializeSublate(s)
		if err != nil {
			return nil, err
		}
		buffer.Write(data)
	}
	return buffer.Bytes(), nil
}

// SerializationHeader provides metadata for serialized data
type SerializationHeader struct {
	Magic    uint32 // "SUBL" magic number
	Version  uint16 // format version
	Count    uint32 // number of sublates
	Checksum uint32 // data integrity checksum
	Reserved uint32 // padding for future use
}

const (
	SerializationMagic   = 0x4C425553 // "SUBL" in little endian
	SerializationVersion = 1
	HeaderSize           = 20 // sizeof(SerializationHeader)
)

// SerializeWithHeader creates a complete serialized format with integrity checking
func SerializeWithHeader(sublates []*Sublate) ([]byte, error) {
	// Serialize sublates first to calculate checksum
	sublateData, err := BatchSerializeSublates(sublates)
	if err != nil {
		return nil, err
	}

	// Create header
	header := SerializationHeader{
		Magic:    SerializationMagic,
		Version:  SerializationVersion,
		Count:    uint32(len(sublates)),
		Checksum: crc32Checksum(sublateData),
		Reserved: 0,
	}

	// Calculate total size
	totalSize := HeaderSize + len(sublateData)
	buf := make([]byte, 0, totalSize)
	buffer := bytes.NewBuffer(buf)

	// Write header
	if err := binary.Write(buffer, binary.LittleEndian, header); err != nil {
		return nil, err
	}

	// Write sublate data
	buffer.Write(sublateData)

	return buffer.Bytes(), nil
}

// DeserializeWithHeader reads a complete serialized format with integrity checking
func DeserializeWithHeader(data []byte) ([]*Sublate, error) {
	if len(data) < HeaderSize {
		return nil, errors.New("data too short for header")
	}

	// Read header
	buf := bytes.NewReader(data)
	var header SerializationHeader
	if err := binary.Read(buf, binary.LittleEndian, &header); err != nil {
		return nil, err
	}

	if header.Magic != SerializationMagic {
		return nil, errors.New("invalid magic number")
	}

	if header.Version != SerializationVersion {
		return nil, errors.New("unsupported serialization version")
	}

	sublateData := data[HeaderSize:]

	// Verify checksum
	if crc32Checksum(sublateData) != header.Checksum {
		return nil, errors.New("data corruption detected")
	}

	return BatchDeserializeSublates(sublateData, int(header.Count))
}

// BatchDeserializeSublates deserializes multiple sublates from a byte slice
func BatchDeserializeSublates(data []byte, count int) ([]*Sublate, error) {
	if len(data) == 0 || count == 0 {
		return nil, nil
	}

	sublates := make([]*Sublate, 0, count)
	buf := bytes.NewReader(data)

	for i := 0; i < count && buf.Len() > 0; i++ {
		// Calculate sublate size by reading headers
		currentPos := int64(len(data)) - int64(buf.Len())
		tempBuf := bytes.NewReader(data[currentPos:])

		// Skip KernelID and Flags
		if _, err := tempBuf.Seek(5, 0); err != nil {
			return nil, err
		}

		// Read topology length
		var topoLen uint16
		if err := binary.Read(tempBuf, binary.LittleEndian, &topoLen); err != nil {
			return nil, err
		}

		// Skip topology data
		if _, err := tempBuf.Seek(int64(topoLen)*2, 1); err != nil {
			return nil, err
		}

		// Read PayloadPrev length
		var prevLen uint32
		if err := binary.Read(tempBuf, binary.LittleEndian, &prevLen); err != nil {
			return nil, err
		}

		// Skip PayloadPrev data
		if _, err := tempBuf.Seek(int64(prevLen), 1); err != nil {
			return nil, err
		}

		// Read PayloadProp length
		var propLen uint32
		if err := binary.Read(tempBuf, binary.LittleEndian, &propLen); err != nil {
			return nil, err
		}

		// Calculate total sublate size
		sublateSize := 1 + 4 + 2 + int(topoLen)*2 + 4 + int(prevLen) + 4 + int(propLen)

		// Read the complete sublate
		sublateData := make([]byte, sublateSize)
		n, err := buf.Read(sublateData)
		if err != nil || n != sublateSize {
			return nil, errors.New("failed to read complete sublate")
		}

		s, err := DeserializeSublate(sublateData)
		if err != nil {
			return nil, err
		}
		sublates = append(sublates, s)
	}

	return sublates, nil
}

// Simple CRC32 checksum for integrity verification
func crc32Checksum(data []byte) uint32 {
	const poly = 0xEDB88320 // IEEE CRC32 polynomial
	crc := uint32(0xFFFFFFFF)

	for _, b := range data {
		crc ^= uint32(b)
		for i := 0; i < 8; i++ {
			if crc&1 != 0 {
				crc = (crc >> 1) ^ poly
			} else {
				crc >>= 1
			}
		}
	}

	return ^crc
}

// MemoryLayout provides detailed memory usage analysis
type MemoryLayout struct {
	TotalSize     int
	HeaderSize    int
	PayloadSize   int
	TopologySize  int
	Alignment     int
	Fragmentation float64
}

// AnalyzeMemoryLayout provides detailed memory usage statistics
func AnalyzeMemoryLayout(sublates []*Sublate) MemoryLayout {
	layout := MemoryLayout{
		HeaderSize: HeaderSize,
		Alignment:  32, // Cache line alignment
	}

	for _, s := range sublates {
		layout.TotalSize += 1 + 4 + 2 + len(s.Topology)*2 + 4 + len(s.PayloadPrev) + 4 + len(s.PayloadProp)
		layout.PayloadSize += len(s.PayloadPrev) + len(s.PayloadProp)
		layout.TopologySize += len(s.Topology) * 2
	}

	// Calculate fragmentation as percentage of overhead
	overhead := layout.TotalSize - layout.PayloadSize
	if layout.TotalSize > 0 {
		layout.Fragmentation = float64(overhead) / float64(layout.TotalSize) * 100.0
	}

	return layout
}

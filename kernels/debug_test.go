package kernels

import (
	"encoding/binary"
	"fmt"
	"testing"
	"unsafe"
)

func TestDataInterpretation(t *testing.T) {
	// Create test data using proper encoding
	data := make([]byte, 16) // 4 float32s

	// Encode 1.0, 2.0, 3.0, 4.0 as little-endian float32
	binary.LittleEndian.PutUint32(data[0:4], *(*uint32)(unsafe.Pointer(&[]float32{1.0}[0])))
	binary.LittleEndian.PutUint32(data[4:8], *(*uint32)(unsafe.Pointer(&[]float32{2.0}[0])))
	binary.LittleEndian.PutUint32(data[8:12], *(*uint32)(unsafe.Pointer(&[]float32{3.0}[0])))
	binary.LittleEndian.PutUint32(data[12:16], *(*uint32)(unsafe.Pointer(&[]float32{4.0}[0])))

	fmt.Printf("Raw data: %x\n", data)

	// Read back the values
	for i := 0; i < 4; i++ {
		val := *(*float32)(unsafe.Pointer(&data[i*4]))
		fmt.Printf("Value[%d]: %f\n", i, val)
	}

	// Test sqrPlusX
	sqrPlusX(data)

	// Check results
	for i := 0; i < 4; i++ {
		val := *(*float32)(unsafe.Pointer(&data[i*4]))
		fmt.Printf("After sqrPlusX[%d]: %f\n", i, val)
	}
}

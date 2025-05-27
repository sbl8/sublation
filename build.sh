#!/bin/bash

# Build script for Sublation project
set -e

echo "Building Sublation..."

# Create bin directory
mkdir -p bin

# Build compiler
echo "Building sublc compiler..."
cd cmd/sublc
go build -ldflags="-s -w" -o ../../bin/sublc .
cd ../..

# Build runtime  
echo "Building sublrun runtime..."
cd cmd/sublrun
go build -ldflags="-s -w" -o ../../bin/sublrun .
cd ../..

# Run tests
echo "Running tests..."
go test ./... -v

# Test compilation pipeline
echo "Testing compilation pipeline..."
./bin/sublc -O -validate examples/example.subs model.subl

echo "Testing runtime..."
echo "1.0 0.5 0.75 1.0" | ./bin/sublrun model.subl

echo "Build complete!"
echo "Binaries available in bin/"
echo "  sublc: Sublation compiler"  
echo "  sublrun: Sublation runtime"

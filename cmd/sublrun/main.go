package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"os"
	"runtime"

	sublation_runtime "github.com/sbl8/sublation/runtime"
)

func main() {
	var (
		workers   = flag.Int("workers", runtime.NumCPU(), "Number of worker goroutines")
		streaming = flag.Bool("streaming", false, "Enable streaming input processing")
		verbose   = flag.Bool("verbose", false, "Enable verbose output")
		version   = flag.Bool("version", false, "Show version information")
	)
	flag.Parse()

	if *version {
		fmt.Println("sublrun - Sublation Runtime v1.0.0")
		fmt.Printf("Built with Go %s\n", runtime.Version())
		return
	}

	args := flag.Args()
	if len(args) < 1 {
		fmt.Fprintf(os.Stderr, "Usage: %s [options] <model.subl> [input]\n", os.Args[0])
		flag.PrintDefaults()
		os.Exit(1)
	}

	modelPath := args[0]

	// Load the compiled model
	graph, err := sublation_runtime.LoadFromFile(modelPath)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}

	if *verbose {
		fmt.Printf("Loaded model with %d nodes and %d bytes payload\n",
			len(graph.Nodes), len(graph.Payload))
	}

	// Configure engine options
	opts := sublation_runtime.EngineOptions{
		Workers:     *workers,
		ArenaSize:   0, // Auto-calculate
		EnableStats: *verbose,
		Streaming:   *streaming,
	}

	// Create runtime engine
	engine, err := sublation_runtime.NewEngine(graph, &opts) // Pass address of opts
	if err != nil {
		log.Fatalf("Failed to create engine: %v", err)
	}

	if *verbose {
		fmt.Printf("Engine configured with %d workers\n", *workers)
	}

	if *streaming {
		runStreaming(engine, args[1:], *verbose)
	} else {
		runSingle(engine, args[1:], *verbose)
	}
}

// runSingle processes a single input or uses stdin
func runSingle(engine *sublation_runtime.Engine, inputs []string, verbose bool) {
	var inputData []byte
	var err error

	if len(inputs) > 0 {
		// Read from file
		inputData, err = os.ReadFile(inputs[0])
		if err != nil {
			log.Fatalf("Failed to read input file: %v", err)
		}
	} else {
		// Read from stdin
		scanner := bufio.NewScanner(os.Stdin)
		for scanner.Scan() {
			inputData = append(inputData, scanner.Bytes()...)
			inputData = append(inputData, '\n') // Preserve newlines if reading line by line
		}
		if err := scanner.Err(); err != nil {
			log.Fatalf("Failed to read from stdin: %v", err)
		}
	}

	if verbose {
		fmt.Printf("Processing %d bytes of input\n", len(inputData))
	}

	// The engine is initialized by NewEngine.
	// For a single execution with dynamic input, we pass it to Execute.
	// The Execute method will handle its own arena and sublate setup for this run.

	// Temporarily update the engine's graph payload with the current inputData.
	// This is a workaround because engine.Execute uses engine.graph.Payload.
	// A better long-term solution would be for engine.Execute to accept input data directly.
	originalPayload := engine.Graph().Payload
	engine.Graph().Payload = inputData

	// Create an execution context for this run.
	ctx := sublation_runtime.NewExecutionContext(len(engine.Graph().Nodes))

	if err := engine.Execute(ctx); err != nil {
		engine.Graph().Payload = originalPayload // Restore payload on error
		log.Fatalf("Engine execution failed: %v", err)
	}

	engine.Graph().Payload = originalPayload // Restore original payload after successful execution

	// Note: The output of engine.Execute(ctx) is implicitly in the sublates within the context's arena.
	// If specific output needs to be written to os.Stdout, further logic to extract it would be needed here.

	if verbose {
		fmt.Println("Execution completed")
	}
}

// runStreaming processes continuous input in streaming mode
func runStreaming(engine *sublation_runtime.Engine, inputs []string, verbose bool) {
	if len(inputs) > 0 {
		// Process multiple input files sequentially
		for _, filename := range inputs {
			data, err := os.ReadFile(filename)
			if err != nil {
				log.Printf("Warning: failed to read %s: %v", filename, err)
				continue
			}

			// Execute with this input
			output := make([]byte, engine.ArenaBytes())
			if err := engine.ExecuteStreaming(data, output); err != nil {
				log.Printf("Streaming execution error: %v", err)
				continue
			}

			if verbose {
				fmt.Printf("Processed %s (%d bytes) -> %d bytes output\n",
					filename, len(data), len(output))
			}

			// Write output
			os.Stdout.Write(output)
			os.Stdout.Write([]byte("\n"))
		}
	} else {
		// Read from stdin line by line
		scanner := bufio.NewScanner(os.Stdin)
		for scanner.Scan() {
			inputData := scanner.Bytes()

			// Execute with this input
			output := make([]byte, engine.ArenaBytes())
			if err := engine.ExecuteStreaming(inputData, output); err != nil {
				log.Printf("Streaming execution error: %v", err)
				continue
			}

			if verbose {
				fmt.Printf("Processed line (%d bytes) -> %d bytes output\n",
					len(inputData), len(output))
			}

			// Write output
			os.Stdout.Write(output)
			os.Stdout.Write([]byte("\n"))
		}
		if err := scanner.Err(); err != nil {
			log.Printf("Error reading stdin: %v", err)
		}
	}
}

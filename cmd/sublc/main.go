package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/sbl8/sublation/compiler"
)

func main() {
	var (
		optimize = flag.Bool("O", false, "Enable layout optimizations")
		validate = flag.Bool("validate", true, "Validate graph structure")
		debug    = flag.Bool("debug", false, "Include debug symbols")
		version  = flag.Bool("version", false, "Show version information")
	)
	flag.Parse()

	if *version {
		fmt.Println("sublc - Sublation Compiler v1.0.0")
		fmt.Println("Built with Go", "1.22.2")
		return
	}

	args := flag.Args()
	if len(args) < 2 {
		fmt.Fprintf(os.Stderr, "Usage: %s [options] <src.subs> <out.subl>\n", os.Args[0])
		flag.PrintDefaults()
		os.Exit(1)
	}

	srcFile, outFile := args[0], args[1]

	opts := compiler.CompileOptions{
		OptimizeLayout: *optimize,
		ValidateGraph:  *validate,
		DebugOutput:    *debug,
	}

	if err := compiler.CompileWithOptions(srcFile, outFile, opts); err != nil {
		log.Fatalf("compilation failed: %v", err)
	}

	fmt.Printf("Successfully compiled %s -> %s\n", srcFile, outFile)
}

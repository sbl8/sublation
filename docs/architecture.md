# Sublation: Vision and Design

Sublation is a minimalist, Go-based AI engine built on a custom *dataflow* model instead of traditional tensors.  Its core idea is **“structured data units”** called *sublates*, each carrying its own data and a local transformation kernel. Unlike flat N-dimensional tensors, each sublate can hold internal structure or topology (for example, a graph, a small array, or procedural pattern) and an associated compute function.  This lets us express models as networks of small data objects with local rules, which can be more memory- and compute-efficient for certain tasks.  Our philosophy is **“less is more”**: a small, self-contained engine (single static Go binary) that does **only what it needs** in a clear, direct way.

* **Project Goals:**  Make a high-performance inference+training engine in pure Go, extremely lightweight and deployable as one static binary.  Support both classic ML tasks (e.g. deep networks) and novel architectures built from sublates.  Emphasize simplicity and clarity in APIs.
* **Why Not Tensors?:**  Dense tensors (arrays) are general but can be wasteful or inflexible.  For example, large tensors force us to allocate big contiguous memory blocks and move data back and forth for each kernel.  In contrast, sublates allow *locality* and *streaming* of data.  Inspired by modern dataflow hardware, we can pipeline operations on sublate streams and avoid repeatedly materializing intermediate results.  This can reduce memory pressure and latency.  In short, sublates let us carve up computation into fine-grained units that map naturally to concurrency and streaming.  (For comparison, modern frameworks like SambaNova’s DataScale use a dataflow approach to fuse kernels and minimize memory transfer.)

**Compute Model:**  A Sublation model is a graph of sublates.  Each sublate has a data payload (e.g. a small array or struct) and one or more *kernels* (functions) that describe how it transforms its data and communicates with neighbors.  Runtime execution works like a streaming pipeline: when the inputs for a sublate are ready, its local kernel runs and produces outputs (possibly feeding other sublates).  This naturally exposes parallelism across sublates.  Since Go’s goroutines and channels make pipelines easy (see “Go Concurrency Patterns”), we can map each stage or sublate to a goroutine that reads input channels, processes data, and writes output channels.  This pipelined style avoids global synchronization and can run efficiently on multi-core CPUs.

For example, suppose we have an image processing model where each *pixel-sublate* applies a local filter.  Each pixel-sublate would carry its color and an index to neighbors, and a kernel that reads neighbor colors, updates itself, and sends results.  The runtime schedules all pixel-kernels in parallel, streaming data row by row.  In general, any graph structure (grids, trees, lists, etc.) can be expressed with sublates, and the compute graph follows the data connections.  This breaks away from monolithic tensor ops: instead of one huge matrix multiplication, we break it into many small math kernels that can overlap.

**Design Philosophy:**  Keep it *simple and explicit*. Avoid heavy magic or hidden state.  Use pure Go code for logic, keep interfaces minimal.  Each sublate’s kernel should be a Go function (or a method) that takes simple data types (scalars, slices, basic structs) and produces outputs.  We avoid metaprogramming or complex DSLs.  The “compiler” part of Sublation can be a command-line tool or library that reads a model specification (maybe Go code or a simple config) and builds an internal graph of sublates.  We then execute that graph directly.  Optional compilation stages (like fusing some kernels into one function) can be added for speed, but only if they keep the code understandable.

Inspiration comes from other minimal frameworks: for example, the Go project **Spago** is a “self-contained machine learning library” that uses a lightweight computation graph for training and inference.  Spago even serializes models with Go’s gob format.  HuggingFace’s **Candle** (in Rust) likewise is a “minimalist ML framework… focused on performance (including GPU support) and ease of use”.  These projects show it’s possible to build effective ML tools without enormous dependencies, which reinforces our goal of an elegant, pure-Go engine.

## Architecture and Internal Components

### Data Layout and Sublates

* **Sublate Structure:** Implement each sublate as a Go struct that holds its data and connections.  For example:

  ```go
  type Sublate struct {
      Data []float32        // the local payload (could be a slice, or custom struct)
      Neighbors []int      // indices or IDs of connected sublates
      Kernel func(*Sublate) // function that implements the local transform
  }
  ```

  This array-of-struct (or struct-of-slice) layout lets us pack each sublate’s data tightly in memory.  We can also use separate slices (struct-of-arrays) if that fits better (e.g. one big float32 slice for all sublates).
* **Memory Pools:**  To minimize GC overhead, we use pre-allocated buffers.  For example, have a single big slice of all sublate data and index into it, or use Go’s `sync.Pool` for temporary slices in kernels.  Since sublates may be fixed-size, allocate a `[]Sublate` upfront.  This means no allocations during runtime loops.  We can also consider `unsafe` and manual memory management (for serious speed) but try to keep pure Go semantics first.

## Graph Representation and Compilation

* **Graph Model:** The model is a directed dataflow graph.  Nodes are sublates, edges are data dependencies (channels or Go slices between kernels).  Build this graph at model-load time.  Store it as adjacency lists or an explicit edge list.
* **Topological Analysis:** Before execution, do a topological sort or Kahn’s algorithm to find execution order.  That gives us a sequence of “kernel launch” events: e.g. run sublate #5, then #2, etc.  In a static pipeline, this can be looped for inference.
* **Kernel Fusion (optional):**  When multiple sublates always fire together and one’s output is immediately the other’s input, consider fusing them into a single Go function to reduce overhead.  This is like manual inlining.  Use compile-time analysis: if A→B and B has no other inputs or outputs, merge A’s kernel and B’s kernel.  This reduces channel communication, emulating what hardware dataflow does.

## Memory and Execution Model

* **Streaming Execution:**  We run the graph either in a loop (for recurrent models) or once (feed-forward).  Use Go goroutines and channels for parallelism: each stage can read input values from channels, process them, and send outputs.  Go pipelines are a natural fit.  Alternatively, simpler: have worker goroutines pick ready nodes from a queue (protected by a mutex or channel) and run kernels in parallel.
* **Data Persistence:**  Sublates hold their state in memory between iterations (if stateful).  Keep one copy of parameters/weights for each sublate.  During training, gradients or updates can be stored in parallel buffers.

## Kernel Interface and Optimizations

* **Kernel API:**  Kernels are just Go functions (or methods).  For example:

  ```go
  func BlurKernel(s *Sublate) {
      // access s.Data and neighbor values, write output to s.Data or out channels
  }
  ```

  This keeps code straightforward.  We can also allow kernels to be defined in a data-driven way (e.g. an interpreter that executes simple math formulas), but direct Go code is simplest to start.
* **Math Performance:**  For heavy math (e.g. linear algebra inside a kernel), prefer well-optimized routines.  Options include:

  * **Pure Go:** Use [Gonum](https://gonum.org/v1/) for matrix/vector ops (which is pure Go) to avoid cgo.  Gonum is idiomatic and reasonably fast for moderate sizes.
  * **Assembly Intrinsics:** For best speed on critical loops, write key operations (like vector dot, axpy) in Go assembly (AVX/SSE).  Note: Go does not have first-class SIMD intrinsics, so we’d write a few `.s` files (as Gorgonia does with `mathutils_amd64.s`).
  * **BLAS via cgo:** Optionally, allow linking to OpenBLAS or BLIS.  Set up the build so that if CGO is enabled, `blas64` bindings call a C BLAS (OpenBLAS, for example).  For performance-critical apps, the user could build with `CGO_ENABLED=1` and provide a static BLAS library.  But by default, stick to pure Go so the engine is self-contained.
* **SIMD/Parallel:** Detect CPU features (`golang.org/x/sys/cpu`) and use them in code paths if writing assembly.  For multi-core, either let Go scheduler handle it (multiple goroutines in parallel) or manually split large kernels among threads.

## Scheduler and Parallelism

* **Task Scheduling:** After topological sort, one simple approach is: iterate layers of the graph, at each layer spawn goroutines for each node, synchronize, then move on.  A more dynamic approach is to use a worker pool: maintain a channel of “ready tasks” (initially the input nodes), and let worker goroutines pull tasks and execute kernels, pushing newly ready tasks when dependencies are satisfied.
* **Goroutines & Channels:** Use Go channels to implement the dataflow.  For example, each edge can be a channel of messages.  Then `select` can be used for multi-input nodes.  However, to keep things minimal, a static schedule loop may be simpler: run node1, then node2, etc.  If concurrency is needed, wrap the kernel call in a goroutine and use `sync.WaitGroup` or channels to wait for all of a layer.  The official Go blog shows how easy it is to chain pipelined stages.
* **Backpressure:** In training or streaming, data may arrive continuously.  The engine should handle bursts without deadlocking.  Use buffered channels or dropping old data if real-time throughput is needed.

## Training Loop and Optimization

* **Gradient Computation:** To train, we need gradients.  We have two main options:

  1. **Autodiff:** Implement automatic differentiation on the sublate graph.  Each kernel must have a corresponding backward kernel.  During a backward pass, propagate gradients from outputs back to inputs.  This is akin to backprop.  We can borrow ideas from autograd: e.g. record operations during forward execution and replay them in reverse.  Spago does this with a “dynamic define-by-run” graph.  We can do similar by instrumenting our run or by generating a static reverse graph from the model definition.
  2. **Symbolic/Manual Gradients:** For each kernel, user provides a gradient function.  This is simpler to implement but less flexible.  Each sublate’s kernel would then also emit gradients for its inputs.
* **Optimizers:** Implement standard optimizers (SGD, Adam, RMSProp, etc.) in Go.  These operate on sublate parameters.  For example, after each batch, loop over all sublates and adjust `sublate.Data` by their stored gradients.  Keep optimizer state (momentum) either inside each sublate or in a parallel structure.
* **Batching:** Even though our model is dataflow, we can still do mini-batching by processing multiple inputs through the pipeline before updating weights.  Alternatively, treat each “data sample” as a separate invocation of the graph, accumulating gradients between runs.
* **Scheduling in Training:** Training naturally splits into forward and backward passes.  Forward pass: run sublates in graph order (possibly in parallel layers).  Record necessary intermediate values or operations.  Backward pass: reverse order.  Use the same scheduling machinery (just reversed order, using stored dependencies).

## Model Serialization and Compilation

* **Serialization:** We need to save and load models (graph structure + weights).  Options:

  * **Go Gob:** Since we’re in pure Go, Gob is a quick built-in binary serializer.  Spago uses Gob for neural models.  We could simply `gob.Encoder` our graph and slice of weights.  Pros: easy; cons: not language-agnostic.
  * **ONNX or Protobuf:** For wider interoperability, use ONNX (via [onnx-go](https://github.com/oramasearch/onnx-go) or similar) or define a custom protobuf.  ONNX has many op definitions, but our sublate model is not tensor-centric, so ONNX might not fit perfectly.
  * **Custom JSON/Flatbuffers:** Define a compact JSON or flatbuffer schema for sublate graphs.  This can be human-readable (JSON) or fast (flatbuffers).  FlatBuffers tend to be faster than Gob, but \[benchmark shows Gob can be fast for Go programs】(no direct ref).
    We recommend starting with Gob for simplicity, then add an ONNX-export path if needed for integration.
* **Compilation:** Since we aim for a single binary, “compilation” here means turning a model description into an optimized runtime plan.  That could be:

  * A build step that generates Go source code for the model (embedding weights as constants) and then runs `go build` to produce a specialized binary.  This is heavyweight but yields a monolithic executable for one model.
  * Or a generic runtime that loads a model file and interprets it.  This is easier and keeps one engine for all models.  The tradeoff is performance.  We can mitigate interpreter overhead by aggressive in-memory optimizations (like fusing kernels or flattening loops).
    We suggest an initial approach of an interpreter-with-optimizations.  If ultimate speed is needed, optional ahead-of-time codegen to Go (or even to assembly via `goasm`) could be implemented later.

## CPU Execution (Old and New Hardware)

* **Portability:** Support both x86-64 and ARM64 (and others).  Use Go’s build tags to provide architecture-specific code.  For example, implement kernels in generic Go first, then special-case hand-tuned AVX2/AVX-512 in `*.amd64.s` and NEON in `*.arm64.s`.  Only compile the ones for the target arch.  This keeps performance high on new CPUs, while default code still runs on older ones.
* **Concurrency:** Fully leverage multi-core: allow specifying number of worker goroutines.  Pin goroutines to OS threads if needed for NUMA optimization (Go 1.18+ has `runtime.LockOSThread()` if we want).  Generally let Go schedule threads across cores.
* **Memory Alignment:** Ensure slices are aligned (Go 1.20+ generally does this) for SIMD.  For large weight buffers, allocate with sufficient padding or use `unsafe.Align`.
* **Profiling and Tuning:** Include support for profiling (built-in Go `pprof`) so the solo engineer can identify hotspots.  For example, time how much is spent in each kernel vs. scheduling overhead.  Use that to decide if manual optimization is needed in a kernel.

## Future GPU/Vendor-Agnostic Support

We aim to keep a path to GPU without rewriting the whole engine.  Since we avoid vendor lock-in, plan for:

* **OpenCL (via Pure Go):** Use a pure-Go OpenCL binding (e.g. [opencl-pure/pureCL](https://pkg.go.dev/github.com/opencl-pure/pureCL)), which uses no cgo.  This can call into the system’s OpenCL or even driver libraries dynamically.  At runtime we could detect GPUs and, for large sublates, dispatch kernels there.  For example, a matrix-heavy sublate kernel could have both CPU Go code and a string with an OpenCL kernel; at runtime decide which to use.  The [pureCL library](#11) shows we can wrap OpenCL calls in Go without blowing up the binary size.
* **Vulkan Compute / CUDA:** Alternatively, one could use Vulkan compute shaders (with a Go binding) or CUDA via C-bindings.  These are more complex and less portable.  We recommend starting with OpenCL for vendor-agnosticism, then later consider a CUDA/CuDNN path for NVIDIA if absolutely needed.
* **Abstracted Kernel Interface:** Design kernel functions so that they can be swapped out.  For example, a CPU `MatMul` kernel and a GPU `MatMul`.  The engine can decide at runtime or compile time which implementation to use.  This keeps the high-level logic the same.

## Recommendations and Roadmap

* **Inference Optimizations:**  Provide a fast path for inference: e.g. disable any training-only machinery, simplify graph (remove gradient nodes), and allow user to indicate “inference mode”.  Pre-compute any static data (e.g. pre-normalize weights if ReLU or batchnorm).  Consider compiling frequent small models into direct Go code for speed.
* **Training Loop:**  Implement mini-batch SGD with standard features: shuffling, learning rate schedules, early stopping.  Make it easy to plug in new optimizers.  Consider “sublate batching”: e.g. allow grouping sublates of the same type into a batch to use vectorized math.
* **Graph/Model Compilation:**  Build a CLI that takes a model spec (e.g. in Go or JSON) and outputs a compiled model file.  The runtime can be the same binary reading that file.  Optionally, output verbose stats on layer sizes and memory use.
* **Hardware Auto-Tuning:**  At startup or compile time, detect CPU features and choose the best code paths (e.g. use `cpu.X86.HasAVX2`).  For GPUs, optionally run small micro-benchmarks to pick block sizes or work-group sizes.
* **Memory Pooling:**  Use object pools (`sync.Pool`) for frequently-used temporary buffers to reduce allocations.  For example, if many sublates need scratch space, reuse slices instead of `make` each time.
* **Threading Strategy:**  Experiment whether a fixed number of worker goroutines (equal to cores) with work-stealing yields better throughput than a goroutine per sublate.  Because Go channels have some overhead, a simpler worker-pool may be leaner for tiny kernels.
* **Evaluation Metrics:**  Build end-to-end benchmarks comparing Sublation to baseline (e.g. Gorgonia or even a simple Python/TensorFlow model) on a few models.  Measure latency, throughput, and memory.  Use this data to focus optimization efforts.

## Libraries and Tools

To keep Sublation elegant yet efficient, consider these candidates:

* **Gonum** (pure Go numeric library): For linear algebra and basic math. Lightweight and well-maintained.  Good fallback if you don’t link BLAS.
* **pureCL** (Go OpenCL wrapper): For GPU support without cgo. It’s minimal and “try to have all functions of OpenCL” with Go types.
* **Spago**: Although conceptually similar, Spago is more of a neural toolkit. However, it’s a living example of a pure-Go graph engine with autodiff.  Reading its code can inspire API design (e.g. define-by-run graph, Gob serialization).
* **ONNX-Go**: If you want interoperability, [onnx-go](https://github.com/oramasearch/onnx-go) can import/export ONNX models.  It’s experimental, but shows how to map between Go and ONNX.  One could optionally export Sublation models to ONNX for inspection.
* **FlatBuffers/Protobuf**: For serialization, Protobuf (via `golang/protobuf`) is stable and fast.  Gob is easiest, Protobuf is language-agnostic.  FlatBuffers is zero-copy but has tricky build steps; can consider it later if serialization speed is a concern.
* **Assembly/Intrinsics:** If micro-optimizing, write critical kernels in Go assembly (`*.s`).  The Go compiler issue shows that built-in intrinsics are not available, so custom asm is the way to get SSE/AVX. Use existing examples (e.g. Gorgonia’s mathutils or the [golang/stdlib](https://golang.org/pkg/math/bits/) for bit ops).
* **Concurrency Primitives:** Go’s standard library (`sync`, channels) is usually enough.  For advanced scheduling, consider a lightweight work-stealing queue implementation (there are Go libraries for that), but first try a simple `chan`+goroutines design.

## Conclusion

Sublation aims to be a **concise, understandable AI engine**.  By focusing on a dataflow model of “sublates” with local kernels, we break free of bulky tensor frameworks.  This opens the door to flexible architectures and efficient streaming execution.  The architecture is designed so that a single Go binary can serve as compiler, interpreter, and runtime.  With careful memory management and Go’s concurrency, we achieve high CPU utilization on a variety of hardware.  And by building in an abstraction layer (e.g. using pure-Go OpenCL), we leave room to offload heavy compute to GPUs in the future, without rewriting our core engine.

**References:** For example, SambaNova’s dataflow paper illustrates how pipelined kernel execution greatly reduces memory movement, and Go’s concurrency blog shows how easily one can build pipelines in Go.  Projects like Spago and Candle demonstrate that minimal, high-performance ML frameworks are feasible in Go or similar languages.  We’ll draw on these insights to make Sublation both elegant and effective.

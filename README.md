# Emmy Kernel IR → NVIDIA PTX Compiler

A Python-based compiler prototype that lowers **Emmy Kernel IR** directly to NVIDIA **PTX assembly**. The project implements an IR-to-PTX compilation pipeline, including graph traversal, instruction lowering, virtual register management, control-flow generation, memory addressing, and kernel code generation.

Developed as part of the **Emmy Hackathon 2026**, this project explores the implementation of a low-level GPU compiler backend without relying on Emmy's CUDA renderer or `nvcc` for code generation.

## Overview

The compiler takes serialized Emmy Kernel IR as input and translates supported operations into PTX assembly.

Its main objective is to bridge a high-level GPU kernel representation and NVIDIA's low-level virtual instruction set, while explicitly managing the intermediate representation, execution flow, register usage, and memory operations.

**Key areas of implementation:**

- IR graph deserialization and visitor-based traversal.
- Lowering of high-level operations into PTX instructions.
- Virtual register creation and management.
- PTX instruction representation and basic-block construction.
- Conditional execution and loop lowering.
- Tensor indexing and memory address calculation.
- Kernel ABI generation, including parameters and launch metadata.
- Reduction primitives involving warp shuffles and shared memory.

## Compiler Architecture

The compiler is structured as a sequence of lowering and code-generation stages:

```text
       Emmy Kernel IR (.ir.json)
                  |
                  v
          Graph.from_dict
                  |
                  v
             NodeVisitor
                  |
          IR-to-PTX Lowering
                  |
                  v
              PTXAssembly
         /        |         \
   Registers  Instructions  Basic Blocks
         \        |         /
                  |
                  v
             PTX Builder
                  |
                  v
           Kernel Codegen
                  |
                  v
           NVIDIA PTX (.ptx)
                  |
                  v
               ptxas
```

### Main Components

| Component | Responsibility |
|---|---|
| `compile.py` | Compiler entry point. Loads the IR graph and invokes the lowering pipeline. |
| `ast_visitor.py` | Implements visitor-based traversal and lowering rules for supported IR operations and expressions. |
| `ptx_ir.py` | Defines the PTX-oriented intermediate representation, including instructions, virtual registers, and basic-block management. |
| `ptx_builder.py` | Renders the internal instruction representation into PTX assembly syntax. |
| `codegen.py` | Generates kernel-level PTX code, including entry points, parameter declarations, register declarations, and memory setup. |
| `meta.py` | Extracts kernel metadata, argument information, and launch configuration. |
| `submit.py` | Provides the submission and validation interface against the hackathon runner. |

## Implemented Features

### 1. IR Traversal and Instruction Lowering

The compiler reconstructs Emmy's typed IR graph and uses a visitor architecture to translate supported operations into lower-level instructions.

This separates IR interpretation from PTX generation and makes individual lowering rules easier to extend and maintain.

### 2. PTX Intermediate Representation

Rather than generating assembly strings directly during IR traversal, the compiler uses an intermediate PTX representation.

It includes:

- Typed virtual registers.
- PTX instruction and opcode abstractions.
- Instruction sequences and basic blocks.
- Label and branch management.
- Shared-memory descriptors.
- Register declaration tracking.

This intermediate representation provides a layer between high-level IR semantics and the final assembly syntax.

### 3. Control Flow

The compiler includes lowering mechanisms for structured control flow:

- Conditional execution through predicates and branches.
- Loop construction with labels and conditional jumps.
- Loop-carried state updates.
- Strided iteration patterns.

These mechanisms provide the foundation for translating nested execution structures into PTX control flow.

### 4. Memory Addressing and Tensor Indexing

The backend implements tensor address calculation using multidimensional indices and row-major layout information.

For a tensor with dimensions \(D_0, D_1, \ldots, D_{n-1}\), the linear element offset is:

\[
\text{offset} =
\sum_{i=0}^{n-1}
\left(
\text{index}_i
\prod_{j=i+1}^{n-1} D_j
\right)
\]

The resulting element offset is converted into a byte offset according to the element size and added to the base pointer.

The implementation also includes optimizations for certain index expressions, such as replacing multiplication by powers of two with shift operations.

### 5. GPU Reduction Primitives

The compiler contains lowering implementations for several reduction building blocks:

- Accumulator initialization and updates.
- Strided iteration over reduction dimensions.
- Warp-level butterfly reductions using shuffle instructions.
- Shared-memory intermediate storage.
- CTA synchronization through barriers.
- Tree-based combination of partial results.

These primitives demonstrate how higher-level reduction operations can be decomposed into warp-level and block-level GPU execution mechanisms.

### 6. Kernel Code Generation

The code-generation stage is responsible for assembling the complete PTX kernel:

- Kernel entry-point declarations.
- Parameter and pointer handling.
- Parameter loads and address-space conversions.
- Virtual register declarations.
- Shared-memory declarations.
- Instruction and basic-block emission.

Physical register allocation and final machine-code assembly are delegated to NVIDIA's `ptxas`.

## Supported Workloads

The repository includes eight initial workloads covering pointwise operations and reductions.

| Category | Workload | Operation |
|---|---|---|
| Pointwise | `p00_scalar_add` | Scalar addition |
| Pointwise | `p01_vector_add` | Vector addition |
| Pointwise | `p02_scale_shift` | Scaling and shifting |
| Pointwise | `p03_relu_mul` | ReLU and multiplication |
| Reduction | `r01_array_sum` | One-dimensional sum |
| Reduction | `r02_big_sum` | Large one-dimensional sum |
| Reduction | `r03_mean` | Mean reduction |
| Reduction | `r04_row_sum` | Row-wise reduction |

The workload set exercises different aspects of the compiler, from basic arithmetic and memory operations to reduction loops, synchronization, and multidimensional tensor indexing.

## Getting Started

### Requirements

The compiler uses Emmy's Python package to deserialize the input IR. The validation harness additionally requires NumPy and Requests.

```bash
python -m pip install emmy-ml==0.3.11 numpy requests
```

### Compile a Kernel

Generate PTX from a serialized Kernel IR file:

```bash
python compile.py problems/p00_scalar_add/k_add_pointwise.ir.json > out.ptx
```

The generated assembly is written to standard output. Diagnostic logging can be enabled with `--verbose`.

### Validate a Workload

Run the submission harness against an individual problem:

```bash
python submit.py p00_scalar_add --compiler ./compile.py
```

For a reduction workload:

```bash
python submit.py r01_array_sum --compiler ./compile.py
```

The runner assembles and executes the generated PTX using the launch configuration specified in `meta.json`, then compares the output against the reference data using the configured numerical tolerances.

The server endpoint and target architecture can be configured through the corresponding `submit.py` arguments.

## Design Considerations

### Separation of Concerns

The compiler separates graph traversal, instruction lowering, PTX representation, assembly rendering, and kernel-level code generation.

This architecture avoids coupling IR semantics to assembly string generation and makes the backend easier to extend.

### Virtual Register Management

The compiler maintains its own virtual register representation and emits register declarations in the generated PTX.

This allows lowering rules to work with symbolic register references without manually managing physical GPU registers.

### Explicit Control-Flow Generation

Structured loops and conditional regions must be translated into PTX labels, predicates, branches, and state updates.

The compiler handles these transformations explicitly rather than delegating control-flow generation to a higher-level CUDA compiler.

### GPU Execution Model

Reduction lowering illustrates the interaction between GPU execution levels:

- Thread-local accumulation.
- Warp-level communication.
- Shared-memory exchange between warps.
- Block-level synchronization.

This makes the project useful for exploring the relationship between compiler transformations and the underlying GPU execution model.

## Current Scope and Limitations

This project is a focused compiler prototype rather than a complete implementation of Emmy Kernel IR.

The current implementation targets a limited set of data types and operations, primarily the supplied FP32 workloads.

The visitor only supports a subset of the available IR nodes. General-purpose control-flow handling, arbitrary expression lowering, and broader operation coverage remain areas for development.

In particular:

- General comparison and conditional-expression support is limited.
- The expression cache requires additional care around control-flow boundaries.
- Multidimensional launch mapping and indexing require broader correctness testing.
- Matrix multiplication, normalization, softmax, and attention are not currently implemented as complete end-to-end compiler paths.
- The compiler does not implement a general physical register allocator or advanced optimization passes.

The supported workload list describes the supplied inputs, not a claim that every workload has passed end-to-end validation on every target GPU.

## Repository Structure

```text
.
├── compile.py
├── ast_visitor.py
├── ptx_ir.py
├── ptx_builder.py
├── codegen.py
├── meta.py
├── utils.py
├── submit.py
├── docs/
│   └── API.md
├── examples/
│   └── k_sum_1_reduce.ptx
└── problems/
    ├── p00_scalar_add/
    ├── p01_vector_add/
    ├── p02_scale_shift/
    ├── p03_relu_mul/
    ├── r01_array_sum/
    ├── r02_big_sum/
    ├── r03_mean/
    └── r04_row_sum/
```

## Motivation

GPU compiler development requires understanding both the semantics of intermediate representations and the hardware execution model.

This project explores that boundary by implementing a compilation pipeline that translates high-level GPU kernel IR into low-level PTX assembly, with explicit handling of instructions, registers, control flow, memory addressing, and parallel reduction mechanisms.

The goal is to build a deeper understanding of GPU code generation and the design decisions involved in implementing a compiler backend.

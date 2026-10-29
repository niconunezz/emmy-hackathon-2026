# Emmy kernel IR to PTX

A small compiler that reads an Emmy kernel IR (`*.ir.json`) and writes PTX.
Built for the Emmy Hackathon 2026.

```
python compile.py problems/p00_scalar_add/k_add_pointwise.ir.json > out.ptx
python submit.py p00_scalar_add out.ptx
```

`meta.json` is read from the same folder as the IR (entry name, kernel
arguments, launch geometry).

## Status

Category 1 (pointwise) passes: `p00_scalar_add`, `p01_vector_add`,
`p02_scale_shift`, `p03_relu_mul`.

Category 2 (reductions) is not implemented. More on that at the end.

## How it works

- `ast_visitor.py` walks the IR and produces PTX-level instructions, grouped
  in blocks. There is one `visit_<NodeClass>` method per IR node type.
- `ptx_ir.py` has the data model: registers, instructions and the register
  allocator.
- `ptx_builder.py` turns those instructions into PTX text.
- `codegen.py` adds the header, the `.param` list, the `.reg` declarations
  and the pointer setup.
- `meta.py` reads `meta.json`.

All registers are virtual and ptxas does the real allocation. The `.reg`
lines are generated from counters after lowering, so they always match what
the kernel uses.

## Design choices

- Indices stay in elements for the whole lowering and become bytes once, when
  the address is formed: `mul.wide.u32` by the element size, then add to the
  base pointer. A constant index folds into a byte offset instead.
- `x * 2^k` in an index expression becomes `shl`.
- Tail guard: the axis extent is compared with the threads launched
  (grid x block from `meta.json`). With fewer elements than threads, the
  extra threads jump to the end of the kernel. With more elements than
  threads, the compiler stops with an error instead of emitting code that
  would be wrong.
- Unrolled bodies keep their shape: a `Write` with several values becomes a
  single `st.global.v2` or `v4`.
- Kernel pointers are loaded with `ld.param` and converted once with
  `cvta.to.global` in the prologue.
- Anything unsupported raises `NotImplementedError` with the name of the
  thing. I preferred a loud failure over PTX that looks fine and is wrong.

## What is missing

- **Reductions (r01 to r04).** The IR needs a strided loop with a
  loop-carried accumulator, warp shuffles, shared memory with a barrier, and
  `TreeHalve`. The accumulator is the awkward part: lowering currently
  allocates a new register per assignment, and `acc <- add(acc, x)` has to
  reuse the same one.
- **Types and shapes.** Only f32 and 1-D indexing. Elementwise ops are add,
  multiply and relu.
- **Address computation.** Every access recomputes its own address. The
  better version is one base address per buffer plus immediate offsets
  (`[reg+4]`), which would also let contiguous loads merge into `ld.v2`.
- **Index cache.** The cache for index expressions assumes straight-line
  code. It has to become block-aware before loops can use it.
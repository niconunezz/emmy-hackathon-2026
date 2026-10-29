from __future__ import annotations
from ptx_builder import render_blocks
from utils import _nbytes

PTX_VERSION = "8.0"
PTX_TARGET = "sm_89"

# register class prefix, type used in its .reg declaration
REG_DECL_TYPE = {"f": "f32", "fd": "f64", "r": "b32", "rd": "b64", "p": "pred"}

PY_TYPE_TO_PTX = {"float" : "f32"}
class Emitter:
    def __init__(self, ctx, mstruct, blockManager):
        self.ctx = ctx
        self.mstruct = mstruct
        self.blockManager = blockManager

    def emit(self) -> str:
        name = self.mstruct.kernel_name
        return "\n".join([
            f".version {PTX_VERSION}",
            f".target {PTX_TARGET}",
            ".address_size 64",
            "",
            f"{self._smem()}",
            f".visible .entry {name}(",
            self._params(name),
            ")",
            "{",
            *self._reg_decls(),
            *self._load_params(name),
            render_blocks(self.blockManager),
            "}",
        ])

    def _smem(self) -> str:
        ret = ""
        for smem in self.ctx.smem_instr:
            dtype = PY_TYPE_TO_PTX[smem.dtype]
            align = max(_nbytes(dtype), smem.align)
            ret += f".shared .align {align} .{dtype} {smem.nm}[{smem.size[0]}];\n"

        return ret

    def _params(self, name: str) -> str:
        return ",\n".join(f"\t.param .u64 {name}_param_{i}" for i in range(self.mstruct.num_args))

    def _reg_decls(self) -> list[str]:
        decls = []
        for prefix, count in self.ctx.reg_cnt.items():
            if prefix not in REG_DECL_TYPE:
                raise ValueError(f"unknown register class {prefix!r}")
            decls.append(f"\t.reg .{REG_DECL_TYPE[prefix]} %{prefix}<{count}>;")
        return decls

    def _load_params(self, name: str) -> list[str]:
        """
        do param init with
        ld.param and cvta.to.global
        """

        names = list(self.ctx.signature)
        loads = [f"\tld.param.u64 %rd{i}, [{name}_param_{i}];" for i in range(len(names))]
        cvtas = [f"\tcvta.to.global.u64 {self.ctx.name_to_reg[n].nm}, %rd{i};"
                 for i, n in enumerate(names)]
        return loads + cvtas
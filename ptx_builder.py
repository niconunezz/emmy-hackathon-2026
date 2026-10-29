from __future__ import annotations
from typing import Callable
from ptx_ir import COND, PTXInstr, PTXOpTy, Register

_SETP_CMP = {COND.GT: "gt"}


def _operand(x) -> str:
    return x.nm if isinstance(x, Register) else str(x)


def _mul(ins, out, dtype, args):
    if (args or {}).get("wide"):               # 32-bit operands -> 64-bit result for idx calcs
        return f"mul.wide.u32 {out}, {ins[0]}, {ins[1]};"
    lo = ".lo" if dtype[0] in "us" else ""     # .lo/.hi/.wide
    return f"mul{lo}.{dtype} {out}, {ins[0]}, {ins[1]};"

_SIMPLE_OPS: dict[PTXOpTy, Callable[..., str]] = {
    PTXOpTy.MUL:  _mul,
    PTXOpTy.SHL:  lambda ins, out, dtype, args: f"shl.b32 {out}, {ins[0]}, {ins[1]};",
    PTXOpTy.ADD:  lambda ins, out, dtype, args: f"add.{dtype} {out}, {ins[0]}, {ins[1]};",
    PTXOpTy.RELU: lambda ins, out, dtype, args: f"max.{dtype} {out}, 0f00000000, {ins[0]};",
}
def _emit_pid(instr: PTXInstr) -> list[str]:
    ctaid, ntid, tid = (r.nm for r in instr.ins)
    return [f"mov.u32 {ctaid}, %ctaid.x;",
            f"mov.u32 {ntid}, %ntid.x;",
            f"mov.u32 {tid}, %tid.x;",
            f"mad.lo.s32 {instr.outs.nm}, {ctaid}, {ntid}, {tid};"]

def _emit_load(instr: PTXInstr) -> str:
    return f"ld.global.{instr.outs.dtype} {instr.outs.nm}, [{instr.ins[0].nm}];"

def _emit_store(instr: PTXInstr) -> str:
    address, values = instr.ins
    dtype = values[0].dtype
    names = ", ".join(v.nm for v in values)
    if len(values) == 1:
        return f"st.global.{dtype} [{address.nm}], {names};"
    return f"st.global.v{len(values)}.{dtype} [{address.nm}], {{{names}}};"

def _emit_guard(instr: PTXInstr, end_label: str) -> list[str]:
    lhs, rhs = instr.ins
    cmp = _SETP_CMP[instr.args["cond"]]
    return [f"setp.{cmp}.{instr.args['dtype']} {instr.outs.nm}, {_operand(lhs)}, {_operand(rhs)};",
            f"@{instr.outs.nm} bra {end_label};"]


def _emit_instr(instr: PTXInstr) -> list[str]:
    if instr.op_type is PTXOpTy.PID:
        return _emit_pid(instr)
    if instr.op_type is PTXOpTy.LOAD:
        return [_emit_load(instr)]
    if instr.op_type is PTXOpTy.STORE:
        return [_emit_store(instr)]
    handler = _SIMPLE_OPS.get(instr.op_type)
    if handler is None:
        raise NotImplementedError(f"cannot render {instr.op_type.name}")
    out = instr.outs
    return [handler([_operand(i) for i in instr.ins],
                    out.nm if out else None, out.dtype if out else None, instr.args)]

""" block0 is straight-line code. Every later block opens with a guard that, when taken,
skips the block's body and lands on the label emitted right after it. A single retOp
closes the kernel, so the guard label always has an instruction to land on."""

def render_blocks(blocks: dict[str, list[PTXInstr]]) -> str:
    
    lines: list[str] = []
    for name, instrs in blocks.items():
        body = list(instrs)
        if name != "block0":
            guard, *body = instrs
            if guard.op_type is not PTXOpTy.IF:
                raise ValueError(f"{name} must start with a guard, got {guard.op_type.name}")
            lines += _emit_guard(guard, end_label=name)
        for instr in body:
            lines += _emit_instr(instr)
        if name != "block0":
            lines.append(f"{name}:")
    lines.append("ret;")
    return "\n".join(l if l.endswith(":") else "\t" + l for l in lines) + "\n"
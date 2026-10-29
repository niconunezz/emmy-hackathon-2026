from __future__ import annotations
import logging
from collections import defaultdict
from meta import MetaStruct
from ptx_ir import COND, PTXAssembly, PTXInstr, PTXOpTy, Register

log = logging.getLogger(__name__)

LITERAL_DTYPE = {int: "u32", float: "f32"}
DTYPE_BYTES = {"f32": 4}
BIN_OPS = {"+": PTXOpTy.ADD, "*": PTXOpTy.MUL}
ELEMENTWISE_OPS = {"add": PTXOpTy.ADD, "multiply": PTXOpTy.MUL, "relu": PTXOpTy.RELU}

def _dtype_of(operand) -> str:
    return operand.dtype if isinstance(operand, Register) else LITERAL_DTYPE[type(operand)]


def dtype_from_bin_expr(lhs, rhs) -> str:
    lhs_dtype = _dtype_of(lhs)
    bits = max(int(lhs_dtype[1:]), int(_dtype_of(rhs)[1:]))
    return ("f" if lhs_dtype[0] == "f" else "u") + str(bits)

def _nbytes(dtype: str) -> int:
    try:
        return DTYPE_BYTES[dtype]
    except KeyError:
        raise NotImplementedError(f"element size of dtype {dtype!r}") from None

def _is_power_of_two(n) -> bool:
    return isinstance(n, int) and n > 0 and (n & (n - 1)) == 0

class NodeVisitor:
    def __init__(self, ctx: PTXAssembly, mstruct: MetaStruct):
        self.ctx = ctx
        self.mstruct = mstruct
        self.ins: dict[str, list[PTXInstr]] = defaultdict(list)   # block name -> instructions set
        self.curr_block = "block0"
        self.block_cnt = 0

    # node type dispatch ---------------------

    def visit(self, node):
        for cls in type(node).__mro__:
            handler = getattr(self, f"visit_{cls.__name__}", None)
            if handler is not None:
                return handler(node)
        log.warning("no lowering for %s", type(node).__name__)

    def _emit(self, instr: PTXInstr) -> None:
        self.ins[self.curr_block].append(instr)

    # containers ---------------

    def visit_list(self, node):
        return [self.visit(n) for n in node]

    def visit_tuple(self, node):
        return tuple(self.visit(n) for n in node)

    def visit_Body(self, node):
        for stmt in node:
            self.visit(stmt)

    def visit_Node(self, node):
        return self.visit(node.op)

    def visit_InputOp(self, node):
        return None          

    def visit_KernelOp(self, node):
        return self.visit(node.body)

    def visit_Tile(self, node):
        self.visit(node.axes)
        self.visit(node.body)

    # leaves --------------------------------------

    def visit_Dim(self, node):
        return node.value
    def visit_DataType(self, node):
        return node.name

    def visit_Literal(self, node):
        return node.value

    def visit_Var(self, node):
        reg = self.ctx.name_to_reg.get(node.name)
        if reg is None:
            raise ValueError(f"variable {node.name!r} has no register")
        return reg

    # pid --------------------------------------------------
    def visit_Axis(self, node):
        ctx = self.ctx
        ctaid, ntid, tid = (ctx.new_reg("u32") for _ in range(3))
        pid = ctx.new_reg("u32")
        ctx.pid_var_nm = node.name
        ctx.name_to_reg[node.name] = pid
        self._emit(PTXInstr(PTXOpTy.PID, (ctaid, ntid, tid), pid))

        extent = self.visit(node.extent)
        launched = self.mstruct.total_threads
        if extent > launched:
            raise ValueError(f"axis {node.name} needs {extent} threads but only {launched} are launched")
        if extent < launched:             # surplus threads -> emit guard
            pred = ctx.new_pred()
            self.block_cnt += 1
            self.curr_block = f"block{self.block_cnt}"
            self._emit(PTXInstr(PTXOpTy.IF, (pid, extent - 1), pred,
                                {"cond": COND.GT, "dtype": pid.dtype}))

    # mem ---------------------------------------------------------------------------

    def _index_of(self, node):
        idx = self.visit(node.index)
        if len(idx) != 1:
            raise NotImplementedError(f"only 1-D indexing is supported, got {len(idx)} indices")
        return idx[0]

    def _element_address(self, ptr: Register, idx, dtype: str) -> Register:
        # ptr + idx * sizeof(dtype)

        nbytes = _nbytes(dtype)
        if isinstance(idx, int):
            offset = idx * nbytes
            if offset == 0:
                return ptr
            address = self.ctx.new_reg(dtype_from_bin_expr(ptr, offset))
            self._emit(PTXInstr(PTXOpTy.ADD, (ptr, offset), address))
            return address
        idx_bytes = self.ctx.new_reg(dtype_from_bin_expr(ptr, idx))
        self._emit(PTXInstr(PTXOpTy.MUL, (idx, nbytes), idx_bytes, {"wide": True}))
        address = self.ctx.new_reg(dtype_from_bin_expr(ptr, idx_bytes))
        self._emit(PTXInstr(PTXOpTy.ADD, (ptr, idx_bytes), address))
        return address

    def visit_Load(self, node):
        ptr = self.ctx.name_to_reg[node.input]
        idx = self._index_of(node)
        dtype = self.visit(node.dtype)
        address = self._element_address(ptr, idx, dtype)
        value = self.ctx.new_reg(dtype)
        self.ctx.name_to_reg[node.name] = value
        self._emit(PTXInstr(PTXOpTy.LOAD, (address,), value))

    def visit_Write(self, node):
        values = [self.ctx.name_to_reg[n] for n in node.values]
        ptr = self.ctx.name_to_reg[node.output]
        idx = self._index_of(node)
        address = self._element_address(ptr, idx, values[0].dtype)
        self._emit(PTXInstr(PTXOpTy.STORE, (address, values), None))

    # arith ops ------------------------------

    def visit_Assign(self, node):
        op = ELEMENTWISE_OPS.get(node.op.name)
        if op is None:
            raise NotImplementedError(f"elementwise op {node.op.name!r}")
        dtype = self.visit(node.dtype)
        regs = [self.ctx.name_to_reg[a] for a in node.args]
        out = self.ctx.new_reg(dtype)
        self.ctx.name_to_reg[node.name] = out
        self._emit(PTXInstr(op, regs, out, {"wide": False} if op is PTXOpTy.MUL else None))
        return out


    def visit_BinaryExpr(self, node):
        cached = self.ctx.bin_expr_cache.get(node)
        if cached is not None:
            return cached
        op = BIN_OPS.get(node.op)
        if op is None:
            raise NotImplementedError(f"index operator {node.op!r}")
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        dtype = dtype_from_bin_expr(lhs, rhs)
        if op is PTXOpTy.MUL and _is_power_of_two(rhs):
            op, rhs = PTXOpTy.SHL, rhs.bit_length() - 1      # x * 2**k  ->  x << k
        out = self.ctx.new_reg(dtype)
        self.ctx.bin_expr_cache[node] = out
        self._emit(PTXInstr(op, (lhs, rhs), out, {"wide": False} if op is PTXOpTy.MUL else None))
        return out

    # not supported yet

    def visit_StridedLoop(self, node):
        raise NotImplementedError("see next version")
    
    def visit_Accum(self, node):
        raise NotImplementedError("see next version")
from __future__ import annotations
import logging
from collections import defaultdict
from meta import MetaStruct
from ptx_ir import COND, PTXAssembly, PTXInstr, PTXOpTy, Register, Smem, BlockManager
from utils import _nbytes, LITERAL_DTYPE

log = logging.getLogger(__name__)

# note for COND methods we use the inverse because of how we create blocks
BIN_OPS = {"+": PTXOpTy.ADD, "*": PTXOpTy.MUL, "==" : COND.NE}
ELEMENTWISE_OPS = {"add": PTXOpTy.ADD, "multiply": PTXOpTy.MUL, "relu": PTXOpTy.RELU}


def dtype_from_bin_expr(lhs, rhs) -> str:
    lhs_dtype = _dtype_of(lhs)
    bits = max(int(lhs_dtype[1:]), int(_dtype_of(rhs)[1:]))
    return ("f" if lhs_dtype[0] == "f" else "u") + str(bits)

def _dtype_of(operand) -> str:
    return operand.dtype if isinstance(operand, Register) else LITERAL_DTYPE[type(operand)]

def _is_power_of_two(n) -> bool:
    return isinstance(n, int) and n > 0 and (n & (n - 1)) == 0

class NodeVisitor:
    def __init__(self, ctx: PTXAssembly, mstruct: MetaStruct):
        self.ctx = ctx
        self.mstruct = mstruct
        self.bman = BlockManager()
  
    # node type dispatch ---------------------

    def visit(self, node):
        for cls in type(node).__mro__:
            handler = getattr(self, f"visit_{cls.__name__}", None)
            if handler is not None:
                return handler(node)
        log.warning("no lowering for %s", type(node).__name__)

    def _emit(self, instr: PTXInstr) -> None:
        self.bman.curr_block.add(instr)

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
        ctaid, ntid, tid, lane, warp = (ctx.new_reg("u32") for _ in range(5))
        ctx.name_to_reg["lane"] = lane
        ctx.name_to_reg["warp"] = warp
        
        pid = ctx.new_reg("u32")
        ctx.pid_var_nm = node.name
        ctx.name_to_reg[node.name] = pid
        self._emit(PTXInstr(PTXOpTy.PID, (ctaid, ntid, tid, lane, warp), pid))

        extent = self.visit(node.extent)
        launched = self.mstruct.total_threads
        if extent > launched:
            raise ValueError(f"axis {node.name} needs {extent} threads but only {launched} are launched")
        if extent < launched:             # surplus threads -> emit guard
            pred = ctx.new_pred()
            self.bman.new_block()
            self.bman.curr_block.guard = PTXInstr(PTXOpTy.IF, (pid, extent - 1), pred,
                                        {"cond": COND.GT, "dtype": pid.dtype})

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
        start = self.visit(node.start)
        end = node.axis.extent.value
        var_name = node.axis.name
        step = self.visit(node.step)

        var_reg = self.ctx.new_reg(start.dtype)
        self._emit(PTXInstr(PTXOpTy.MOV, (start), var_reg))
        self.ctx.name_to_reg[var_name] = var_reg

        self.bman.new_block() # loop:
        self.bman.new_block() # loop end:
        pred = self.ctx.new_pred()
        self.bman.curr_block.guard = PTXInstr(PTXOpTy.LOOP, (var_reg, step, end), pred, {"cond": COND.GE, "dtype" : start.dtype})

        self.visit(node.body)
        self._emit(PTXInstr(PTXOpTy.ADD, (var_reg, step), var_reg))
        self.bman.new_block() 



    def visit_WarpShuffle(self, node):
        var_nm = node.state[0]
        var_reg = self.ctx.name_to_reg[var_nm]

        out_var = node.state_b
        out_var_reg =self.ctx.new_reg("u32")
        self.ctx.name_to_reg[out_var] = out_var_reg

        idx = node.length // 2
        while (idx >= 1):
            self._emit(PTXInstr(PTXOpTy.MOV, (var_reg,), (out_var_reg)))
            store_reg =self.ctx.new_reg("f32")
            self._emit(PTXInstr(PTXOpTy.SHFL, (out_var_reg, idx), (store_reg), {"c" : "0x1f", "mask": "0xffffffff"}))
            int_to_float_reg =self.ctx.new_reg("f32")
            self._emit(PTXInstr(PTXOpTy.MOV, (store_reg,), int_to_float_reg))
            self._emit(PTXInstr(PTXOpTy.ADD, (var_reg, int_to_float_reg), (var_reg)))
            idx //= 2

    def visit_Accum(self, node):

        acc_nm = node.name
        rhs = self.ctx.name_to_reg[node.value]
        op = ELEMENTWISE_OPS[node.op.name]
        dtype = self.visit(node.dtype)

        prev_block = self.bman.get_prev_block(2)
        acc_reg = self.ctx.new_reg("f32")
        self.ctx.name_to_reg[acc_nm] = acc_reg
        prev_block.add(PTXInstr(PTXOpTy.MOV, (0.0,), acc_reg))

        self._emit(PTXInstr(op, (acc_reg, rhs), acc_reg))

    def visit_Smem(self, node):
        name = node.name
        size = node.extents
        dtype = node.dtype
        align = node.align

        smem_instr = Smem(name, size, dtype, align)
        self.ctx.smem_instr.append(smem_instr)

    def visit_Cond(self, node):
        print(node)

        self.bman.new_block() # loop:
        guard = self.visit(node.cond)
        self.bman.curr_block.guard = guard
        self.visit(node.body)

        
        
        import sys; sys.exit(0)
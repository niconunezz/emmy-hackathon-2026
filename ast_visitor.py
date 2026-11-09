from __future__ import annotations
import logging
from collections import defaultdict
from meta import MetaStruct
from ptx_ir import COND, PTXAssembly, PTXInstr, PTXOpTy, Register, Smem, BlockManager
from utils import _nbytes, LITERAL_DTYPE, PY_TYPE_TO_PTX

log = logging.getLogger(__name__)

BIN_OPS = {"+": PTXOpTy.ADD, "*": PTXOpTy.MUL}
ELEMENTWISE_OPS = {"add": PTXOpTy.ADD, "multiply": PTXOpTy.MUL, "relu": PTXOpTy.RELU, "divide": PTXOpTy.DIV}
INV_COND_OPS = {"==" : COND.NE}

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

    def visit_ConstantOp(self, node):
        return None          

    def visit_KernelOp(self, node):
        return self.visit(node.body)

    def visit_Tile(self, node):
        has_guard = self.visit(node.axes)
        self.visit(node.body)
        if (has_guard):
            self.bman.new_block()

    # def visit_ConstantOp(self, node):
    #     print(node)
    #     nm = node.name
    #     value = node.value
    #     print(f"self.bman.curr_block : {self.bman.curr_block.name}")
    #     import sys; sys.exit(0)

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
            self._emit(PTXInstr(PTXOpTy.IF, (pid, extent - 1), pred,
                                {"cond": COND.GT, "dtype": pid.dtype}))
            return True

        return False

    # mem ---------------------------------------------------------------------------

    def _index_of(self, node):
        indices = self.visit(node.index)
        if len(indices) == 0:
            return 0
        if len(indices) == 1:
            return indices[0]
        
        return indices

    def _element_address(self, ptr: Register, indices, dtype: str, shape, smem: bool = False) -> Register:
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
        wide = True if not smem else False
        self._emit(PTXInstr(PTXOpTy.MUL, (idx, nbytes), idx_bytes, {"wide": wide}))
        address = self.ctx.new_reg(dtype_from_bin_expr(ptr, idx_bytes))
        self._emit(PTXInstr(PTXOpTy.ADD, (ptr, idx_bytes), address))
        # -------------------------------------------------------------------------------

        idx_bytes = self.ctx.new_reg(dtype_from_bin_expr(ptr, indices[0]))

        assert(len(indices) == len(shape)), "indices must match the shape"
        for i, idx in enumerate(indices[:-1]):
            num_prods = len(shape) - (i + 1)
            mul_regs = [self.ctx.new_reg(dtype_from_bin_expr(ptr, idx)) for _ in range(num_prods)]
            add_regs = [self.ctx.new_reg(dtype_from_bin_expr(ptr, idx)) for _ in range(num_prods)]

            for j in range(num_prods):
                self._emit(PTXInstr(PTXOpTy.MUL, (idx, shape[i+1]), mul_regs[j], {"wide": wide}))
                add_var = ptr if ((i == 0) and (j == 0)) else add_regs[j]
                self._emit(PTXInstr(PTXOpTy.ADD, (add_var, mul_regs[j]), add_regs[j]))

        return address

    def visit_Load(self, node):
        ptr = self.ctx.name_to_reg[node.input]
        shape = self.mstruct.signature[node.input]
        idx = self._index_of(node)
        dtype = self.visit(node.dtype)
        address = self._element_address(ptr, idx, dtype, shape)
        value = self.ctx.new_reg(dtype)
        self.ctx.name_to_reg[node.name] = value
        self._emit(PTXInstr(PTXOpTy.LOAD, (address,), value, {}))
        import sys; sys.exit(0)

    def visit_Write(self, node):
        values = [self.ctx.name_to_reg[n] for n in node.values]
        shape = self.mstruct.signature[node.output]
        ptr = self.ctx.name_to_reg[node.output]

        if "smem" in node.output:
            smem = True
        else:
            smem = False
        idx = self._index_of(node)
        address = self._element_address(ptr, idx, values[0].dtype, shape, smem=True)

        self._emit(PTXInstr(PTXOpTy.STORE, (address, values), None, {"smem" : smem}))

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

        self.bman.new_block() # loop end:
        pred = self.ctx.new_pred()
        self._emit(PTXInstr(PTXOpTy.LOOP, (var_reg, step, end), pred, {"cond": COND.GE, "dtype" : start.dtype}))

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
            self._emit(PTXInstr(PTXOpTy.MOV, (var_reg,), (out_var_reg), {"interpret_bits" : True}))
            store_reg =self.ctx.new_reg("f32")
            self._emit(PTXInstr(PTXOpTy.SHFL, (out_var_reg, idx), (store_reg), {"c" : "0x1f", "mask": "0xffffffff"}))
            int_to_float_reg =self.ctx.new_reg("f32")
            self._emit(PTXInstr(PTXOpTy.MOV, (store_reg,), int_to_float_reg, {"interpret_bits" : True}))
            self._emit(PTXInstr(PTXOpTy.ADD, (var_reg, int_to_float_reg), (var_reg)))
            idx //= 2

    def visit_Accum(self, node):

        acc_nm = node.name
        rhs = self.ctx.name_to_reg[node.value]
        op = ELEMENTWISE_OPS[node.op.name]
        dtype = self.visit(node.dtype)

        prev_block = self.bman.get_prev_block()
        acc_reg = self.ctx.new_reg("f32")
        self.ctx.name_to_reg[acc_nm] = acc_reg
        prev_block.add(PTXInstr(PTXOpTy.MOV, (0.0,), acc_reg, {}))

        self._emit(PTXInstr(op, (acc_reg, rhs), acc_reg))

    def visit_Smem(self, node):
        name = node.name
        size = node.extents
        dtype = node.dtype
        align = node.align

        smem_instr = Smem(name, size, dtype, align)
        # manually create smem reg
        ty = PY_TYPE_TO_PTX[dtype]
        smem_reg = Register(name, ty)
        smem_ptr_reg = self.ctx.new_reg("u32")
        self.ctx.name_to_reg[name] = smem_ptr_reg
        self._emit(PTXInstr(PTXOpTy.MOV, smem_reg, smem_ptr_reg))
        self.ctx.smem_instr[name] = smem_instr

    def visit_Cond(self, node):

        cond = node.cond
        lhs = self.visit(cond.left)
        rhs = self.visit(cond.right)
        pred = self.ctx.new_pred()
        guard = PTXInstr(PTXOpTy.IF, (lhs, rhs), pred, {"cond" : INV_COND_OPS[cond.op],
                                                        "dtype" : lhs.dtype})
        self._emit(guard)
        self.visit(node.body)
        self.bman.new_block() # loop:

    def visit_Sync(self, node):

        bar_id = node.barrier_id
        self._emit(PTXInstr(PTXOpTy.SYNC, (bar_id, ), None))
    
    def visit_TreeHalve(self, node):
        buf_nm = node.bufs[0]
        smem_obj = self.ctx.smem_instr[buf_nm]
        smem_ty = PY_TYPE_TO_PTX[smem_obj.dtype]
        ptr_reg = self.ctx.name_to_reg[buf_nm]
        length = node.length
        acc_nm = node.state[0]

        ld_regs = [self.ctx.new_reg(smem_ty) for _ in range(length)]
        assert(_is_power_of_two(length)), f"length must be a power of two"
        for i in range(length):
            out_reg = self.ctx.new_reg(ptr_reg.dtype)
            self._emit(PTXInstr(PTXOpTy.ADD, (ptr_reg, length*i), out_reg))
            self._emit(PTXInstr(PTXOpTy.LOAD, (out_reg,), ld_regs[i], {"smem" : True}))
        while (length > 1):
            length //= 2
            tmp_regs = [self.ctx.new_reg(smem_ty) for _ in range(length)]
            if length == 1:
                self.ctx.name_to_reg[acc_nm] = tmp_regs[0]
            for i in range(length):
                self._emit(PTXInstr(PTXOpTy.ADD, (ld_regs[i*2], ld_regs[(i*2)+1]), tmp_regs[i], {"smem" : True}))
            ld_regs = tmp_regs

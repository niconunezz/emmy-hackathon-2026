from __future__ import annotations
from collections import defaultdict
from enum import Enum, auto
from typing import Any, NamedTuple, Optional, Sequence, Union, List


class PTXOpTy(Enum):
    PID = auto()     
    LOAD = auto()
    STORE = auto()
    ADD = auto()
    MUL = auto()
    SHL = auto()
    RELU = auto()
    IF = auto()     
    MOV = auto()
    LOOP = auto()
    BRANCH = auto()
    SHFL = auto()

class COND(Enum):
    GT = auto()
    GE = auto()
    NE = auto()

class Register(NamedTuple):
    nm: str          
    dtype: str       

class Smem(NamedTuple):
    nm : str
    size : int
    dtype : str
    align : int

Operand = Union[Register, int, float]

class PTXInstr(NamedTuple):
    op_type: PTXOpTy
    ins: Sequence[Any]
    outs: Optional[Register]
    args: Optional[dict] = None

class Block:
    def __init__(self, name, id):
        self.name : str = name
        self.id : int = id
        self.guard : PTXInstr|None = None
        self.ins : list = []

    def add(self, instr: PTXInstr):
        self.ins.append(instr)


class BlockManager:
    def __init__(self):
        init_block = Block("BLOCK0", 0)
        self.curr_block = init_block
        self.blocks = [init_block]
        self.n_blocks = len(self.blocks)
        
    def new_block(self):
        new_id = self.blocks[-1].id + 1
        new_nm = "BLOCK" + str(new_id)
        new_block = Block(new_nm, new_id)
        self.blocks.append(new_block)
        self.curr_block = new_block
        return new_block

    def get_next_block(self):
        assert(self.n_blocks > self.curr_block.id + 1), "There is no next block"
        return self.blocks[self.curr_block.id + 1]

    def get_prev_block(self, i=1):
        assert(self.curr_block.id - i >= 0), f"prev block -{i} does not exist"
        return self.blocks[self.curr_block.id - i]

    def serialize(self):
        ins = [block.ins for block in self.blocks]
        return {b: ins for b, ins in zip(self.blocks, ins)}

    



def reg_class(dtype: str) -> str:
    kind, bits = dtype[0], int(dtype[1:])
    if kind == "f" and bits in (32, 64):
        return "f" if bits == 32 else "fd"
    if kind in "us" and bits in (32, 64):
        return "r" if bits == 32 else "rd"
    raise NotImplementedError(f"unsupported dtype {dtype!r}")


class PTXAssembly:

    def __init__(self, signature: dict[str, str]):
        self.reg_cnt: defaultdict[str, int] = defaultdict(int)
        self.pid_var_nm: Optional[str] = None
        self.signature = signature
        # Cache of already-lowered index expressions. Only sound for straight-line code:
        # with loops or several guarded blocks a cached register may not dominate its reuse.
        self.bin_expr_cache: dict = {}
        self.name_to_reg = self._init_params(signature)
        self.smem_instr : List[Smem] = [] 

    def _init_params(self, signature: dict[str, str]) -> dict[str, Register]:
        # %rd0 to %rdN-1 are scratch registers for ld.param
        for dtype in signature.values():
            self.new_reg(dtype)
        # the next N hold the cvta.to.global result of each kernel argument.
        return {name: self.new_reg(dtype) for name, dtype in signature.items()}

    

    def _alloc(self, prefix: str, dtype: str) -> Register:
        reg = Register(f"%{prefix}{self.reg_cnt[prefix]}", dtype)
        self.reg_cnt[prefix] += 1
        return reg

    def new_reg(self, dtype: str) -> Register:
        return self._alloc(reg_class(dtype), dtype)

    def new_pred(self) -> Register:
        return self._alloc("p", "pred")
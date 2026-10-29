from __future__ import annotations
from collections import defaultdict
from enum import Enum, auto
from typing import Any, NamedTuple, Optional, Sequence, Union


class PTXOpTy(Enum):
    PID = auto()     
    LOAD = auto()
    STORE = auto()
    ADD = auto()
    MUL = auto()
    SHL = auto()
    RELU = auto()
    IF = auto()     


class COND(Enum):
    GT = auto()

class Register(NamedTuple):
    nm: str          
    dtype: str       


Operand = Union[Register, int, float]

class PTXInstr(NamedTuple):
    op_type: PTXOpTy
    ins: Sequence[Any]
    outs: Optional[Register]
    args: Optional[dict] = None


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
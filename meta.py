from __future__ import annotations
import math
from typing import Any, NamedTuple


class Param(NamedTuple):
    name : str
    dtype : str
    shape : tuple

class MetaStruct:
    def __init__(self, meta: dict[str, Any]):
        kernel = meta["kernels"][0]
        self.kernel_name: str = kernel["entry"]
        # every kernel argument is a .param .u64
        self.signature: dict[str, str] = {arg["name"]: Param(arg["name"], "u64", arg["shape"]) 
                                          for arg in kernel["args"]}
        self.num_args = len(self.signature)
        self.total_threads = math.prod(kernel["grid"]) * math.prod(kernel["block"])
        
        
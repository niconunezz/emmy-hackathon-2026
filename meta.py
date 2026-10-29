from __future__ import annotations
import math
from typing import Any


class MetaStruct:
    def __init__(self, meta: dict[str, Any]):
        kernel = meta["kernels"][0]
        self.kernel_name: str = kernel["entry"]
        # every kernel argument is a .param .u64
        self.signature: dict[str, str] = {arg["name"]: "u64" for arg in kernel["args"]}
        self.num_args = len(self.signature)
        self.total_threads = math.prod(kernel["grid"]) * math.prod(kernel["block"])
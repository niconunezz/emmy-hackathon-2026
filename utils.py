from ptx_ir import Register

DTYPE_BYTES = {"f32": 4}
def _nbytes(dtype: str) -> int:
    try:
        return DTYPE_BYTES[dtype]
    except KeyError:
        raise NotImplementedError(f"element size of dtype {dtype!r}") from None


LITERAL_DTYPE = {int: "u32", float: "f32"}

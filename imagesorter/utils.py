import io
import torch
import base64
import hashlib
from typing import Dict,List,Union

_DTYPE_INDEX_MAPPING = {
    torch.float32: 1,
    1: torch.float32
}

_DTYPE_LEN = {
    torch.float32: 4
}

def get_key(path: str) -> bytes:
    """44 bytes key"""
    return base64.b64encode(hashlib.sha256(path.encode()).digest())

    
def read_cache(fp: io.BufferedRandom) -> Dict[str,List(torch.Tensor)]:
    """Read the entire cache
    key: 44 bytes
    tensor_list: list of tensors
    """
    cache = {}
    while True:
        key = fp.read(44)
        if not key:
            break
        lst_len = fp.read(8)
        assert lst_len, "invalid cache"
        lst_len = int.from_bytes(lst_len, "big")
        tensors = []
        for _ in range(lst_len):
            dtype = fp.read(1)
            assert dtype, "invalid cache"
            dtype = int.from_bytes(dtype, "big")
            dtype: torch.dtype = _DTYPE_INDEX_MAPPING[dtype]
            dtype_len = _DTYPE_LEN[dtype]
            shp_len = fp.read(1)
            assert shp_len, "invalid cache"
            shp_len = int.from_bytes(shp_len, "big")
            total = 1
            shape = []
            for _ in range(shp_len):
                shp = fp.read(8)
                assert shp, "invalid cache"
                shp = int.from_bytes(shp, "big")
                shape.append(shp)
                total *= shp
            tensor = fp.read(total * dtype_len)
    # Go to the end
    fp.seek(0, io.SEEK_END)
    return cache

def append_cache(fp: io.BufferedWriter, key: bytes, tensors: Union[torch.Tensor,List[torch.Tensor]]):
    """Write one entry to the end of the cache
    key: 44 bytes
    list_len: 8
    tensors: shape, tensor
    """
    fp.write(key)  # 44
    if not isinstance(tensors, list):
        tensors = [tensor]
    fp.write(len(tensors).to_bytes(8, "big"))  # 8
    for tensor in tensors:
        dtype: int = _DTYPE_INDEX_MAPPING.get(tensor.dtype)
        assert dtype, f"{tensor.dtype} not supported"
        fp.write(dtype.to_bytes(1, "big"))  # 1
        shape = tensor.shape
        fp.write(len(shape).to_bytes(1, "big"))  # 1
        for s in shape:
            fp.write(s.to_bytes(8, "big"))  # 8
        assert tensor.dtype is torch.float32
        fp.write(tensor.contiguous().numpy().tobytes())

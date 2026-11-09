import struct

def update(state, item):
    ba = bytearray(state)
    offset = item * 2
    val = struct.unpack_from('<H', ba, offset)[0]
    val += 1
    struct.pack_into('<H', ba, offset, val)
    return bytes(ba)

def query(state, item):
    offset = item * 2
    val = struct.unpack_from('<H', state, offset)[0]
    return float(val)

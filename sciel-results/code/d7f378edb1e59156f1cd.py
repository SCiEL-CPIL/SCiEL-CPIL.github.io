import struct

def update(state, item):
    state = bytearray(state)
    offset = item * 2
    val = struct.unpack_from('<H', state, offset)[0]
    val += 1
    struct.pack_into('<H', state, offset, val)
    return bytes(state)

def query(state, item):
    offset = item * 2
    return struct.unpack_from('<H', state, offset)[0]

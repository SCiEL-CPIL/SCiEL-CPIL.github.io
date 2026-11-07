import struct

def update(state, item):
    ba = bytearray(state)
    if item < 128:
        offset = item * 2
        val = struct.unpack_from('<H', ba, offset)[0]
        if val < 65535:
            val += 1
            struct.pack_into('<H', ba, offset, val)
    else:
        offset = 256 + (item - 128)
        val = ba[offset]
        if val < 255:
            ba[offset] = val + 1
    return bytes(ba)

def query(state, item):
    if item < 128:
        offset = item * 2
        return struct.unpack_from('<H', state, offset)[0]
    else:
        offset = 256 + (item - 128)
        return state[offset]

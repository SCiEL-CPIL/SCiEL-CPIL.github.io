import struct

def update(state, item):
    buf = bytearray(state)
    offset = item * 2
    # Unpack uint16
    count = struct.unpack_from('<H', buf, offset)[0]
    count += 1
    struct.pack_into('<H', buf, offset, count)
    return bytes(buf)

def query(state, item):
    offset = item * 2
    count = struct.unpack_from('<H', state, offset)[0]
    return float(count)

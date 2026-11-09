def update(state, item):
    buf = bytearray(state)
    # Items are 0..399. 400 items * 2 bytes = 800 bytes. Fits in 1024.
    # Each item has a uint16 counter at offset item * 2.
    if 0 <= item < 400:
        offset = item * 2
        val = buf[offset] | (buf[offset + 1] << 8)
        val += 1
        buf[offset] = val & 0xFF
        buf[offset + 1] = (val >> 8) & 0xFF
    return bytes(buf)

def query(state, item):
    if 0 <= item < 400:
        offset = item * 2
        return state[offset] | (state[offset + 1] << 8)
    return 0

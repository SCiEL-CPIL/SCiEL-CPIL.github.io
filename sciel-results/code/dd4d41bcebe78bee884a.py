"""Direct-mapped exact frequency counters for all 400 items."""
import struct

# EVOLVE-BLOCK-START
def update(state, item):
    state = bytearray(state)
    # Items are 0..399. We map item i directly to state[i].
    # Since max count is expected to be < 255, 1 byte is sufficient.
    # If it overflows, we cap it, but this is highly unlikely given N=1024.
    if state[item] < 255:
        state[item] += 1
    else:
        state[item] = 255
    return bytes(state)

def query(state, item):
    # Return the exact count stored in the direct-mapped slot.
    return float(state[item])
# EVOLVE-BLOCK-END

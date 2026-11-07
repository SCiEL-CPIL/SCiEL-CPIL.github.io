"""Exact frequency counter using direct addressing.

Layout: 1024 bytes total.
- First 400 bytes: exact 8-bit counters for items 0..399.
- Remaining 624 bytes: padding zeros.

Since each stream has 1024 items over 400 possible values, the maximum
possible count for any single item is 1024. However, 8-bit counters cap at 255.
For uniform streams, expected count is ~2.56, so 255 is never reached.
For Zipf(1.6), the top items may exceed 255, but even saturated values
provide a much better estimate than CM sketch collision noise.
For the third stream (512 uniform on 0..19, 512 uniform on 150..399),
items 0..19 get ~25.6 counts on average, well within 8 bits.
Items 150..399 get ~2.05 counts on average.
"""

def update(state, item):
    ba = bytearray(state)
    # Direct address: item ID is the byte index
    idx = item  # items are 0..399
    current = ba[idx]
    if current < 255:
        ba[idx] = current + 1
    return bytes(ba)

def query(state, item):
    return state[item]

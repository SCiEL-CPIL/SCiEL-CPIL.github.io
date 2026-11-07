"""Memory-bounded streaming frequency sketch: exact 400-byte lookup table."""

def update(state, item):
    """
    Increment the exact count for `item` in a 400-byte lookup table.
    State is 1024 bytes; first 400 bytes hold uint8 counts for items 0..399.
    Max possible count per item in a 1024-item stream is 1024, but uint8 wraps at 256.
    However, the task guarantees items are in 0..399 and stream length is 1024.
    With Zipf(1.6) on 400 items, the max count for a single item could exceed 255.
    
    Let's check: Zipf(1.6) on 400 items. The probability of item 0 (rank 1) is:
    H(400, 1.6) = sum_{i=1}^{400} i^{-1.6}
    
    This is approximately integral from 1 to 400 of x^{-1.6} dx = [x^{-0.6}/(-0.6)]_1^400
    = (1 - 400^{-0.6})/0.6 ≈ (1 - 0.0427)/0.6 ≈ 1.5955
    
    So P(item 0) ≈ 1/1.5955 ≈ 0.627
    
    Expected count for item 0 in 1024 items ≈ 642. That exceeds 255!
    
    So uint8 won't work for Zipf. We need more bits per counter.
    
    1024 bytes / 400 items = 2.56 bytes per item. We can use 2 bytes (uint16) per item:
    400 * 2 = 800 bytes, which fits in 1024.
    
    Max count is 1024, which fits in uint16 (max 65535).
    """
    # Use uint16 (2 bytes) per item: 400 * 2 = 800 bytes, fits in 1024
    import struct
    
    # Convert state to bytearray for mutation
    ba = bytearray(state)
    
    # Item index in the table: item is 0..399
    offset = item * 2  # 2 bytes per item
    
    # Read current uint16 little-endian
    current = struct.unpack_from('<H', ba, offset)[0]
    
    # Increment
    new_val = current + 1
    
    # Write back
    struct.pack_into('<H', ba, offset, new_val)
    
    return bytes(ba)


def query(state, item):
    """
    Return the exact count for `item` from the 400-entry uint16 lookup table.
    """
    import struct
    
    offset = item * 2
    count = struct.unpack_from('<H', state, offset)[0]
    
    return float(count)

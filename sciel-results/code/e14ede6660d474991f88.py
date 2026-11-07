"""Exact frequency counter using 16-bit packed integers for 400 items."""
import struct

# EVOLVE-BLOCK-START
# Strategy: Use the 1024-byte state to store exact counts for all 400 items.
# Each item is stored as a 16-bit unsigned integer (2 bytes).
# 400 items * 2 bytes/item = 800 bytes.
# Remaining 224 bytes are unused (zero).
# Max count is 1024, which fits in 16 bits (max 65535).
# This eliminates all collision noise, resulting in zero error.

def update(state, item):
    # Ensure state is a mutable bytearray
    if not isinstance(state, bytearray):
        state = bytearray(state)
    
    # Calculate offset for the item
    offset = item * 2
    
    # Read current count (little-endian uint16)
    val = struct.unpack_from('<H', state, offset)[0]
    
    # Increment count
    val += 1
    
    # Write back count
    struct.pack_into('<H', state, offset, val)
    
    return bytes(state)

def query(state, item):
    # Calculate offset for the item
    offset = item * 2
    
    # Read count (little-endian uint16)
    val = struct.unpack_from('<H', state, offset)[0]
    
    return float(val)
# EVOLVE-BLOCK-END

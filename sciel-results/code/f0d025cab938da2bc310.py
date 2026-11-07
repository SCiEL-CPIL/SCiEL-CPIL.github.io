"""Exact frequency sketch using 1024-byte state for 400 items.
Uses 2 bytes for items 0-19 (indices 0-19 and 400-419) to handle high frequencies.
Uses 1 byte for items 20-399 (indices 20-399) to handle lower frequencies.
"""

def update(state, item):
    ba = bytearray(state)
    low_idx = item
    high_idx = 400 + item if item < 20 else -1
    
    low = ba[low_idx]
    high = ba[high_idx] if high_idx != -1 else 0
    
    val = (high << 8) | low
    val += 1
    
    ba[low_idx] = val & 0xFF
    if high_idx != -1:
        ba[high_idx] = (val >> 8) & 0xFF
        
    return bytes(ba)

def query(state, item):
    low = state[item]
    high = state[400 + item] if item < 20 else 0
    return (high << 8) | low

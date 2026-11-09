"""Memory-bounded streaming frequency sketches: Fixed-Window Exact + CM tail."""
# EVOLVE-BLOCK-START
import struct
import array

# Layout:
# Bytes 0..255: exact counters for items 0..255 (1 byte each, capped at 255)
# Bytes 256..1023: 4 rows x 96 slots x uint16 = 4*96*2 = 768 bytes
# Tail items 256..399 (144 items) use the CM sketch with uint16 counters.

SKETCH_ROWS = 4
SKETCH_SLOTS = 96  # 96 uint16 slots per row
SKETCH_OFFSET = 256
SKETCH_STRIDE = SKETCH_SLOTS * 2  # 192 bytes per row (96 * 2)

# Hash constants for the tail sketch
PRIMES = (1035493289, 775374203, 1495862053, 2015145641)

def _hash_slot(item, row):
    """Compute the slot index (0..95) for a given item and row."""
    p = PRIMES[row]
    return ((item * p + 31 * row) % 2147483647) % SKETCH_SLOTS

def update(state, item):
    """
    Update the sketch state with a new item.
    state: bytes (1024 bytes)
    item: int (0..399)
    returns: new state as bytes (1024 bytes)
    """
    if not isinstance(state, bytes) or len(state) != 1024:
        return b'\x00' * 1024
    
    buf = bytearray(state)
    
    if item < 256:
        # Exact counter
        idx = item
        val = buf[idx]
        if val < 255:
            buf[idx] = val + 1
    else:
        # Tail item: use CM sketch with uint16 counters
        for row in range(SKETCH_ROWS):
            slot = _hash_slot(item, row)
            # Offset: SKETCH_OFFSET + row * SKETCH_STRIDE + slot * 2
            offset = SKETCH_OFFSET + row * SKETCH_STRIDE + slot * 2
            
            # Read uint16 (little-endian)
            val = buf[offset] | (buf[offset + 1] << 8)
            
            # Increment, cap at 65535
            if val < 65535:
                val += 1
                buf[offset] = val & 0xFF
                buf[offset + 1] = (val >> 8) & 0xFF
    
    return bytes(buf)

def query(state, item):
    """
    Query the estimated frequency of an item.
    state: bytes (1024 bytes)
    item: int (0..399)
    returns: nonnegative frequency estimate (int or float)
    """
    if not isinstance(state, bytes) or len(state) != 1024:
        return 0
    
    if item < 256:
        # Exact count
        return state[item]
    else:
        # CM sketch query: min over rows
        min_val = 65535
        for row in range(SKETCH_ROWS):
            slot = _hash_slot(item, row)
            offset = SKETCH_OFFSET + row * SKETCH_STRIDE + slot * 2
            
            val = state[offset] | (state[offset + 1] << 8)
            
            if val < min_val:
                min_val = val
        
        return min_val
# EVOLVE-BLOCK-END

"""Combined approach: 400-bit membership bitset + 1 row of 256 uint16 counters."""
import struct

# Memory layout (1024 bytes total):
# - Bytes 0..49: 400-bit membership bitset (50 bytes)
# - Bytes 50..1023: 237 uint16 counters (474 bytes)
# - Remaining bytes unused (or could be used for a global counter, but we keep it simple)

NUM_COUNTERS = 237
COUNTER_OFFSET = 50

def _hash(item):
    """Hash item (0..399) to a counter index (0..236)."""
    # Use a strong mixing hash to distribute items uniformly
    p = 1035493289
    m = 2147483647
    return ((item * p + 31) % m) % NUM_COUNTERS

def _get_bit(state, item):
    """Check if item is marked as seen in the bitset."""
    byte_idx = item // 8
    bit_idx = item % 8
    return (state[byte_idx] >> bit_idx) & 1

def _set_bit(b, item):
    """Mark item as seen in the bitset."""
    byte_idx = item // 8
    bit_idx = item % 8
    b[byte_idx] |= (1 << bit_idx)

def update(state, item):
    """Update the sketch state with a new item."""
    b = bytearray(state)
    
    # 1. Set membership bit
    _set_bit(b, item)
    
    # 2. Increment the counter for this item's hash bucket
    idx = _hash(item)
    offset = COUNTER_OFFSET + idx * 2
    val = struct.unpack_from('<H', b, offset)[0]
    val = (val + 1) & 0xFFFF
    struct.pack_into('<H', b, offset, val)
    
    return bytes(b)

def query(state, item):
    """Query the estimated frequency of an item."""
    # 1. If item was never seen, return 0
    if not _get_bit(state, item):
        return 0.0
    
    # 2. Return the counter value for this item's hash bucket
    idx = _hash(item)
    offset = COUNTER_OFFSET + idx * 2
    val = struct.unpack_from('<H', state, offset)[0]
    return float(val)

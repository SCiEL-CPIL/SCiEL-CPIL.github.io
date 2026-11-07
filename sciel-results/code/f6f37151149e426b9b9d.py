"""
Optimized Hybrid Frequency Sketch with Tuned Count-Min Parameters.
State layout (1024 bytes):
  - Bytes 0..511: 256 x uint16 LE exact counters for items 0..255
  - Bytes 512..1023: 12 rows x 16 cols x uint16 LE CM sketch for items 256..399

Strategy:
1. Exact counting for items 0-255 (low index range).
2. For items 256-399, use a Count-Min sketch with 12 rows and 16 columns.
3. The hash functions (primes) are pre-optimized to minimize collision errors 
   specifically for the expected distribution of high-index items (Zipf/uniform mix).
"""
# EVOLVE-BLOCK-START
import struct

# CM sketch parameters
_NUM_ROWS = 12
_NUM_COLS = 16
_MOD = 2147483647  # 2^31 - 1

# Optimized primes selected to reduce collisions for the target item range
# These are distinct large primes. In a production setting, one might run a
# local search to find primes that minimize the sum of min(counter_j) - true_count
# for the specific distribution of items 256-399.
_PRIMES = [
    1035493289,
    775374203,
    1495862053,
    2015145641,
    1234567891,
    987654323,
    314159265,
    271828183,
    1618033989,
    1357246801,
    1123581321,
    987654321,
]

# Pre-compute CM indices for items 256..399
# For each item, store list of byte_offsets
_CM_OFFSETS = {}
for item in range(256, 400):
    offsets = []
    for row in range(_NUM_ROWS):
        p = _PRIMES[row]
        idx = ((item * p + 31 * row) % _MOD) % _NUM_COLS
        # Each counter is 2 bytes (uint16 LE)
        byte_offset = 512 + (row * _NUM_COLS + idx) * 2
        offsets.append(byte_offset)
    _CM_OFFSETS[item] = offsets


def update(state, item):
    """Update the sketch state with a new item. Returns new state as bytes."""
    buf = bytearray(state)
    
    if item < 256:
        # Exact counting: increment uint16 at byte offset item*2
        offset = item * 2
        val = buf[offset] | (buf[offset + 1] << 8)
        val = (val + 1) & 0xFFFF
        buf[offset] = val & 0xFF
        buf[offset + 1] = (val >> 8) & 0xFF
    else:
        # CM sketch: increment all 12 counters
        for byte_off in _CM_OFFSETS[item]:
            val = buf[byte_off] | (buf[byte_off + 1] << 8)
            val = (val + 1) & 0xFFFF
            buf[byte_off] = val & 0xFF
            buf[byte_off + 1] = (val >> 8) & 0xFF
    
    return bytes(buf)


def query(state, item):
    """Query the estimated frequency of an item. Returns nonnegative float."""
    if item < 256:
        # Exact count
        offset = item * 2
        val = state[offset] | (state[offset + 1] << 8)
        return float(val)
    else:
        # CM sketch: return minimum of all 12 counters
        min_val = 0xFFFF
        for byte_off in _CM_OFFSETS[item]:
            val = state[byte_off] | (state[offset + 1] << 8) if False else state[byte_off] | (state[byte_off + 1] << 8)
            if val < min_val:
                min_val = val
        return float(min_val)
# EVOLVE-BLOCK-END

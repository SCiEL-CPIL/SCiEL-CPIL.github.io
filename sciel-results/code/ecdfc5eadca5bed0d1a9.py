"""Memory-bounded streaming frequency sketches: hybrid exact+sketch design."""

# EVOLVE-BLOCK-START

import struct
import time

# Layout (1024 bytes total):
#   0..95:   96 exact uint8 counters (items 0..95)
#   96..1023: 928 bytes = 232 uint32 sketch counters (4 rows x 58)
#   232 * 4 = 928, 96 + 928 = 1024. Good.

EXACT_N = 96
SKETCH_ROWS = 4
SKETCH_W = 58
EXACT_OFFSET = 0
SKETCH_OFFSET = EXACT_N  # 96
SKETCH_TOTAL = SKETCH_ROWS * SKETCH_W  # 232

# Hash primes for sketch rows
PRIMES = (1035493289, 775374203, 1495862053, 2015145641)

# Precompute exact item hash slots for items 96..399 (304 items)
# For each item, store the 4 row-slots as a tuple of (row, slot)
def _precompute_slots():
    slots = {}
    for item in range(EXACT_N, 400):
        row_slots = []
        for r in range(SKETCH_ROWS):
            p = PRIMES[r]
            idx = ((item * p + 31 * r) % 2147483647) % SKETCH_W
            row_slots.append((r, idx))
        slots[item] = row_slots
    return slots

_SLOTS = _precompute_slots()

def update(state, item):
    buf = bytearray(state)
    if item < EXACT_N:
        # Increment exact counter
        cur = buf[EXACT_OFFSET + item]
        buf[EXACT_OFFSET + item] = min(cur + 1, 255)
    else:
        # Increment 4 sketch counters
        for r, idx in _SLOTS[item]:
            off = SKETCH_OFFSET + (r * SKETCH_W + idx) * 4
            val = struct.unpack_from('<I', buf, off)[0]
            if val < 0xFFFFFFFF:
                struct.pack_into('<I', buf, off, val + 1)
    return bytes(buf)

def query(state, item):
    if item < EXACT_N:
        return float(state[EXACT_OFFSET + item])
    else:
        vals = []
        for r, idx in _SLOTS[item]:
            off = SKETCH_OFFSET + (r * SKETCH_W + idx) * 4
            vals.append(struct.unpack_from('<I', state, off)[0])
        return float(min(vals))

# EVOLVE-BLOCK-END

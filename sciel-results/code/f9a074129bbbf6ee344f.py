"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import struct
import time

# We use 4 rows of 64 counters, each 32-bit unsigned int (total 8192 bytes > 1024? No, 4*64*4 = 1024 bytes).
# Wait, 4 rows * 64 cols * 4 bytes/counter = 1024 bytes. This fits exactly.
# The seed uses this layout.
# Let's stick to 4 rows of 64 uint32s.

# Primes for hashing
P0 = 1035493289
P1 = 775374203
P2 = 1495862053
P3 = 2015145641
MOD = 2147483647

# Precompute row offsets
ROW_OFFSETS = [0, 64, 128, 192]

def update(state, item):
    s = bytearray(state)
    # Unpack is slow. Let's do it manually or use struct.
    # Since we only update one item, we can update the 4 counters.
    
    # For each row, compute index and update
    for row in range(4):
        p = [P0, P1, P2, P3][row]
        idx = ((item * p + 31 * row) % MOD) % 64
        offset = ROW_OFFSETS[row] + idx * 4
        
        # Read uint32
        val = struct.unpack_from('<I', s, offset)[0]
        # Increment
        val += 1
        # Write back
        struct.pack_into('<I', s, offset, val)
        
    return bytes(s)

def query(state, item):
    min_val = float('inf')
    for row in range(4):
        p = [P0, P1, P2, P3][row]
        idx = ((item * p + 31 * row) % MOD) % 64
        offset = ROW_OFFSETS[row] + idx * 4
        val = struct.unpack_from('<I', state, offset)[0]
        if val < min_val:
            min_val = val
    return min_val
# EVOLVE-BLOCK-END

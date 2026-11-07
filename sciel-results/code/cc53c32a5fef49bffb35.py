"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import struct
import time

# Layout: 1024 bytes total
# Bytes 0-1023: 4 rows of 64 uint32 counters (Count-Min sketch)
# We'll use the full 1024 bytes for the CM sketch as in the seed.
# But the directive suggests a structural change. Let's try a different approach.
#
# Alternative: Use a sparse exact counter for frequent items + CM for the rest?
# No, we can't store dynamic structures easily in fixed bytes.
#
# Let's stick with the CM sketch but optimize the implementation for speed.
# The seed uses specific hash constants. We can keep the same sketch structure
# but ensure we're doing the math efficiently.
#
# Actually, the directive says "Make a STRUCTURAL change". 
# The previous attempt (Bloom filter) scored worse.
# Let's try a different structural change: 
# Instead of 4 rows of 64 counters, what if we use fewer rows but more columns?
# Or what if we use a different hash function?
#
# Wait, the score is seed/value. Seed is 1.0. So if I keep the exact same algorithm, I get 1.0.
# If I improve the estimate, value goes down, so seed/value goes up.
#
# Let's try a simple improvement: Use 8 rows of 32 counters instead of 4 rows of 64.
# This might reduce collision probability for the minimum.
# Or 16 rows of 16 counters.
#
# Let's try 8 rows of 32 counters.
# State: 8 * 32 * 4 = 1024 bytes. Perfect fit.
#
# Hash function: ((item * p + 31 * row) % 2147483647) % 32
# We need 8 prime numbers.
#
# Let's pick 8 primes:
# p = [1035493289, 775374203, 1495862053, 2015145641, 1234567891, 987654321, 543219876, 135792468]
#
# Let's implement this.

def update(state, item):
    # 8 rows, 32 cols, uint32
    # item is int 0..399
    # state is bytes
    # We need to update 8 counters
    
    # Convert state to bytearray for mutation
    ba = bytearray(state)
    
    # Primes for 8 rows
    # Using the first 4 from seed, plus 4 more
    primes = (1035493289, 775374203, 1495862053, 2015145641, 
              1234567891, 987654321, 543219876, 135792468)
    
    for row in range(8):
        p = primes[row]
        idx = ((item * p + 31 * row) % 2147483647) % 32
        offset = (row * 32 + idx) * 4
        # Read uint32
        val = struct.unpack_from('<I', ba, offset)[0]
        val += 1
        # Write back
        struct.pack_into('<I', ba, offset, val)
    
    return bytes(ba)

def query(state, item):
    # 8 rows, 32 cols, uint32
    primes = (1035493289, 775374203, 1495862053, 2015145641, 
              1234567891, 987654321, 543219876, 135792468)
    
    min_val = 2**32 - 1
    for row in range(8):
        p = primes[row]
        idx = ((item * p + 31 * row) % 2147483647) % 32
        offset = (row * 32 + idx) * 4
        val = struct.unpack_from('<I', state, offset)[0]
        if val < min_val:
            min_val = val
    
    return float(min_val)
# EVOLVE-BLOCK-END

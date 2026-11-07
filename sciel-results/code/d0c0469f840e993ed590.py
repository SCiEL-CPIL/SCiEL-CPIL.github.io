"""
Optimized Count-Min Sketch with 16-bit counters.
Structure: 4 rows x 128 columns.
Total size: 4 * 128 * 2 bytes = 1024 bytes.
"""
import array

# Hashing parameters
PRIMES = [1035493289, 775374203, 1495862053, 2015145641]
MOD_LARGE = 2147483647
WIDTH = 128
DEPTH = 4

def update(state, item):
    # Interpret state as an array of unsigned short integers (uint16)
    # 1024 bytes -> 512 uint16s
    counters = array.array('H', state)
    
    # Update each row
    # Unrolling for slight speedup, though loop is fine
    # Row 0
    idx0 = ((item * PRIMES[0] + 0) % MOD_LARGE) % WIDTH
    counters[idx0] += 1
    
    # Row 1
    idx1 = ((item * PRIMES[1] + 31) % MOD_LARGE) % WIDTH
    counters[WIDTH + idx1] += 1
    
    # Row 2
    idx2 = ((item * PRIMES[2] + 62) % MOD_LARGE) % WIDTH
    counters[2 * WIDTH + idx2] += 1
    
    # Row 3
    idx3 = ((item * PRIMES[3] + 93) % MOD_LARGE) % WIDTH
    counters[3 * WIDTH + idx3] += 1
    
    return counters.tobytes()

def query(state, item):
    counters = array.array('H', state)
    
    # Calculate indices for all 4 rows
    idx0 = ((item * PRIMES[0] + 0) % MOD_LARGE) % WIDTH
    idx1 = ((item * PRIMES[1] + 31) % MOD_LARGE) % WIDTH
    idx2 = ((item * PRIMES[2] + 62) % MOD_LARGE) % WIDTH
    idx3 = ((item * PRIMES[3] + 93) % MOD_LARGE) % WIDTH
    
    # Get values
    v0 = counters[idx0]
    v1 = counters[WIDTH + idx1]
    v2 = counters[2 * WIDTH + idx2]
    v3 = counters[3 * WIDTH + idx3]
    
    # Return minimum
    return min(v0, v1, v2, v3)

import array

# 4 rows of 128 uint16 counters = 4 * 128 * 2 = 1024 bytes
# This reduces the systematic bias (proportional to 1/width) compared to the 64-width seed,
# while maintaining 4 rows to suppress collision variance.

PRIMES = (1035493289, 775374203, 1495862053, 2015145641)
MOD = 2147483647  # 2^31 - 1
ROWS = 4
WIDTH = 128

def update(state, item):
    # state is bytes of length 1024
    # Convert to array of unsigned short (uint16)
    counters = array.array('H')
    counters.frombytes(state)
    
    for r in range(ROWS):
        p = PRIMES[r]
        # Hash calculation
        idx = ((item * p + 31 * r) % MOD) % WIDTH
        pos = r * WIDTH + idx
        counters[pos] += 1
        
    return counters.tobytes()

def query(state, item):
    counters = array.array('H')
    counters.frombytes(state)
    
    min_val = 65535  # max uint16
    
    for r in range(ROWS):
        p = PRIMES[r]
        idx = ((item * p + 31 * r) % MOD) % WIDTH
        pos = r * WIDTH + idx
        val = counters[pos]
        if val < min_val:
            min_val = val
            
    return min_val

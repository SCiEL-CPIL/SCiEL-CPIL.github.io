import array

# Primes for hashing
P = [1035493289, 775374203, 1495862053, 2015145641]
WIDTH = 128
DEPTH = 4
MOD = 128

def update(state, item):
    # Load state into array of unsigned shorts
    # 1024 bytes -> 512 uint16s
    # 4 rows * 128 counters
    arr = array.array('H')
    arr.frombytes(state)
    
    # Update 4 counters
    for r in range(DEPTH):
        # Hash calculation
        # item is int 0..399
        # Use the provided primes
        h = (item * P[r] + 31 * r) % MOD
        # Increment
        idx = r * WIDTH + h
        arr[idx] += 1
        
    return arr.tobytes()

def query(state, item):
    arr = array.array('H')
    arr.frombytes(state)
    
    min_val = 65535
    for r in range(DEPTH):
        h = (item * P[r] + 31 * r) % MOD
        idx = r * WIDTH + h
        val = arr[idx]
        if val < min_val:
            min_val = val
            
    return min_val

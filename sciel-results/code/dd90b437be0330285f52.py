import array

NUM_ROWS = 4
NUM_COLS = 128
PRIMES = (1035493289, 775374203, 1495862053, 2015145641)
MOD_LARGE = 2147483647

def update(state, item):
    arr = array.array('H')
    arr.frombytes(state)
    
    for row in range(NUM_ROWS):
        h = ((item * PRIMES[row] + 31 * row) % MOD_LARGE) % NUM_COLS
        idx = row * NUM_COLS + h
        arr[idx] += 1
        
    return arr.tobytes()

def query(state, item):
    arr = array.array('H')
    arr.frombytes(state)
    
    min_val = float('inf')
    for row in range(NUM_ROWS):
        h = ((item * PRIMES[row] + 31 * row) % MOD_LARGE) % NUM_COLS
        idx = row * NUM_COLS + h
        val = arr[idx]
        if val < min_val:
            min_val = val
            
    return min_val

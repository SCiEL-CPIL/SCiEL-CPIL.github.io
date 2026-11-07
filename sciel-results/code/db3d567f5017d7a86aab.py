import numpy as np

# Primes for hashing, same as seed to maintain consistency in distribution quality
PRIMES = (1035493289, 775374203, 1495862053, 2015145641)
NUM_ROWS = 4
NUM_COUNTERS = 128  # Doubled from 64 to utilize the space saved by uint16
HASH_MOD = 2147483647 # Large prime for the first hash step

def _get_indices(item, row):
    """
    Calculate the index for a given item and row.
    Uses the formula: ((item * p + 31 * row) % 2147483647) % 128
    """
    p = PRIMES[row]
    # Ensure item is treated as a large integer for multiplication to avoid overflow issues in Python (though Python ints are arbitrary precision)
    # The modulo 2147483647 is a large prime to ensure good distribution before the final modulo 128
    idx = (item * p + 31 * row) % HASH_MOD
    return idx % NUM_COUNTERS

def update(state, item):
    """
    Update the sketch state with a new item.
    state: bytes (1024 bytes)
    item: int (0..399)
    returns: bytes (1024 bytes)
    """
    # Convert state to numpy array of uint16
    # 4 rows * 128 counters * 2 bytes = 1024 bytes
    sketch = np.frombuffer(state, dtype=np.uint16).reshape(NUM_ROWS, NUM_COUNTERS)
    
    # Increment counters for each row
    for r in range(NUM_ROWS):
        idx = _get_indices(item, r)
        # Increment. Since max frequency is 1024 and uint16 max is 65535, no overflow risk.
        sketch[r, idx] += 1
        
    # Convert back to bytes
    return sketch.tobytes()

def query(state, item):
    """
    Query the estimated frequency of an item.
    state: bytes (1024 bytes)
    item: int (0..399)
    returns: nonnegative frequency estimate (int)
    """
    # Convert state to numpy array of uint16
    sketch = np.frombuffer(state, dtype=np.uint16).reshape(NUM_ROWS, NUM_COUNTERS)
    
    # Get values for each row
    values = []
    for r in range(NUM_ROWS):
        idx = _get_indices(item, r)
        values.append(int(sketch[r, idx]))
        
    # Return the minimum value to reduce overestimation bias
    return min(values)


"""
Optimized streaming frequency sketch.
Strategy: Use a 1-row Count-Min Sketch with 512 columns (uint16 counters).
Rationale: The baseline used 4 rows of 64 uint32 columns (256 counters total).
The previous iteration used 1 row of 256 uint32 columns.
By using uint16 counters, we can double the number of columns to 512 within the 1024-byte limit.
Max possible count is ~650 (Zipf heavy hitter + others), which fits in uint16.
Doubling the buckets reduces collision noise by half, significantly improving MAE for low-frequency items.
Total memory: 512 * 2 bytes = 1024 bytes.
"""

def update(state, item):
    # Convert immutable bytes to mutable bytearray
    b = bytearray(state)
    
    # Hash function: Reuse the strong mixing polynomial from the seed, 
    # but map to 512 buckets instead of 64.
    # p = 1035493289
    # offset = 31
    # modulus for mixing = 2147483647 (Mersenne prime)
    # modulus for indexing = 512
    
    idx = ((item * 1035493289 + 31) % 2147483647) % 512
    offset = idx * 2
    
    # Unpack little-endian uint16
    val = int.from_bytes(b[offset:offset+2], 'little')
    
    # Increment
    val += 1
    
    # Pack back
    b[offset:offset+2] = val.to_bytes(2, 'little')
    
    return bytes(b)

def query(state, item):
    idx = ((item * 1035493289 + 31) % 2147483647) % 512
    offset = idx * 2
    val = int.from_bytes(state[offset:offset+2], 'little')
    return val

import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    # 1. Generate the [18, 9, 6] extended QR code
    # The [17, 9, 5] QR code is a cyclic code with generator polynomial g(x) = x^8 + x^5 + x^4 + x^3 + 1
    # We construct the generator matrix for the [17, 9, 5] code, then extend to [18, 9, 6].
    
    # Generator polynomial powers (index = power of x)
    g_powers = [0, 3, 4, 5, 8]
    
    # Generate 9 basis vectors for the [17, 9, 5] code
    # Row i corresponds to x^i * g(x) mod (x^17 - 1)
    basis_17 = []
    for i in range(9):
        word = 0
        for p in g_powers:
            pos = (i + p) % 17
            word |= (1 << pos)
        basis_17.append(word)
        
    # Extend to [18, 9, 6] by adding a parity bit (bit 17)
    # The parity bit ensures the total weight is even, increasing min distance from 5 to 6.
    basis_18 = []
    for word in basis_17:
        # Use int.bit_count() for faster population count (Python 3.10+)
        parity = word.bit_count() % 2
        if parity:
            word |= (1 << 17)
        basis_18.append(word)
        
    # Generate all 512 codewords by XORing subsets of the basis vectors
    # Optimized: Pre-compute basis as list, use local variable lookup
    b = basis_18
    current_code = [0] * 512
    for msg in range(512):
        word = 0
        # Unroll or simple loop. Simple loop is fast enough for 9 iterations.
        for i in range(9):
            if (msg >> i) & 1:
                word ^= b[i]
        current_code[msg] = word
        
    return current_code

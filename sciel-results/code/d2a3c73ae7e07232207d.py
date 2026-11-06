import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin before 90s kill

    N = 18
    MIN_DISTANCE = 6
    
    # The optimal code for A_2(18, 6) is the extended quadratic residue code [18, 9, 6].
    # It has exactly 512 codewords.
    
    # Generator polynomial for the [17, 9, 5] QR code:
    # g(x) = x^8 + x^5 + x^4 + x^3 + 1
    # Coefficients are at powers: 0, 3, 4, 5, 8
    g_powers = (0, 3, 4, 5, 8)
    
    # Generate 9 basis vectors for the [17, 9, 5] cyclic code
    # Row i of the generator matrix is x^i * g(x) mod (x^17 - 1)
    basis_17 = []
    for i in range(9):
        word = 0
        for p in g_powers:
            pos = (i + p) % 17
            word |= (1 << pos)
        basis_17.append(word)
        
    # Extend to [18, 9, 6] by adding an overall parity check bit (bit 17)
    # This ensures all codewords have even weight, increasing min distance from 5 to 6.
    basis_18 = []
    for word in basis_17:
        parity = bin(word).count('1') & 1
        if parity:
            word |= (1 << 17)
        basis_18.append(word)
        
    # Generate all 512 codewords
    # We can do this by iterating through all 9-bit messages and XORing the corresponding basis vectors.
    # For efficiency and simplicity, we use a Gray-code-like incremental approach or direct XOR.
    # Direct XOR for 512 items is very fast.
    
    codewords = [0] * 512
    # codewords[0] is already 0
    
    # To speed up generation, we can precompute the XOR of basis vectors for each bit position
    # but since it's only 9 bits, we can just iterate.
    
    # A faster way to generate all combinations:
    # Iterate from 1 to 511.
    # Use the property that code[m] = code[m ^ (m & -m)] ^ basis[bit_index]
    # where bit_index is the index of the lowest set bit in m.
    
    for msg in range(1, 512):
        # Get the lowest set bit
        lsb = msg & -msg
        bit_idx = lsb.bit_length() - 1
        prev_msg = msg ^ lsb
        codewords[msg] = codewords[prev_msg] ^ basis_18[bit_idx]
        
    # The code is guaranteed to have distinct codewords and min distance 6.
    # No further verification is needed for correctness, but we can do a quick distinctness check
    # if we want to be absolutely safe, though it's overkill for 512 items.
    
    return codewords

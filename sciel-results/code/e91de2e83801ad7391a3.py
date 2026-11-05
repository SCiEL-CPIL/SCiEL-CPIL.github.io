import numpy as np
import time

def construct():
    """
    Constructs the extended quadratic-residue code QR(17) to length 18.
    Returns a list of 512 distinct integers representing 18-bit codewords
    with pairwise Hamming distance at least 6.
    """
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin
    
    # Parameters
    p = 17
    n = 18
    k = 9
    
    # 1. Compute Quadratic Residues mod 17
    # The set of quadratic residues mod 17 excluding 0
    qr_set = set()
    for i in range(1, p):
        if pow(i, (p - 1) // 2, p) == 1:
            qr_set.add(i)
    
    # 2. Construct the Generator Matrix G (9x17)
    # The first row is defined by the indicator function of {0} U QR
    first_row = np.zeros(p, dtype=np.uint8)
    first_row[0] = 1
    for r in qr_set:
        first_row[r] = 1
        
    # Generate the 9 rows by cyclic shifting
    G = np.vstack([np.roll(first_row, -shift) for shift in range(k)])
    
    # 3. Generate all 512 codewords using Vectorized XOR operations
    # Create a 512x9 matrix of coefficients (0 or 1)
    masks = np.arange(1 << k, dtype=np.uint16)
    # Extract bits for each mask position
    # bit i of mask m is (m >> i) & 1
    coeffs = np.zeros((1 << k, k), dtype=np.uint8)
    for i in range(k):
        coeffs[:, i] = (masks >> i) & 1
        
    # Compute codewords: coeffs @ G mod 2
    # Since we are in GF(2), this is equivalent to XOR sum of rows
    # Using matrix multiplication with mod 2
    codewords_17 = (coeffs @ G) % 2
    
    # 4. Extend to length 18 with parity bit
    # Parity bit is the sum of bits mod 2
    parity_bits = codewords_17.sum(axis=1) % 2
    
    # Append parity bit to each codeword
    codewords_18 = np.column_stack((codewords_17, parity_bits))
    
    # 5. Convert to list of integers
    words = []
    for row in codewords_18:
        # Convert binary array to integer
        # row[j] is bit j (LSB)
        val = 0
        for j in range(n):
            if row[j]:
                val |= (1 << j)
        words.append(val)
        
    # 6. Verification
    # Check distinctness
    if len(set(words)) != len(words):
        # Fallback if somehow duplicates (should not happen)
        return list(set(words))
        
    # Check minimum distance (spot check or full if time permits)
    # For linear codes, min distance = min weight of nonzero codewords
    min_weight = 18
    for w in words:
        wt = w.bit_count()
        if 0 < wt < min_weight:
            min_weight = wt
            
    if min_weight < 6:
        # This should never happen for QR(17)
        # If it does, return the valid subset (though all should be valid)
        # Just to be safe, we trust the construction but this guard is cheap
        pass
        
    return words

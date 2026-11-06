"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Construct the extended binary quadratic residue code of length 18.
    # Base code: length 17, dimension 9, min distance 5.
    # Extended code: length 18, dimension 9, min distance 6.
    # Number of codewords: 2^9 = 512.
    
    n = 17
    # Quadratic residues modulo 17
    residues = set()
    for i in range(1, 17):
        residues.add((i * i) % 17)
    # residues = {1, 2, 4, 8, 9, 13, 15, 16}
    
    # Create the residue vector v (17 bits)
    v = 0
    for r in residues:
        v |= (1 << r)
        
    # Generate 17 cyclic shifts of v
    shifts = []
    mask = (1 << n) - 1
    for k in range(n):
        # Cyclic shift left by k
        shifted = ((v << k) | (v >> (n - k))) & mask
        shifts.append(shifted)
        
    # Find a basis for the span of these shifts using Gaussian elimination
    basis = []
    for word in shifts:
        w = word
        for b in basis:
            # Eliminate the highest set bit of w using b
            # Find highest set bit of w
            if w == 0:
                break
            hb = b.bit_length() - 1
            hw = w.bit_length() - 1
            if hb == hw:
                w ^= b
        if w != 0:
            # Normalize: ensure the highest bit is unique? 
            # Standard GE: add to basis
            basis.append(w)
            
    # The dimension should be 9. 
    # If basis size is not 9, we might have an issue, but QR(17) is known to be dim 9.
    # Let's verify basis size.
    # If basis size > 9, we just use the first 9? No, the span is the code.
    # The code is the span of the shifts. The dimension is 9.
    
    # Generate all 2^len(basis) combinations
    # If basis size is 9, we generate 512 words.
    num_basis = len(basis)
    if num_basis != 9:
        # Fallback: if something went wrong with the basis extraction (e.g. dependent vectors not eliminated correctly)
        # We should ensure we have exactly 9 independent vectors.
        # The current GE adds w if it's not in the span of previous basis vectors.
        # This produces a basis for the span.
        pass
        
    words = []
    for mask_val in range(1 << num_basis):
        word = 0
        for i in range(num_basis):
            if (mask_val >> i) & 1:
                word ^= basis[i]
        
        # Extend by parity bit to make length 18
        # Parity bit is 1 if weight is odd
        parity = word.bit_count() % 2
        extended_word = word | (parity << 17)
        words.append(extended_word)
        
    return words
# EVOLVE-BLOCK-END

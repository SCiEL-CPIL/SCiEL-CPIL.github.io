import time
import random
import numpy as np

def construct():
    N = 18
    MIN_DIST = 6
    
    # Construct the Extended Quadratic Residue Code [18, 9, 6]
    # Based on prime p = 17
    p = 17
    # Quadratic residues modulo 17
    Q = set()
    for i in range(1, p):
        Q.add((i * i) % p)
    
    # Base vector of length 17
    # v[j] = 1 if j in Q else 0
    v = [0] * p
    for q in Q:
        v[q] = 1
        
    # Generate 9 basis vectors of length 18
    basis = []
    for i in range(9):
        word = 0
        weight_sum = 0
        for j in range(p):
            # Cyclic shift: bit j in row i corresponds to v[(j - i) % p]
            bit = v[(j - i) % p]
            if bit:
                word |= (1 << j)
                weight_sum += 1
        # Extend with parity bit at position 17
        if weight_sum % 2 == 1:
            word |= (1 << 17)
        basis.append(word)
        
    # Generate all 2^9 = 512 linear combinations
    codewords = []
    for mask in range(1 << 9):
        w = 0
        for i in range(9):
            if (mask >> i) & 1:
                w ^= basis[i]
        codewords.append(w)
        
    # Ensure distinctness (should be distinct by construction of linear code)
    # And sort for consistency
    codewords = sorted(list(set(codewords)))
    
    return codewords

import numpy as np
import time

def construct():
    N = 18
    MIN_DIST = 6
    
    # Construct the 512-word Extended Quadratic Residue Code
    # Quadratic residues mod 17: {1, 2, 4, 8, 9, 13, 15, 16}
    # The idempotent vector has 1s at Q U {0} = {0, 1, 2, 4, 8, 9, 13, 15, 16}
    
    residues = {0, 1, 2, 4, 8, 9, 13, 15, 16}
    
    # Create base integer (17 bits)
    base_int = 0
    for r in residues:
        base_int |= (1 << r)
        
    # Create generator integers (basis for the 17-bit part)
    G_ints = [0] * 9
    # G[0] is all 1s in 17 bits
    G_ints[0] = (1 << 17) - 1
    
    # G[1] is base
    G_ints[1] = base_int
    
    # G[2..8] are cyclic shifts of base_int within 17 bits
    for i in range(2, 9):
        shift = i - 1
        # Cyclic shift left by 'shift' positions in a 17-bit field
        G_ints[i] = ((base_int << shift) | (base_int >> (17 - shift))) & ((1 << 17) - 1)
        
    codes = []
    for m in range(512):
        val = 0
        for j in range(9):
            if (m >> j) & 1:
                val ^= G_ints[j]
        
        # Extend with parity bit (bit 17)
        if val.bit_count() % 2 != 0:
            val |= (1 << 17)
            
        codes.append(val)
        
    return codes

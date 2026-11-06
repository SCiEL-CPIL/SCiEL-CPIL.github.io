"""Improved program using Exhaustive Sign-Search on Structured Exponents."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def count_nonzero(arr):
    return sum(1 for x in arr if x != 0)

def compute_square_nonzero(poly):
    square = np.convolve(poly, poly)
    return count_nonzero(square)

def construct():
    start_time = time.time()
    best_poly = None
    best_nonzero = float('inf')
    
    # Strategy: Exhaustive search over all +/-1 coefficient assignments
    # for a set of carefully selected exponent structures.
    # This avoids local minima of hill climbing by guaranteeing global optimum
    # within the +/-1 subspace for each structure.
    
    structures = []
    
    # 1. Arithmetic Progressions
    # These create many collisions in the square (e_i + e_j = e_k + e_l)
    for step in range(1, 15):
        for start in range(0, 10):
            exponents = [start + i * step for i in range(12)]
            if max(exponents) <= 40:
                structures.append(exponents)
                
    # 2. Symmetric Structures (Palindromic exponents)
    # e.g., 0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15
    for gap in range(1, 10):
        for offset in range(0, 5):
            base = list(range(6))
            top = [x + 5 + gap + offset for x in range(5, -1, -1)] # 5 down to 0
            exponents = sorted(base + top)
            if max(exponents) <= 40 and len(set(exponents)) == 12:
                structures.append(exponents)

    # 3. Random distinct exponents (Seeded for reproducibility in logic, but we want diversity)
    # We will pick a fixed set of random structures to avoid time variance
    import random
    random.seed(42)
    for _ in range(20):
        exponents = sorted(random.sample(range(0, 41), 12))
        structures.append(exponents)

    # Deduplicate structures
    unique_structures = []
    seen = set()
    for s in structures:
        t = tuple(s)
        if t not in seen:
            seen.add(t)
            unique_structures.append(s)
            
    structures = unique_structures
    
    # Sort by potential "collision density" (smaller range often better for cancellations?)
    # Or just iterate all. 4096 * N_structures operations.
    # N_structures is likely < 100. 4096 * 100 = 409,600 convolutions.
    # Convolution of size 41 is very fast.
    
    # Pre-compute max degree for each structure to size the array
    for exponents in structures:
        if time.time() - start_time > 55:
            break
            
        max_deg = max(exponents)
        poly_len = max_deg + 1
        
        # Exhaustive search over 2^12 sign combinations
        # We can optimize by noting that flipping all signs doesn't change the square.
        # So we can fix the first coefficient to 1 and search 2^11 = 2048 combos.
        
        # Generate all masks for 11 bits (indices 1 to 11)
        # Index 0 is always +1
        for mask in range(1 << 11):
            if time.time() - start_time > 55:
                break
                
            coeffs = [1] * 12
            # Set signs for indices 1..11 based on mask bits
            for i in range(11):
                if (mask >> i) & 1:
                    coeffs[i+1] = -1
            # coeffs[0] is 1
            
            # Construct polynomial
            current_poly = [0] * poly_len
            for e, c in zip(exponents, coeffs):
                current_poly[e] = c
                
            current_nz = compute_square_nonzero(current_poly)
            
            if current_nz < best_nonzero:
                best_nonzero = current_nz
                best_poly = current_poly[:]

    # Fallback if nothing found
    if best_poly is None:
        best_poly = [0]*41
        for i in range(12):
            best_poly[i] = 1
            
    # Trim trailing zeros
    while len(best_poly) > 1 and best_poly[-1] == 0:
        best_poly.pop()
    
    # Ensure length <= 41
    if len(best_poly) > 41:
        best_poly = best_poly[:41]
        while len(best_poly) > 1 and best_poly[-1] == 0:
            best_poly.pop()
    
    # Ensure nonzero final coefficient
    if best_poly[-1] == 0:
        best_poly[-1] = 1
        
    return best_poly
# EVOLVE-BLOCK-END

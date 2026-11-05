"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # 8s safety margin
    
    best_square_count = 13
    best_poly = [1]*12
    
    # Verify baseline
    sq = np.convolve(np.array(best_poly, dtype=int), np.array(best_poly, dtype=int))
    best_square_count = int(np.count_nonzero(sq))
    
    def count_nonzero_square(poly):
        p = np.array(poly, dtype=int)
        sq = np.convolve(p, p)
        return int(np.count_nonzero(sq))
    
    # === STRUCTURAL CHANGE: Exponent-structure-first approach ===
    # Key insight: The number of distinct sums i+j determines the MINIMUM
    # possible nonzero terms in the square. If we minimize distinct sums,
    # we maximize coefficient collisions, enabling cancellation.
    #
    # For 12 exponents, minimum distinct sums = 23 (arithmetic progression).
    # AP with step 1: [0,1,...,11] -> 13 distinct sums (all coefficients positive, no cancellation)
    # AP with step 2: [0,2,...,22] -> 23 distinct sums (all positive, no cancellation)
    #
    # The breakthrough: Use AP exponents BUT with carefully chosen coefficients
    # that create cancellation in the square.
    
    # Phase 1: Enumerate ALL 12-subsets of exponents from [0..40] that form
    # arithmetic progressions (most collision-prone structures)
    ap_candidates = []
    for step in range(1, 11):  # step 1 to 10
        max_start = 40 - 11 * step
        for start in range(0, max_start + 1):
            exponents = [start + i * step for i in range(12)]
            ap_candidates.append(exponents)
    
    # Phase 2: For each AP, try coefficient patterns that create cancellation
    # The square coefficient at position (i+j)*step is sum of c_a * c_b where a+b = i+j
    # For AP [s, s+k, s+2k, ..., s+11k], the square has terms at degrees
    # 2s, 2s+k, 2s+2k, ..., 2s+22k (23 positions)
    # Coefficient at position 2s+m*k = sum_{a+b=m} c_a * c_b for a,b in 0..11
    
    # Try to find coefficients c_0..c_11 (integers in [-64,64], nonzero)
    # such that many of the 23 convolution sums are zero
    
    # Strategy: Use the fact that if c_i = (-1)^i, then for even m,
    # sum_{a+b=m} (-1)^a (-1)^b = sum_{a+b=m} 1 (since (-1)^a * (-1)^b = (-1)^{a+b} = (-1)^m = 1 for even m)
    # For odd m, sum = sum_{a+b=m} (-1) = -(number of pairs)
    # So alternating signs DON'T help for AP - all terms remain nonzero.
    
    # Better strategy: Use coefficients that satisfy c_i * c_j = 0 for many (i,j) pairs
    # This means we need some c_i = 0, but we need exactly 12 nonzero terms.
    
    # Alternative: Use coefficients from a small set like {-2, -1, 1, 2}
    # and search for patterns where convolution sums cancel.
    
    # Phase 3: Systematic search over AP exponents with small coefficient sets
    for exponents in ap_candidates:
        if time.time() > deadline:
            break
        
        # For this AP, search over coefficient assignments
        # Use a smarter approach: try all combinations of {-1, 1} first (2^12 = 4096)
        for trial in range(4096):
            if time.time() > deadline:
                break
            
            # Decode trial as binary pattern for signs
            coeffs = [1 if (trial >> i) & 1 else -1 for i in range(12)]
            
            poly = [0] * 41
            for i, e in enumerate(exponents):
                if e < 41:
                    poly[e] = coeffs[i]
            
            nz = count_nonzero_square(poly)
            if nz < best_square_count:
                best_square_count = nz
                best_poly = poly[:]
                if best_square_count <= 12:
                    break
        
        if best_square_count <= 12:
            break
    
    # Phase 4: Try AP with coefficients from {-2, -1, 1, 2}
    if best_square_count > 12 and time.time() < deadline:
        for exponents in ap_candidates[:50]:  # Limit to first 50 APs
            if time.time() > deadline:
                break
            
            # Random search over coefficient assignments from {-2,-1,1,2}
            for trial in range(2000):
                if time.time() > deadline:
                    break
                
                coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
                
                poly = [0] * 41
                for i, e in enumerate(exponents):
                    if e < 41:
                        poly[e] = coeffs[i]
                
                # Check exactly 12 nonzero terms
                if sum(1 for c in poly if c != 0) != 12:
                    continue
                if any(c < -64 or c > 64 for c in poly):
                    continue
                
                nz = count_nonzero_square(poly)
                if nz < best_square_count:
                    best_square_count = nz
                    best_poly = poly[:]
                    if best_square_count <= 12:
                        break
            
            if best_square_count <= 12:
                break
    
    # Phase 5: Local search refinement on best candidate
    if time.time() < deadline:
        current = best_poly[:]
        current_nz = count_nonzero_square(current)
        
        for iteration in range(5000):
            if time.time() > deadline:
                break
            
            new_poly = current[:]
            nonzero_indices = [i for i, c in enumerate(current) if c != 0]
            
            if not nonzero_indices:
                break
            
            op = random.choice(['flip_sign', 'move_term', 'change_coeff', 'swap_terms'])
            
            if op == 'flip_sign':
                idx = random.choice(nonzero_indices)
                new_poly[idx] = -current[idx]
            elif op == 'move_term':
                src = random.choice(nonzero_indices)
                zero_indices = [i for i, c in enumerate(current) if c == 0]
                if zero_indices:
                    dst = random.choice(zero_indices)
                    new_poly[dst] = current[src]
                    new_poly[src] = 0
                else:
                    continue
            elif op == 'change_coeff':
                idx = random.choice(nonzero_indices)
                new_coeff = random.choice([-3, -2, -1, 1, 2, 3])
                if new_coeff != current[idx]:
                    new_poly[idx] = new_coeff
            elif op == 'swap_terms':
                if len(nonzero_indices) >= 2:
                    i, j = random.sample(nonzero_indices, 2)
                    new_poly[i], new_poly[j] = current[j], current[i]
                else:
                    continue
            
            # Validate
            nz_count = sum(1 for c in new_poly if c != 0)
            if nz_count != 12:
                continue
            if any(c < -64 or c > 64 for c in new_poly):
                continue
            
            new_nz = count_nonzero_square(new_poly)
            if new_nz < current_nz:
                current = new_poly
                current_nz = new_nz
                if current_nz < best_square_count:
                    best_square_count = current_nz
                    best_poly = current[:]
    
    # Phase 6: Try non-AP structures with high collision potential
    # Use exponents that are close together to maximize sum collisions
    if time.time() < deadline:
        # Try clusters: groups of exponents that are close
        cluster_patterns = [
            [0,1,2,3,4,5,6,7,8,9,10,11],  # AP step 1
            [0,1,2,3,4,5,6,7,8,9,10,12],  # Near AP
            [0,1,2,3,4,5,6,7,8,10,11,12],
            [0,1,2,3,4,5,7,8,9,10,11,12],
            [0,1,2,3,4,6,7,8,9,10,11,12],
        ]
        
        for exponents in cluster_patterns:
            if time.time() > deadline:
                break
            
            for trial in range(1000):
                if time.time() > deadline:
                    break
                
                coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
                
                poly = [0] * 41
                for i, e in enumerate(exponents):
                    if e < 41:
                        poly[e] = coeffs[i]
                
                if sum(1 for c in poly if c != 0) != 12:
                    continue
                if any(c < -64 or c > 64 for c in poly):
                    continue
                
                nz = count_nonzero_square(poly)
                if nz < best_square_count:
                    best_square_count = nz
                    best_poly = poly[:]
    
    # Final validation
    final_sq = np.convolve(np.array(best_poly, dtype=int), np.array(best_poly, dtype=int))
    final_nz = int(np.count_nonzero(final_sq))
    
    result = best_poly[:]
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    
    if not result:
        result = [1]
    
    return result
# EVOLVE-BLOCK-END

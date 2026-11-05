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
    
    def count_nonzero_square(poly):
        p = np.array(poly, dtype=int)
        sq = np.convolve(p, p)
        return int(np.count_nonzero(sq))
    
    # === STRUCTURAL CHANGE: Targeted algebraic search ===
    # Key insight: For a polynomial with exponents E = {e_0 < e_1 < ... < e_11},
    # the square has nonzero terms at positions e_i + e_j for all i,j.
    # The number of DISTINCT positions is at most 23 (if all sums are distinct).
    # We want to maximize cancellations: find coefficients c_i such that
    # sum_{i+j=k} c_i * c_j = 0 for as many k as possible.
    #
    # Strategy: Use exponent sets where many sums collide, then solve for
    # coefficients that force zero at those positions.
    
    # Phase 1: Generate candidate exponent sets with high collision structure
    # Use sets where the difference multiset has many repeats
    
    candidate_exponents = []
    
    # Arithmetic progressions (step 1-5)
    for step in range(1, 6):
        max_start = 40 - 11 * step
        for start in range(0, max_start + 1):
            exponents = [start + i * step for i in range(12)]
            candidate_exponents.append(exponents)
    
    # Near-AP: AP with one perturbation
    for step in range(1, 4):
        max_start = 40 - 11 * step
        for start in range(0, max_start + 1):
            base = [start + i * step for i in range(12)]
            for pos in range(12):
                for delta in [-1, 1]:
                    new_exp = base[pos] + delta
                    if 0 <= new_exp <= 40 and new_exp not in base:
                        candidate_exponents.append(sorted(base[:pos] + [new_exp] + base[pos+1:]))
    
    # Random structured sets: two interleaved APs
    for step1 in range(1, 3):
        for step2 in range(1, 3):
            if step1 == step2:
                continue
            for start1 in range(0, 30):
                for start2 in range(0, 30):
                    exps1 = [start1 + i * step1 for i in range(6)]
                    exps2 = [start2 + i * step2 for i in range(6)]
                    if all(0 <= e <= 40 for e in exps1 + exps2) and len(set(exps1 + exps2)) == 12:
                        candidate_exponents.append(sorted(exps1 + exps2))
    
    # Deduplicate
    candidate_exponents = list(set(tuple(e) for e in candidate_exponents))
    candidate_exponents = [list(e) for e in candidate_exponents]
    
    # Phase 2: For each exponent set, use targeted coefficient search
    # Focus on small coefficient sets: {-1, 1}, {-2, -1, 1, 2}, {-3, -2, -1, 1, 2, 3}
    
    coeff_sets = [
        [-1, 1],
        [-2, -1, 1, 2],
        [-3, -2, -1, 1, 2, 3],
    ]
    
    # Sort candidates by expected collision count (heuristic: prefer APs)
    def collision_score(exponents):
        """Higher score = more sum collisions = better candidate"""
        sums = [exponents[i] + exponents[j] for i in range(12) for j in range(12)]
        unique_sums = len(set(sums))
        return -unique_sums  # Minimize unique sums
    
    candidate_exponents.sort(key=collision_score)
    
    # Phase 3: Systematic search with early termination
    for exponents in candidate_exponents[:200]:  # Limit to top 200 candidates
        if time.time() > deadline:
            break
        
        # Try each coefficient set
        for coeff_set in coeff_sets:
            if time.time() > deadline:
                break
            
            # For small sets, do systematic search
            if len(coeff_set) == 2:
                # Binary search: 2^12 = 4096 possibilities
                for trial in range(4096):
                    if time.time() > deadline:
                        break
                    coeffs = [coeff_set[(trial >> i) & 1] for i in range(12)]
                    poly = [0] * 41
                    for i, e in enumerate(exponents):
                        if e < 41:
                            poly[e] = coeffs[i]
                    
                    if sum(1 for c in poly if c != 0) != 12:
                        continue
                    
                    nz = count_nonzero_square(poly)
                    if nz < best_square_count:
                        best_square_count = nz
                        best_poly = poly[:]
                        if best_square_count <= 13:
                            break
            else:
                # For larger sets, use random sampling
                for trial in range(5000):
                    if time.time() > deadline:
                        break
                    coeffs = [random.choice(coeff_set) for _ in range(12)]
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
                        if best_square_count <= 13:
                            break
        
        if best_square_count <= 13:
            break
    
    # Phase 4: Local search refinement
    if time.time() < deadline:
        current = best_poly[:]
        current_nz = count_nonzero_square(current)
        
        for iteration in range(10000):
            if time.time() > deadline:
                break
            
            new_poly = current[:]
            nonzero_indices = [i for i, c in enumerate(current) if c != 0]
            
            if not nonzero_indices:
                break
            
            op = random.choice(['flip_sign', 'move_term', 'change_coeff', 'swap_terms', 'scale'])
            
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
                new_coeff = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
                if new_coeff != current[idx]:
                    new_poly[idx] = new_coeff
            elif op == 'swap_terms':
                if len(nonzero_indices) >= 2:
                    i, j = random.sample(nonzero_indices, 2)
                    new_poly[i], new_poly[j] = current[j], current[i]
                else:
                    continue
            elif op == 'scale':
                idx = random.choice(nonzero_indices)
                new_poly[idx] = current[idx] * random.choice([-2, 2])
            
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

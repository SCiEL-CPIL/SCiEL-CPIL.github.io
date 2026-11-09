"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np
from collections import Counter, defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s kill
    
    def count_nonzero_square_dense(coeffs):
        """Calculate nonzero square terms for a dense list."""
        n = len(coeffs)
        square_coeffs = [0] * (2 * n - 1)
        for i in range(n):
            ai = coeffs[i]
            if ai == 0: continue
            for j in range(i, n):
                aj = coeffs[j]
                if aj == 0: continue
                k = i + j
                prod = ai * aj
                if i == j:
                    square_coeffs[k] += prod
                else:
                    square_coeffs[k] += 2 * prod
        return sum(1 for c in square_coeffs if c != 0)

    def count_nonzero_sparse(exponents, coeffs):
        """Calculate nonzero square terms for sparse representation."""
        # exponents: list of ints
        # coeffs: list of ints corresponding to exponents
        n = len(exponents)
        # Map sum -> coefficient
        sums = defaultdict(int)
        for i in range(n):
            ei = exponents[i]
            ci = coeffs[i]
            if ci == 0: continue
            for j in range(i, n):
                ej = exponents[j]
                cj = coeffs[j]
                if cj == 0: continue
                s = ei + ej
                prod = ci * cj
                if i == j:
                    sums[s] += prod
                else:
                    sums[s] += 2 * prod
        return sum(1 for v in sums.values() if v != 0)

    def is_valid_sparse(exponents, coeffs):
        if len(exponents) != 12:
            return False
        if len(set(exponents)) != 12:
            return False
        if any(e < 0 or e > 40 for e in exponents):
            return False
        if any(abs(c) > 64 for c in coeffs):
            return False
        if any(c == 0 for c in coeffs):
            return False
        # Nonzero final coefficient condition:
        # The polynomial length is max(exponents) + 1.
        # The coefficient of x^max(exponents) must be non-zero.
        # Since all coeffs are non-zero, this is satisfied.
        return True

    def to_dense_list(exponents, coeffs):
        """Convert sparse representation to dense list for return."""
        max_deg = max(exponents)
        dense = [0] * (max_deg + 1)
        for e, c in zip(exponents, coeffs):
            dense[e] = c
        # Ensure length <= 41
        if len(dense) > 41:
            # This shouldn't happen if exponents <= 40
            pass
        return dense

    best_score = 100
    best_result_dense = None
    
    # --- Phase 1: Find High-Collision Supports ---
    # A "good" support has many collisions in pair sums.
    # We generate candidate supports using a greedy/evolutionary approach.
    
    candidate_supports = []
    
    # 1. Add the standard dense support 0..11
    candidate_supports.append(list(range(12)))
    
    # 2. Generate supports with high collision density
    # Strategy: Start with random sets, mutate by moving elements to create collisions.
    # Collision count metric: Number of pairs (i,j) with i<j such that e_i+e_j = e_k+e_l for some other pair.
    # Actually, simpler metric: Minimize the number of distinct sums.
    
    def score_support(exponents):
        sums = set()
        n = len(exponents)
        for i in range(n):
            for j in range(i, n):
                sums.add(exponents[i] + exponents[j])
        return len(sums)
        
    # Greedy construction of supports
    # We want exponents in [0, 40].
    # Let's try to pack them tightly to increase collisions.
    
    random.seed(42)
    for _ in range(5000):
        # Random start
        exponents = random.sample(range(0, 41), 12)
        exponents.sort()
        
        # Local search on support to minimize distinct sums
        improved = True
        while improved:
            improved = False
            for i in range(12):
                old_e = exponents[i]
                # Try moving this exponent to a nearby value
                for delta in range(-3, 4):
                    new_e = old_e + delta
                    if new_e < 0 or new_e > 40:
                        continue
                    # Check uniqueness
                    if new_e in exponents and exponents[i] != new_e:
                        continue
                    # Actually, we need to check if new_e collides with other exponents
                    # We maintain a sorted list, so insertion is easy, but let's just check set
                    temp_exponents = exponents[:]
                    temp_exponents[i] = new_e
                    if len(set(temp_exponents)) != 12:
                        continue
                    temp_exponents.sort()
                    
                    old_score = score_support(exponents)
                    new_score = score_support(temp_exponents)
                    
                    if new_score < old_score:
                        exponents = temp_exponents
                        improved = True
                        break
                    elif new_score == old_score:
                        # Accept equal score to diversify? No, stick to improvements for speed.
                        pass
        
        if exponents not in candidate_supports:
            candidate_supports.append(exponents)
        
        if time.time() > deadline - 10:
            break
            
    # --- Phase 2: Optimize Coefficients for Best Supports ---
    
    # Sort supports by their collision score (lower is better)
    candidate_supports.sort(key=score_support)
    
    # We will evaluate the top K supports
    K = min(50, len(candidate_supports))
    
    for support in candidate_supports[:K]:
        if time.time() > deadline - 5:
            break
            
        # For this support, find coefficients in [-64, 64] \ {0} that minimize nonzero square terms.
        # This is a hard problem. We use a heuristic:
        # 1. Try all sign assignments for coefficients = 1.
        # 2. Try small integers {-1, 1, 2, -2}.
        
        best_support_score = 100
        best_coeffs = None
        
        # Heuristic 1: Sign search for c_i in {1, -1}
        # 2^12 = 4096 combinations. Very fast.
        for bits in range(4096):
            coeffs = [1] * 12
            for i in range(12):
                if (bits >> i) & 1:
                    coeffs[i] = -1
            
            score = count_nonzero_sparse(support, coeffs)
            if score < best_support_score:
                best_support_score = score
                best_coeffs = coeffs[:]
                
        # Heuristic 2: Local search expanding coefficient values
        if best_coeffs is None:
            best_coeffs = [1] * 12
            best_support_score = count_nonzero_sparse(support, best_coeffs)
            
        current_coeffs = best_coeffs[:]
        current_score = best_support_score
        
        # Local search on coefficients
        # We allow coefficients to be any integer in [-64, 64] \ {0}
        # We focus on small integers first as they are more likely to cancel.
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(12):
                old_val = current_coeffs[i]
                # Try a set of candidate values
                # Focus on small values: -3..3 excluding 0
                candidates = set()
                for v in range(-3, 4):
                    if v != 0:
                        candidates.add(v)
                # Also try neighbors of old_val
                for d in range(-2, 3):
                    if d == 0: continue
                    nv = old_val + d
                    if nv != 0 and abs(nv) <= 64:
                        candidates.add(nv)
                        
                for nv in candidates:
                    if nv == old_val:
                        continue
                    current_coeffs[i] = nv
                    new_score = count_nonzero_sparse(support, current_coeffs)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                    else:
                        current_coeffs[i] = old_val
                        
        if current_score < best_score:
            best_score = current_score
            best_result_dense = to_dense_list(support, current_coeffs)
            
            if best_score <= 5: # Theoretical lower bound is 1 (perfect square), but 12 terms is hard. 5 is very good.
                break

    # Fallback if nothing found or error
    if best_result_dense is None:
        # Return a valid dense polynomial
        best_result_dense = [1] * 12
        best_score = count_nonzero_square_dense(best_result_dense)

    # Final validation
    # Ensure length <= 41
    if len(best_result_dense) > 41:
         # Truncate? No, this would change the polynomial. 
         # But our supports are <= 40, so length <= 41 is guaranteed.
         pass
         
    # Ensure nonzero final coefficient
    if best_result_dense[-1] == 0:
         # Should not happen given our construction
         best_result_dense[-1] = 1

    return best_result_dense

# EVOLVE-BLOCK-END

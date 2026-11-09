import time
import random
import numpy as np
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safe margin for 60s limit (evaluator kills at 90s, but let's be safe)
    
    best_score = float('inf')
    best_poly = None
    
    # Helper to count nonzeros in square using NumPy
    def count_nz_square(poly):
        p_np = np.array(poly, dtype=np.int64)
        sq = np.convolve(p_np, p_np)
        return np.count_nonzero(sq)

    # Structural approach: P = A + B where A, B are sparse
    # P^2 = A^2 + 2AB + B^2
    # If we can make 2AB cancel terms in A^2 + B^2, we get sparse P^2
    
    # Key insight: If A and B have disjoint support and we choose coefficients
    # such that cross terms cancel with self terms, we can get very sparse squares.
    
    # Strategy 1: P = A + B where A has terms at even positions, B at odd positions
    # Then A^2 has only even degrees, B^2 has only even degrees, 2AB has only odd degrees
    # No cancellation possible between these, so total = |A^2| + |B^2| + |2AB|
    
    # Strategy 2: P = A - B where A and B overlap in support
    # P^2 = A^2 - 2AB + B^2
    # If A^2 and B^2 have overlapping terms, and 2AB cancels some, we can reduce count
    
    # Better strategy: Use the identity (x^k - 1)(x^m - 1) = x^(k+m) - x^k - x^m + 1
    # This gives 4 terms. For 12 terms, we can use products of such sparse polynomials.
    
    # Let's try: P(x) = (1 - x^a)(1 - x^b)(1 - x^c)(1 - x^d) * Q(x)
    # Each (1 - x^a) contributes 2 terms, so 4 such factors give up to 16 terms
    # But with overlaps, we can get exactly 12 terms.
    
    # Actually, let's try a more direct approach:
    # P(x) = sum_{i=0}^{11} c_i x^{e_i}
    # We want P(x)^2 to have few nonzero terms.
    # 
    # Key: If the exponents form a Sidon set (all pairwise sums distinct), then
    # P^2 has exactly 12*13/2 = 78 terms (no cancellation).
    # To minimize, we want many collisions in the sumset, AND coefficients that cancel.
    #
    # Best structure: Exponents in arithmetic progression!
    # If e_i = i*d, then sums are 0, d, 2d, ..., 22d (23 distinct sums)
    # Coeff of x^(kd) in P^2 = sum_{i=0}^{k} c_i * c_{k-i} (for k <= 11)
    #                         = sum_{i=k-11}^{11} c_i * c_{k-i} (for 11 < k <= 22)
    #
    # For k = 0: c_0^2 (always nonzero)
    # For k = 22: c_11^2 (always nonzero)
    # For k = 11: sum_{i=0}^{11} c_i^2 (sum of squares, always positive, nonzero)
    #
    # So minimum is at least 3 nonzero terms! Can we achieve exactly 3?
    # We need all other coefficients to be zero.
    #
    # For k = 1: 2*c_0*c_1 = 0 => c_1 = 0 (contradiction, we need 12 nonzero terms)
    # So AP doesn't work directly.
    #
    # Alternative: Use a structure where many cross terms cancel.
    # 
    # Let's try: P(x) = 1 - x + x^2 - x^3 + ... + x^10 - x^11 (alternating)
    # P(x) = (1 - x^12)/(1 - x)
    # P(x)^2 = (1 - x^12)^2/(1-x)^2 = (1 - 2x^12 + x^24)/(1 - 2x + x^2)
    # This is not sparse in standard basis.
    
    # Let me try a completely different approach:
    # Use the fact that (a+b)^2 = a^2 + b^2 + 2ab
    # If we set P = A + B where A^2 and B^2 are sparse, and 2AB cancels most terms...
    
    # Actually, let's just do a smarter local search with better initialization.
    # The current approach does random search. Let's add structured candidates
    # based on known sparse polynomial identities.
    
    candidates = []
    
    # Candidate 1: Alternating AP (known to have some structure)
    p = [0]*41
    for i in range(12):
        p[i] = 1 if i % 2 == 0 else -1
    candidates.append(p)
    
    # Candidate 2: 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1 (period 4)
    p = [0]*41
    pattern = [1, -1, -1, 1]
    for i in range(12):
        p[i] = pattern[i % 4]
    candidates.append(p)
    
    # Candidate 3: Try to construct P where P^2 has few terms
    # Use the idea: if P(x) = Q(x)^2 * R(x) where Q is sparse...
    # Too complex. Let's just add more structured random candidates.
    
    # Add candidates with coefficients from {-2, -1, 1, 2} on AP
    for d in [1, 2]:
        for start in range(0, 41 - 11*d + 1):
            for coeffs in [
                [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
                [1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1],
                [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
                [2, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
                [1, -2, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
            ]:
                p = [0]*41
                for i in range(12):
                    idx = start + i*d
                    if idx < 41:
                        p[idx] = coeffs[i]
                candidates.append(p)
    
    # Evaluate all candidates
    for p in candidates:
        nz = sum(1 for x in p if x != 0)
        if nz != 12:
            continue
        if p[-1] == 0:
            # Move a term to position 40
            nz_idx = [i for i in range(41) if p[i] != 0]
            if nz_idx:
                k = nz_idx[-1]
                p[40] = p[k]
                p[k] = 0
        if any(abs(x) > 64 for x in p):
            continue
        score = count_nz_square(p)
        if score < best_score:
            best_score = score
            best_poly = p[:]
    
    # If no valid candidate, use default
    if best_poly is None:
        best_poly = [0]*41
        for i in range(12):
            best_poly[i] = 1
        best_score = count_nz_square(best_poly)
    
    # Local search from best_poly
    current_poly = best_poly[:]
    current_score = best_score
    
    T = 3.0
    T_min = 0.1
    cooling_rate = 0.995
    
    no_improve_count = 0
    max_no_improve = 300
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            if T > T_min:
                T = max(T_min, T * cooling_rate)
            else:
                # Random restart from a structured candidate
                if candidates:
                    current_poly = random.choice(candidates).copy()
                else:
                    current_poly = best_poly.copy()
                current_score = count_nz_square(current_poly)
                no_improve_count = 0
                T = 3.0
            continue
        
        # Generate neighbor
        move_type = random.random()
        
        if move_type < 0.5:
            # Change coefficient
            nz_indices = [i for i in range(41) if current_poly[i] != 0]
            idx = random.choice(nz_indices)
            old_val = current_poly[idx]
            
            # Try specific values that might create cancellations
            new_val = random.choice([-2, -1, 1, 2])
            if new_val == old_val:
                new_val = old_val + random.choice([-1, 1])
                if new_val == 0:
                    new_val = 1
            if abs(new_val) > 64:
                new_val = 64 if new_val > 0 else -64
            
            current_poly[idx] = new_val
            new_score = count_nz_square(current_poly)
            
            if new_score < current_score:
                current_score = new_score
                no_improve_count = 0
                if current_score < best_score:
                    best_score = current_score
                    best_poly = current_poly[:]
            else:
                delta_score = new_score - current_score
                if delta_score > 0 and random.random() < np.exp(-delta_score / T):
                    current_score = new_score
                else:
                    current_poly[idx] = old_val
                    no_improve_count += 1
        else:
            # Move index
            nz_indices = [i for i in range(41) if current_poly[i] != 0]
            z_indices = [i for i in range(41) if current_poly[i] == 0]
            if not z_indices:
                no_improve_count += 1
                continue
                
            src_idx = random.choice(nz_indices)
            dst_idx = random.choice(z_indices)
            
            if src_idx == 40:
                no_improve_count += 1
                continue
            
            val = current_poly[src_idx]
            current_poly[src_idx] = 0
            current_poly[dst_idx] = val
            
            new_score = count_nz_square(current_poly)
            
            if new_score < current_score:
                current_score = new_score
                no_improve_count = 0
                if current_score < best_score:
                    best_score = current_score
                    best_poly = current_poly[:]
            else:
                delta_score = new_score - current_score
                if delta_score > 0 and random.random() < np.exp(-delta_score / T):
                    current_score = new_score
                else:
                    current_poly[dst_idx] = 0
                    current_poly[src_idx] = val
                    no_improve_count += 1
    
    return best_poly

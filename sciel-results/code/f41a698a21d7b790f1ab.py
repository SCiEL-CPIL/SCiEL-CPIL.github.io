"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools
import time
import random

# EVOLVE-BLOCK-START
import numpy as np

def _square_coeffs(c):
    """Return coefficient list of c*c (convolution), trimmed to remove trailing zeros."""
    c_arr = np.array(c, dtype=np.int64)
    result = np.convolve(c_arr, c_arr)
    nz = np.nonzero(result)[0]
    if len(nz) == 0:
        return [0]
    last = nz[-1]
    return result[:last+1].tolist()

def _nnz(c):
    return sum(1 for x in c if x != 0)

def _is_valid(poly):
    """Check if polynomial satisfies all constraints."""
    if len(poly) > 41:
        return False
    if len(poly) == 0 or poly[-1] == 0:
        return False
    if sum(1 for x in poly if x != 0) != 12:
        return False
    if any(abs(x) > 64 for x in poly):
        return False
    return True

def _constructive_search():
    """
    Search for palindromic polynomials of degree 11 by forcing low-degree
    coefficients of the square to be zero.
    """
    best_sq_nnz = float('inf')
    best_poly = None
    
    # Range for c0, c1
    range_c = list(range(-32, 33))
    range_c.remove(0)
    
    # We will fix c0, c1 and solve for c2, c3, c4 to make d2, d3, d4 zero.
    # Then iterate c5.
    # d2 = c1^2 + 2 c0 c2 = 0  => c2 = -c1^2 / (2 c0)
    # d3 = 2 c1 c2 + 2 c0 c3 = 0 => c3 = -c1 c2 / c0
    # d4 = c2^2 + 2 c1 c3 + 2 c0 c4 = 0 => c4 = -(c2^2 + 2 c1 c3) / (2 c0)
    
    for c0 in range_c:
        for c1 in range_c:
            # Check divisibility for c2
            num2 = -c1 * c1
            den2 = 2 * c0
            if num2 % den2 != 0:
                continue
            c2 = num2 // den2
            if c2 == 0 or abs(c2) > 64:
                continue
            
            # Check divisibility for c3
            num3 = -c1 * c2
            den3 = c0
            if num3 % den3 != 0:
                continue
            c3 = num3 // den3
            if c3 == 0 or abs(c3) > 64:
                continue
            
            # Check divisibility for c4
            num4 = -(c2 * c2 + 2 * c1 * c3)
            den4 = 2 * c0
            if num4 % den4 != 0:
                continue
            c4 = num4 // den4
            if c4 == 0 or abs(c4) > 64:
                continue
            
            # Now iterate c5
            for c5 in range(-64, 65):
                if c5 == 0:
                    continue
                
                # Construct palindromic poly: c0..c5, c5..c0
                poly = [c0, c1, c2, c3, c4, c5, c5, c4, c3, c2, c1, c0]
                
                # Verify all 12 are non-zero (they are by construction)
                # Verify range (checked above)
                
                sq = _square_coeffs(poly)
                nnz_sq = _nnz(sq)
                
                if nnz_sq < best_sq_nnz:
                    best_sq_nnz = nnz_sq
                    best_poly = poly[:]
                    
    return best_poly, best_sq_nnz

def _local_search(poly, max_steps=200, time_limit=2.0):
    """Hill-climb: try flipping signs or changing individual coefficients."""
    start = time.time()
    current = poly[:]
    current_sq_nnz = _nnz(_square_coeffs(current))
    
    for step in range(max_steps):
        if time.time() - start > time_limit:
            break
        improved = False
        # Try changing one coefficient at a time
        for i in range(len(current)):
            old_val = current[i]
            # Try small deltas
            for delta in [-2, -1, 1, 2]:
                new_val = old_val + delta
                if new_val == 0 or abs(new_val) > 64:
                    continue
                candidate = current[:]
                candidate[i] = new_val
                if not _is_valid(candidate):
                    continue
                sq = _square_coeffs(candidate)
                nnz_sq = _nnz(sq)
                if nnz_sq < current_sq_nnz:
                    current = candidate
                    current_sq_nnz = nnz_sq
                    improved = True
                    break
            if improved:
                break
        
        # If no improvement with deltas, try sign flips
        if not improved:
            improved = False
            for i in range(len(current)):
                if current[i] != 0:
                    candidate = current[:]
                    candidate[i] = -candidate[i]
                    if not _is_valid(candidate):
                        continue
                    sq = _square_coeffs(candidate)
                    nnz_sq = _nnz(sq)
                    if nnz_sq < current_sq_nnz:
                        current = candidate
                        current_sq_nnz = nnz_sq
                        improved = True
                        break
            if not improved:
                break
    
    return current, current_sq_nnz

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # safety margin before 90s
    
    best_sq_nnz = float('inf')
    best_poly = [1]*12
    
    # 1. Constructive Search
    p, nnz = _constructive_search()
    if p:
        best_poly = p
        best_sq_nnz = nnz
    
    # 2. Local Search on the best candidate
    if best_sq_nnz < float('inf'):
        improved_poly, improved_nnz = _local_search(best_poly)
        if improved_nnz < best_sq_nnz:
            best_poly = improved_poly
            best_sq_nnz = improved_nnz
    
    # 3. Random Search with Local Optimization
    random.seed(42)
    non_zero_vals = list(range(-64, 65))
    non_zero_vals.remove(0)
    
    attempts = 0
    max_attempts = 100000
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        if attempts % 10000 == 0 and time.time() > deadline:
            break
        
        # Random palindromic degree 11
        free = [random.choice(non_zero_vals) for _ in range(6)]
        poly = free + [free[5], free[4], free[3], free[2], free[1], free[0]]
        
        sq = _square_coeffs(poly)
        nnz_sq = _nnz(sq)
        
        # If this random candidate is good, try local search
        if nnz_sq <= best_sq_nnz + 2: 
            # Only polish if it's close
            if time.time() < deadline - 5.0:
                improved_poly, improved_nnz = _local_search(poly, max_steps=50, time_limit=1.0)
                if improved_nnz < best_sq_nnz:
                    best_poly = improved_poly
                    best_sq_nnz = improved_nnz
        elif nnz_sq < best_sq_nnz:
            best_poly = poly
            best_sq_nnz = nnz_sq

    # Final Verification
    if len(best_poly) == 0 or best_poly[-1] == 0:
        best_poly = [1]*12
    if sum(1 for x in best_poly if x != 0) != 12:
        best_poly = [1]*12
    if len(best_poly) > 41:
        best_poly = best_poly[:41]
        
    return best_poly
# EVOLVE-BLOCK-END

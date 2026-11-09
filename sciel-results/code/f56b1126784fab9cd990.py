import time
import itertools
import random
import numpy as np

# EVOLVE-BLOCK-START

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

def _local_search(poly, max_steps=2000, time_limit=10.0):
    """Hill-climb with single and pairwise moves."""
    start = time.time()
    current = poly[:]
    current_sq_nnz = _nnz(_square_coeffs(current))
    n = len(current)
    
    for step in range(max_steps):
        if time.time() - start > time_limit:
            break
        improved = False
        
        # 1. Try changing one coefficient
        for i in range(n):
            old_val = current[i]
            # Try small deltas
            for delta in [-1, 1, -2, 2, -3, 3, -4, 4, -5, 5]:
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
            
            # Sign flip
            if old_val != 0:
                candidate = current[:]
                candidate[i] = -old_val
                if _is_valid(candidate):
                    sq = _square_coeffs(candidate)
                    nnz_sq = _nnz(sq)
                    if nnz_sq < current_sq_nnz:
                        current = candidate
                        current_sq_nnz = nnz_sq
                        improved = True
                        break

        if not improved:
            # 2. Try changing two coefficients (pairwise)
            if random.random() < 0.5:
                i, j = random.sample(range(n), 2)
                old_i, old_j = current[i], current[j]
                for delta_i in [-1, 1, -2, 2, -3, 3]:
                    for delta_j in [-1, 1, -2, 2, -3, 3]:
                        new_i = old_i + delta_i
                        new_j = old_j + delta_j
                        if new_i == 0 or new_j == 0 or abs(new_i) > 64 or abs(new_j) > 64:
                            continue
                        candidate = current[:]
                        candidate[i] = new_i
                        candidate[j] = new_j
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

        if not improved:
             # Stagnation: break
             break
    
    return current, current_sq_nnz

def _make_product_poly(A, B):
    """Multiply two polynomials A and B, return coefficient list."""
    if len(A) == 0 or len(B) == 0:
        return [0]
    A_arr = np.array(A, dtype=np.int64)
    B_arr = np.array(B, dtype=np.int64)
    result = np.convolve(A_arr, B_arr)
    nz = np.nonzero(result)[0]
    if len(nz) == 0:
        return [0]
    last = nz[-1]
    return result[:last+1].tolist()

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    best_sq_nnz = float('inf')
    best_poly = [1]*12
    random.seed(42)
    
    # Phase 1: Exhaustive search on structured candidates (Palindromic/Anti-palindromic)
    search_values = [-3, -2, -1, 1, 2, 3]
    
    # 1. Palindromic degree 11
    for coeffs in itertools.product(search_values, repeat=6):
        if time.time() > deadline - 15: break
        a0, a1, a2, a3, a4, a5 = coeffs
        poly = [a0, a1, a2, a3, a4, a5, a5, a4, a3, a2, a1, a0]
        if not _is_valid(poly):
            continue
        sq = _square_coeffs(poly)
        nnz_sq = _nnz(sq)
        if nnz_sq < best_sq_nnz:
            best_sq_nnz = nnz_sq
            best_poly = poly[:]
    
    # 2. Anti-palindromic degree 12 (13 coeffs, middle 0)
    for coeffs in itertools.product(search_values, repeat=6):
        if time.time() > deadline - 15: break
        a0, a1, a2, a3, a4, a5 = coeffs
        poly = [a0, a1, a2, a3, a4, a5, 0, -a5, -a4, -a3, -a2, -a1, -a0]
        if not _is_valid(poly):
            continue
        sq = _square_coeffs(poly)
        nnz_sq = _nnz(sq)
        if nnz_sq < best_sq_nnz:
            best_sq_nnz = nnz_sq
            best_poly = poly[:]
            
    # 3. Larger coefficients for Palindromic degree 11
    if best_sq_nnz > 16 and time.time() < deadline - 25:
        search_values_large = [-4, -3, -2, -1, 1, 2, 3, 4]
        for coeffs in itertools.product(search_values_large, repeat=6):
            if time.time() > deadline - 25: break
            a0, a1, a2, a3, a4, a5 = coeffs
            poly = [a0, a1, a2, a3, a4, a5, a5, a4, a3, a2, a1, a0]
            if not _is_valid(poly):
                continue
            sq = _square_coeffs(poly)
            nnz_sq = _nnz(sq)
            if nnz_sq < best_sq_nnz:
                best_sq_nnz = nnz_sq
                best_poly = poly[:]

    # Phase 2: Product-of-factors approach
    # P(x) = A(x) * B(x) where A and B are sparse
    # Try products of small sparse polynomials
    
    # Precompute some good sparse bases
    # (1+x)^2 = 1+2x+x^2 has 3 terms
    # (1+x+x^2)^2 has 5 terms
    # (1-x+x^2)^2 has 5 terms
    
    # Generate candidate factors
    small_factors = []
    # Factors with 2-4 terms
    for nterms in [2, 3, 4]:
        for degree in range(1, 12):
            for coeffs in itertools.product([-1, 1], repeat=nterms):
                for positions in itertools.combinations(range(degree+1), nterms):
                    f = [0]*(degree+1)
                    for p, c in zip(positions, coeffs):
                        f[p] = c
                    if f[-1] == 0:
                        continue
                    small_factors.append(f)
    
    # Limit to avoid explosion
    random.shuffle(small_factors)
    small_factors = small_factors[:500]
    
    # Try products of pairs
    factor_sample = small_factors[:200]
    for i, f1 in enumerate(factor_sample):
        if time.time() > deadline - 20:
            break
        for f2 in factor_sample[i:]:
            if time.time() > deadline - 20:
                break
            prod = _make_product_poly(f1, f2)
            nnz_prod = _nnz(prod)
            if nnz_prod != 12:
                continue
            if not _is_valid(prod):
                continue
            sq = _square_coeffs(prod)
            nnz_sq = _nnz(sq)
            if nnz_sq < best_sq_nnz:
                best_sq_nnz = nnz_sq
                best_poly = prod[:]

    # Phase 3: Random search with local optimization
    non_zero_vals = list(range(-10, 11))
    non_zero_vals.remove(0)
    
    attempts = 0
    max_attempts = 50000
    
    while time.time() < deadline - 10 and attempts < max_attempts:
        attempts += 1
        
        r = random.random()
        
        if r < 0.3:
            # Palindromic degree 11
            free = [random.choice(non_zero_vals) for _ in range(6)]
            poly = free + [free[5], free[4], free[3], free[2], free[1], free[0]]
        elif r < 0.5:
            # Anti-palindromic degree 12
            free = [random.choice(non_zero_vals) for _ in range(6)]
            poly = free + [0] + [-free[5], -free[4], -free[3], -free[2], -free[1], -free[0]]
        elif r < 0.7:
            # Random 12-term polynomial
            degree = random.randint(11, 25)
            poly = [0]*(degree+1)
            positions = random.sample(range(degree+1), 12)
            for p in positions:
                poly[p] = random.choice(non_zero_vals)
            if poly[-1] == 0:
                poly[-1] = random.choice(non_zero_vals)
        else:
            # Product of sparse factors
            n1 = random.choice([2, 3, 4])
            n2 = 12 // n1
            if n2 < 2: n2 = 2
            f1_len = random.randint(n1, 8)
            f1 = [0]*f1_len
            pos1 = random.sample(range(f1_len), min(n1, f1_len))
            for p in pos1:
                f1[p] = random.choice([-1, 1])
            f2_len = random.randint(n2, 8)
            f2 = [0]*f2_len
            pos2 = random.sample(range(f2_len), min(n2, f2_len))
            for p in pos2:
                f2[p] = random.choice([-1, 1])
            prod = np.convolve(f1, f2)
            poly = prod.tolist()
            while len(poly) > 1 and poly[-1] == 0:
                poly.pop()
        
        if not _is_valid(poly):
            continue
        
        sq = _square_coeffs(poly)
        nnz_sq = _nnz(sq)
        
        if nnz_sq < best_sq_nnz:
            best_sq_nnz = nnz_sq
            best_poly = poly[:]
            
            # Local search
            if time.time() < deadline - 15:
                improved_poly, improved_nnz = _local_search(poly, max_steps=500, time_limit=5.0)
                if improved_nnz < best_sq_nnz:
                    best_sq_nnz = improved_nnz
                    best_poly = improved_poly[:]
    
    # Phase 4: Final polish on best found
    if time.time() < deadline - 3:
        improved_poly, improved_nnz = _local_search(best_poly, max_steps=2000, time_limit=8.0)
        if improved_nnz < best_sq_nnz:
            best_sq_nnz = improved_nnz
            best_poly = improved_poly[:]
    
    # Ensure output is valid
    while len(best_poly) > 1 and best_poly[-1] == 0:
        best_poly.pop()
    
    if len(best_poly) == 0 or best_poly[-1] == 0:
        best_poly = [1]*12
    
    if sum(1 for x in best_poly if x != 0) != 12:
        best_poly = [1]*12
    
    if len(best_poly) > 41:
        best_poly = best_poly[:41]
        
    return best_poly

# EVOLVE-BLOCK-END

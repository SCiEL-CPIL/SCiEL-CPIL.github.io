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

def _product_structure_search():
    """
    Search using product structures like (1 ± x^a ± x^b ± ...) which have
    built-in cancellation in their squares.
    
    Key insight: (1 + x^a)^2 = 1 + 2x^a + x^{2a} has 3 terms.
    (1 + x^a + x^b)^2 has 6 terms if no overlaps.
    (1 + x^a - x^b)^2 = 1 + x^{2a} + x^{2b} + 2x^a - 2x^b - 2x^{a+b} (6 terms)
    
    For 12 terms, we can use products of 4 binomials:
    (1+x^a)(1+x^b)(1+x^c)(1+x^d) has 16 terms, square has up to 33 terms.
    But with signs: (1+x^a)(1+x^b)(1-x^c)(1-x^d) etc.
    
    Better: use sums of monomials with specific coefficient patterns.
    """
    best_sq_nnz = float('inf')
    best_poly = None
    
    # Strategy 1: Products of binomials (1 ± x^k)
    # (1 + x^a)(1 + x^b) = 1 + x^a + x^b + x^{a+b} (4 terms)
    # Square has 6 terms (if all exponents distinct)
    # (1 + x^a)(1 + x^b)(1 + x^c) = 8 terms
    # Square has 12 terms (if all pair sums distinct and no self-overlaps)
    
    # Let's try: P = sum of 12 monomials c_i * x^{e_i}
    # If P = (1+x^a)(1+x^b)(1+x^c)(1+x^d) with some signs, we get 16 terms.
    # We need exactly 12 nonzero terms, so we need some cancellation.
    
    # Alternative: P = (1 + x^a + x^b + x^c + x^d + x^e) * (1 + x^f)
    # This gives 12 terms if f is large enough to avoid overlap.
    # Square: (1+x^a+...+x^e)^2 * (1+x^f)^2
    # = (1 + 2x^a + ... + 2x^e + x^{2a} + ... ) * (1 + 2x^f + x^{2f})
    
    # Let's try specific structured forms:
    # Form 1: P = (1 + x^a)(1 + x^b)(1 + x^c)(1 + x^d) - k*x^m
    # This has 16-1 = 15 terms, not 12.
    
    # Form 2: P = sum_{i=0}^{11} c_i * x^{e_i} where e_i are chosen to create cancellation
    # in the square.
    
    # Let's try: P = 1 + x + x^2 + x^3 + x^4 + x^5 + x^6 + x^7 + x^8 + x^9 + x^10 + x^11
    # Square has 23 terms (triangular).
    
    # Better approach: use the fact that (x^a - x^b)^2 = x^{2a} - 2x^{a+b} + x^{2b}
    # has 3 terms.
    
    # Let's try P = A + B where A and B have disjoint supports.
    # P^2 = A^2 + 2AB + B^2
    # If A and B have disjoint supports, 2AB has terms at positions i+j where i in supp(A), j in supp(B).
    # A^2 has terms at 2i where i in supp(A).
    # B^2 has terms at 2j where j in supp(B).
    
    # For minimal nnz, we want 2AB to cancel with parts of A^2 and B^2.
    # This is hard to achieve directly.
    
    # Let's try a different approach: use small coefficient values and specific exponent patterns.
    
    # Try: P = 1 + x^a + x^b + x^c + x^d + x^e + x^f + x^g + x^h + x^i + x^j + x^k
    # with specific choices of exponents to create cancellation in P^2.
    
    # Known good structure: P = 1 + x + x^2 - x^3 - x^4 - x^5 + x^6 + x^7 + x^8 - x^9 - x^{10} - x^{11}
    # This is related to Ramanujan's 1-2-4-8-16-32-64 pattern.
    
    # Let's try systematic search over small exponent sets.
    
    max_exp = 40
    
    # Try polynomials of the form: sum of 6 pairs (c, -c) at symmetric positions
    # P = c*(x^a - x^b) + c*(x^c - x^d) + c*(x^e - x^f) + c*(x^g - x^h) + c*(x^i - x^j) + c*(x^k - x^l)
    # This has 12 terms.
    
    # Square: sum of 36 terms (c^2 * (x^a-x^b)^2 etc.) plus cross terms.
    # (x^a - x^b)^2 = x^{2a} - 2x^{a+b} + x^{2b}
    # Cross: 2c^2(x^a - x^b)(x^c - x^d) = 2c^2(x^{a+c} - x^{a+d} - x^{b+c} + x^{b+d})
    
    # This is complex. Let's just do a focused random search with structure.
    
    random.seed(123)
    
    for _ in range(50000):
        # Generate random 12-term polynomial with small coefficients
        exponents = sorted(random.sample(range(0, max_exp + 1), 12))
        coeffs = [random.choice([1, -1, 2, -2, 3, -3]) for _ in range(12)]
        
        poly = [0] * (max_exp * 2 + 1)
        for e, c in zip(exponents, coeffs):
            poly[e] = c
        
        # Trim
        while len(poly) > 0 and poly[-1] == 0:
            poly.pop()
        
        if len(poly) == 0 or poly[-1] == 0:
            continue
        if len(poly) > 41:
            continue
        if sum(1 for x in poly if x != 0) != 12:
            continue
        if any(abs(x) > 64 for x in poly):
            continue
        
        sq = _square_coeffs(poly)
        nnz_sq = _nnz(sq)
        
        if nnz_sq < best_sq_nnz:
            best_sq_nnz = nnz_sq
            best_poly = poly[:]
    
    return best_poly, best_sq_nnz

def _algebraic_structure_search():
    """
    Search using algebraic structures that create cancellation.
    Key idea: P = Q * R where Q and R are sparse, and Q^2, R^2, QR have
    overlapping terms that cancel.
    """
    best_sq_nnz = float('inf')
    best_poly = None
    
    # Try P = (1 + x^a + x^b + x^c + x^d + x^e) * (1 + x^f)
    # This has 12 terms if f > e.
    # P^2 = (1 + x^a + ... + x^e)^2 * (1 + 2x^f + x^{2f})
    # = (1 + 2x^a + 2x^b + 2x^c + 2x^d + 2x^e + x^{2a} + ... ) * (1 + 2x^f + x^{2f})
    
    # Let's try specific cases.
    
    # Case 1: a=1, b=2, c=3, d=4, e=5, f=6
    # Q = 1 + x + x^2 + x^3 + x^4 + x^5
    # Q^2 = 1 + 2x + 3x^2 + 4x^3 + 5x^4 + 6x^5 + 5x^6 + 4x^7 + 3x^8 + 2x^9 + x^10
    # P = Q * (1 + x^6) = Q + x^6 * Q
    # P^2 = Q^2 * (1 + x^6)^2 = Q^2 * (1 + 2x^6 + x^{12})
    
    # Let's compute this.
    
    def _poly_mul(p1, p2):
        """Multiply two polynomials given as coefficient lists."""
        res = [0] * (len(p1) + len(p2) - 1)
        for i, a in enumerate(p1):
            if a == 0:
                continue
            for j, b in enumerate(p2):
                if b == 0:
                    continue
                res[i + j] += a * b
        while len(res) > 0 and res[-1] == 0:
            res.pop()
        return res if res else [0]
    
    # Try various structured forms
    for a in range(1, 10):
        for b in range(a+1, 15):
            for c in range(b+1, 20):
                for d in range(c+1, 25):
                    for e in range(d+1, 30):
                        for f in range(e+1, 41):
                            # P = (1 + x^a + x^b + x^c + x^d + x^e) * (1 + x^f)
                            Q = [0] * (e + 1)
                            Q[0] = 1
                            Q[a] = 1
                            Q[b] = 1
                            Q[c] = 1
                            Q[d] = 1
                            Q[e] = 1
                            
                            R = [0] * (f + 1)
                            R[0] = 1
                            R[f] = 1
                            
                            P = _poly_mul(Q, R)
                            
                            if len(P) > 41:
                                continue
                            if sum(1 for x in P if x != 0) != 12:
                                continue
                            if P[-1] == 0:
                                continue
                            if any(abs(x) > 64 for x in P):
                                continue
                            
                            sq = _square_coeffs(P)
                            nnz_sq = _nnz(sq)
                            
                            if nnz_sq < best_sq_nnz:
                                best_sq_nnz = nnz_sq
                                best_poly = P[:]
    
    return best_poly, best_sq_nnz

def _simulated_annealing(poly, time_limit=30.0):
    """
    Simulated annealing to escape local minima.
    """
    import math
    
    start = time.time()
    current = poly[:]
    current_sq = _square_coeffs(current)
    current_nnz = _nnz(current_sq)
    
    best = current[:]
    best_nnz = current_nnz
    
    T = 10.0
    T_min = 0.1
    cooling = 0.99
    
    non_zero_vals = list(range(-64, 65))
    non_zero_vals.remove(0)
    
    while time.time() - start < time_limit and T > T_min:
        T *= cooling
        
        # Generate neighbor
        candidate = current[:]
        move_type = random.random()
        
        if move_type < 0.3:
            # Change one coefficient
            i = random.randint(0, len(candidate) - 1)
            candidate[i] = random.choice(non_zero_vals)
        elif move_type < 0.5:
            # Swap two coefficients
            i, j = random.sample(range(len(candidate)), 2)
            candidate[i], candidate[j] = candidate[j], candidate[i]
        elif move_type < 0.7:
            # Add or remove a term
            if random.random() < 0.5:
                # Add term
                if len(candidate) < 41:
                    pos = random.randint(0, 40)
                    while pos < len(candidate) and candidate[pos] != 0:
                        pos += 1
                    if pos < 41:
                        while len(candidate) <= pos:
                            candidate.append(0)
                        candidate[pos] = random.choice(non_zero_vals)
            else:
                # Remove term
                nz_indices = [i for i, x in enumerate(candidate) if x != 0]
                if len(nz_indices) > 12:
                    i = random.choice(nz_indices)
                    candidate[i] = 0
        else:
            # Sign flip
            i = random.randint(0, len(candidate) - 1)
            if candidate[i] != 0:
                candidate[i] = -candidate[i]
        
        # Validate
        if len(candidate) == 0 or candidate[-1] == 0:
            continue
        if sum(1 for x in candidate if x != 0) != 12:
            continue
        if len(candidate) > 41:
            continue
        if any(abs(x) > 64 for x in candidate):
            continue
        
        cand_sq = _square_coeffs(candidate)
        cand_nnz = _nnz(cand_sq)
        
        delta = cand_nnz - current_nnz
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current = candidate
            current_nnz = cand_nnz
            
            if cand_nnz < best_nnz:
                best = candidate[:]
                best_nnz = cand_nnz
    
    return best, best_nnz

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s
    
    best_sq_nnz = float('inf')
    best_poly = [1]*12
    
    # 1. Product structure search
    if time.time() < deadline - 60:
        p, nnz = _product_structure_search()
        if p:
            best_poly = p
            best_sq_nnz = nnz
    
    # 2. Algebraic structure search
    if time.time() < deadline - 30:
        p, nnz = _algebraic_structure_search()
        if p and nnz < best_sq_nnz:
            best_poly = p
            best_sq_nnz = nnz
    
    # 3. Simulated annealing from best candidate
    if best_sq_nnz < float('inf') and time.time() < deadline - 10:
        improved_poly, improved_nnz = _simulated_annealing(best_poly, time_limit=25.0)
        if improved_nnz < best_sq_nnz:
            best_poly = improved_poly
            best_sq_nnz = improved_nnz
    
    # Final Verification
    if len(best_poly) == 0 or best_poly[-1] == 0:
        best_poly = [1]*12
    if sum(1 for x in best_poly if x != 0) != 12:
        best_poly = [1]*12
    if len(best_poly) > 41:
        best_poly = best_poly[:41]
        
    return best_poly
# EVOLVE-BLOCK-END

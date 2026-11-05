import time
import itertools
import random
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = 85.0  # 85s limit to be safe within 90s
    
    best_poly = None
    best_score = float('inf')
    
    def count_square_terms(coeffs_dict):
        """Count nonzero terms in square given sparse dict {exp: coeff}."""
        if not coeffs_dict:
            return 0
        # Compute square via convolution of sparse terms
        square = {}
        items = list(coeffs_dict.items())
        for i, (ei, ci) in enumerate(items):
            for j, (ej, cj) in enumerate(items):
                if i <= j:
                    key = ei + ej
                    val = 2 * ci * cj if i < j else ci * ci
                    square[key] = square.get(key, 0) + val
        return sum(1 for v in square.values() if v != 0)
    
    def evaluate_from_dict(coeffs_dict):
        """Validate and score a sparse polynomial representation."""
        if not coeffs_dict:
            return None
        max_exp = max(coeffs_dict.keys())
        if max_exp > 40:
            return None
        # Convert to list for final check
        n = max_exp + 1
        coeffs_list = [0] * n
        for e, c in coeffs_dict.items():
            coeffs_list[e] = c
        # Trim trailing zeros
        while len(coeffs_list) > 1 and coeffs_list[-1] == 0:
            coeffs_list.pop()
        
        if len(coeffs_list) > 41:
            return None
        if coeffs_list[-1] == 0:
            return None
        nonzero_count = sum(1 for c in coeffs_list if c != 0)
        if nonzero_count != 12:
            return None
        for c in coeffs_list:
            if not (-64 <= c <= 64):
                return None
        
        score = count_square_terms(coeffs_dict)
        return score
    
    def dict_to_list(d):
        """Convert sparse dict to dense list (ascending degree)."""
        if not d:
            return [0]
        max_exp = max(d.keys())
        lst = [0] * (max_exp + 1)
        for e, c in d.items():
            lst[e] = c
        return lst
    
    # ====================================================================
    # STRATEGY: Structural decomposition
    # 
    # Key insight: If P(x) = Q(x) - x^k * R(x) where Q and R have specific
    # relationships, the cross term -2x^k*Q(x)*R(x) can cancel many terms
    # from Q^2 and R^2.
    #
    # Sub-strategy 1: P(x) = A(x) - x^k * A(-x)  (anti-palindromic-ish)
    #   Then P^2 = A^2 - 2x^k A(x)A(-x) + x^{2k} A(-x)^2
    #   If A has even/odd symmetry properties, huge cancellation.
    #
    # Sub-strategy 2: P(x) = A(x) - x^k * B(x) where B is a shifted/reflected A
    #
    # Sub-strategy 3: Use the fact that for P with terms at exponents forming
    # an arithmetic progression, the square has a known structure. We can
    # parameterize coefficients as follows:
    #   P(x) = sum_{i=0}^{11} c_i x^{a*i}  (geometric spacing)
    #   or P(x) = sum_{i=0}^{11} c_i x^{i}  (consecutive)
    #   and use the constraint that we want S_k = sum_{i+j=k} c_i*c_j = 0
    #   for most k. This is a system of quadratic equations.
    #
    # Sub-strategy 4: "Two-cluster" approach
    #   P(x) = A(x) + x^M * B(x) where A has m terms, B has (12-m) terms,
    #   and M is large enough that A^2 and B^2 don't overlap, but the
    #   cross term 2x^M A(x)B(x) has structured cancellation.
    #   If M is chosen so that cross terms cancel with parts of A^2 or B^2.
    #
    # Sub-strategy 5: "Perfect difference" 
    #   P(x) = C(x) - D(x) where C^2 and D^2 have overlapping support
    #   and 2*C*D cancels most of the combined terms.
    # ====================================================================
    
    # --- Sub-strategy 1: Consecutive exponents with structured coefficients ---
    # For exponents 0..11, the square has terms at degrees 0..22.
    # S_k = sum_{i+j=k} c_i c_j for k=0..22.
    # We want S_k = 0 for as many k as possible.
    # 
    # Key: If we set c_i = (-1)^i * d_i, then for the "cross" terms in S_k,
    # signs alternate and may cancel.
    #
    # Even better: try c_i = a * (-1)^i for all i (geometric with ratio -1).
    # Then P(x) = a * sum_{i=0}^{11} (-x)^i = a * (1 - (-x)^12) / (1 + x) = a * (1 + x^12)/(1+x)
    # P(x) = a*(1+x^12)/(1+x) -> this is a rational function, not a polynomial unless we're careful.
    # Actually sum_{i=0}^{11} (-x)^i = (1-(-x)^12)/(1-(-x)) = (1-x^12)/(1+x)
    # So P(x) = a*(1-x^12)/(1+x) which IS a polynomial: a*(1+x+x^2+...+x^11) * ... no.
    # sum_{i=0}^{11} (-x)^i = 1 - x + x^2 - x^3 + ... + x^10 - x^11
    # This has 12 terms. P^2 = (1 - x + x^2 - ... - x^11)^2
    # Let's just compute it.
    
    # Try the alternating sign pattern with various exponents
    def try_alternating(exps, a=1):
        d = {}
        for i, e in enumerate(exps):
            d[e] = a * ((-1) ** i)
        return d
    
    # Try consecutive exponents with alternating signs
    for base in range(0, 5):
        exps = [base + i for i in range(12)]
        if max(exps) > 40:
            continue
        d = try_alternating(exps)
        score = evaluate_from_dict(d)
        if score is not None and score < best_score:
            best_score = score
            best_poly = dict_to_list(d)
    
    # --- Sub-strategy 2: Two-cluster with large gap ---
    # P(x) = A(x) + x^M * B(x)
    # A has m terms at low exponents, B has (12-m) terms at high exponents (shifted by M)
    # A^2 has support in [0, 2*maxA], B^2 in [2M, 2M+2*maxB], cross in [M, M+maxA+maxB]
    # If M > 2*maxA + 2*maxB, no overlap at all -> total terms = terms(A^2)+terms(B^2)+terms(cross)
    # We want overlap to cause cancellation.
    # 
    # Better: choose M so that cross term overlaps with A^2 or B^2.
    # If A = 1 (single term at 0), B has 11 terms.
    # P = 1 + x^M * B(x)
    # P^2 = 1 + 2x^M B(x) + x^{2M} B(x)^2
    # Terms: 1 (from constant), 11 from 2x^M B, and terms from B^2.
    # If B has 11 terms, B^2 can have up to 21 terms (if consecutive).
    # Total: 1 + 11 + up_to_21 = up to 33. Not great.
    # 
    # What if B is chosen so B^2 has few terms?
    # B(x) = x^0 + x^1 + x^2 + ... no, that gives many terms.
    # B(x) with terms at powers of 2: B^2 has all distinct sums -> 11*12/2 + 11 = 72 terms. Bad.
    # B(x) with consecutive terms: B^2 has 21 terms. Total = 1+11+21 = 33.
    # 
    # What about A with 6 terms, B with 6 terms?
    # A consecutive 0..5, B consecutive 0..5, M chosen for overlap.
    # A^2: degrees 0..10 (11 terms)
    # B^2: degrees 2M..2M+10 (11 terms)  
    # Cross: 2x^M * A*B, A*B has degrees 0..10 (11 terms)
    # Total without cancellation: 11+11+11 = 33
    # With overlap: if M=10, cross is at 10..20, overlaps with A^2 (0..10) at degree 10,
    # and with B^2 (20..30) at degree 20.
    # This gives some cancellation.
    
    # Let's try: A = sum_{i=0}^{5} a_i x^i, B = sum_{j=0}^{5} b_j x^j, P = A + x^M B
    # We'll search over small coefficient values.
    
    for m_a in [4, 5, 6]:
        m_b = 12 - m_a
        if m_b < 1:
            continue
        
        # A exponents: 0 to m_a-1 (consecutive)
        # B exponents: 0 to m_b-1 (consecutive), shifted by M
        # Try various M values
        max_a = m_a - 1
        max_b = m_b - 1
        
        for M in range(max_a + max_b, max_a + max_b + 20):
            # Check total degree
            if M + max_b > 40:
                break
            
            # Try small coefficient patterns
            # A coefficients: try all +/-1 patterns (2^m_a possibilities, up to 2^6=64)
            # B coefficients: try all +/-1 patterns (2^m_b possibilities)
            # That's up to 64*64 = 4096 combos per M, times ~20 M values = 81920
            # Each eval is fast. Should be doable.
            
            a_patterns = list(itertools.product([-1, 1], repeat=m_a))
            b_patterns = list(itertools.product([-1, 1], repeat=m_b))
            
            for ap in a_patterns:
                if time.time() > deadline:
                    break
                for bp in b_patterns:
                    d = {}
                    for i in range(m_a):
                        d[i] = ap[i]
                    for j in range(m_b):
                        e = M + j
                        d[e] = d.get(e, 0) + bp[j]
                        if d[e] == 0:
                            del d[e]
                    
                    # Check we have exactly 12 nonzero terms
                    if len(d) != 12:
                        continue
                    
                    score = evaluate_from_dict(d)
                    if score is not None and score < best_score:
                        best_score = score
                        best_poly = dict_to_list(d)
    
    # --- Sub-strategy 3: "Difference of squares" structure ---
    # P(x) = C(x) - D(x) where C and D are sparse.
    # P^2 = C^2 - 2CD + D^2
    # If C and D share the same exponents but different coefficients,
    # CD has the same support as C^2 and D^2, allowing cancellation.
    #
    # Specifically: C(x) = sum c_i x^{e_i}, D(x) = sum d_i x^{e_i} (same exponents)
    # P(x) = sum (c_i - d_i) x^{e_i}
    # Let p_i = c_i - d_i. We need 12 nonzero p_i.
    # P^2 = sum_{i,j} p_i p_j x^{e_i+e_j}
    # This is just the standard square. The "difference" framing doesn't help
    # unless C and D have different exponent sets.
    #
    # Better: C has exponents S1, D has exponents S2, S1 != S2.
    # P has exponents S1 ∪ S2 (with possible overlaps where c_i = d_i giving 0).
    # We need |{i: c_i - d_i != 0}| = 12.
    #
    # If S1 and S2 are both arithmetic progressions with the same difference,
    # C^2, D^2, and CD all have structured supports.
    
    # Try: C = sum_{i=0}^{5} x^{2i} (even powers 0,2,4,6,8,10)
    #      D = sum_{j=0}^{5} x^{2j+1} (odd powers 1,3,5,7,9,11)
    # P = C - D = x^0 - x^1 + x^2 - x^3 + ... + x^10 - x^11 (alternating, 12 terms)
    # This is the same as sub-strategy 1 with alternating signs. Already tried.
    
    # Try: C = x^0 + x^3 + x^6 + x^9 + x^12 + x^15 (6 terms, step 3)
    #      D = x^1 + x^4 + x^7 + x^10 + x^13 + x^16 (6 terms, step 3)
    # P = C - D: 12 terms at exponents 0..16 (specifically 0,1,3,4,6,7,9,10,12,13,15,16)
    # C^2: exponents 0,3,6,9,12,15,18,21,24,27,30 (11 terms)
    # D^2: exponents 2,5,8,11,14,17,20,23,26,29,32 (11 terms)
    # CD: exponents 1,4,7,10,13,16,19,22,25,28,31 (11 terms)
    # P^2 = C^2 - 2CD + D^2
    # All three have 11 terms each at distinct exponent sets!
    # Total: 33 terms. No cancellation possible since supports are disjoint.
    # Not helpful.
    
    # What if C and D have OVERLAPPING exponent sets?
    # C = x^0 + x^2 + x^4 + x^6 + x^8 + x^10 (even 0..10)
    # D = x^1 + x^3 + x^5 + x^7 + x^9 + x^11 (odd 1..11)
    # Already covered.
    
    # --- Sub-strategy 4: Use numpy FFT for fast square computation in search ---
    # This enables much faster evaluation, allowing more search iterations.
    
    import numpy as np
    
    def fast_count_square(coeffs_list):
        """Use numpy FFT to compute square and count nonzero terms.
        Note: FFT may introduce floating point errors, so we use it only as a
        pre-filter, then verify with exact integer arithmetic."""
        n = len(coeffs_list)
        # Pad to next power of 2 for FFT
        size = 1
        while size < 2 * n - 1:
            size *= 2
        f = np.zeros(size, dtype=np.complex128)
        f[:n] = coeffs_list
        f = np.fft.fft(f)
        f = f * f
        sq = np.fft.ifft(f).real
        sq = sq[:2*n-1]
        # Count terms with |value| > 0.5 (threshold for floating point)
        return int(np.sum(np.abs(sq) > 0.5))
    
    # --- Sub-strategy 5: Randomized search with FFT pre-filter ---
    # Generate random 12-term polynomials, use FFT for fast pre-filtering,
    # then exact verification for candidates.
    
    random.seed(42)
    
    for iteration in range(500000):
        if time.time() > deadline:
            break
        
        # Random 12 distinct exponents from 0..40
        exps = sorted(random.sample(range(0, 41), 12))
        coeffs_list = [0] * 41
        for e in exps:
            coeffs_list[e] = random.choice([-3, -2, -1, 1, 2, 3])
        
        # Quick FFT pre-filter
        # Trim to actual length
        last_nonzero = 40
        while last_nonzero > 0 and coeffs_list[last_nonzero] == 0:
            last_nonzero -= 1
        trimmed = coeffs_list[:last_nonzero + 1]
        
        ff_count = fast_count_square(trimmed)
        
        # Only do exact check if FFT count is promising
        if ff_count < best_score:
            d = {e: coeffs_list[e] for e in exps}
            score = evaluate_from_dict(d)
            if score is not None and score < best_score:
                best_score = score
                best_poly = coeffs_list[:]
    
    # --- Sub-strategy 6: Targeted search on "nearly perfect" structures ---
    # For consecutive exponents 0..11, we have 23 sum positions.
    # S_0 = c_0^2 (always nonzero)
    # S_22 = c_11^2 (always nonzero)
    # S_k for k=1..21: we want these to be 0.
    # That's 21 equations in 12 unknowns. Generally no solution,
    # but we can minimize the number of nonzero S_k.
    #
    # Use simulated annealing on the 12 coefficients.
    
    def exact_square_count_from_dict(d):
        """Exact count of nonzero terms in square."""
        square = {}
        items = list(d.items())
        for i in range(len(items)):
            ei, ci = items[i]
            for j in range(i, len(items)):
                ej, cj = items[j]
                key = ei + ej
                if i == j:
                    val = ci * ci
                else:
                    val = 2 * ci * cj
                square[key] = square.get(key, 0) + val
        return sum(1 for v in square.values() if v != 0)
    
    # Simulated annealing on consecutive exponents 0..11
    exps_0_11 = list(range(12))
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Initialize
        curr = {}
        for e in exps_0_11:
            curr[e] = random.choice([-2, -1, 1, 2])
        
        curr_score = exact_square_count_from_dict(curr)
        
        T = 10.0
        for step in range(2000):
            if time.time() > deadline:
                break
            
            # Pick a random coefficient to change
            idx = random.randint(0, 11)
            e = exps_0_11[idx]
            old_val = curr[e]
            new_val = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
            if new_val == old_val:
                continue
            
            curr[e] = new_val
            new_score = exact_square_count_from_dict(curr)
            
            # Accept or reject
            if new_score < curr_score:
                curr_score = new_score
                if curr_score < best_score:
                    best_score = curr_score
                    best_poly = dict_to_list(curr)
            else:
                # Metropolis acceptance
                delta = new_score - curr_score
                if random.random() < np.exp(-delta / max(T, 0.01)):
                    curr_score = new_score
                else:
                    curr[e] = old_val
            
            # Cool down
            T *= 0.995
    
    # --- Sub-strategy 7: Try specific known good patterns ---
    # Pattern: c_i = 1 for even i, -1 for odd i (alternating)
    # Already tried.
    
    # Pattern: c_i = 1 for i in {0,1,2,3,4,5}, c_i = -1 for i in {6,7,8,9,10,11}
    d = {i: (1 if i < 6 else -1) for i in range(12)}
    score = evaluate_from_dict(d)
    if score is not None and score < best_score:
        best_score = score
        best_poly = dict_to_list(d)
    
    # Pattern: c_i = 1 for i in {0,1,2,3}, -1 for i in {4,5,6,7}, 1 for i in {8,9,10,11}
    d = {i: (1 if i % 4 < 2 else -1) for i in range(12)}
    score = evaluate_from_dict(d)
    if score is not None and score < best_score:
        best_score = score
        best_poly = dict_to_list(d)
    
    # Pattern: c_i = 1 for i in {0,1,2}, -1 for i in {3,4,5}, 1 for i in {6,7,8}, -1 for i in {9,10,11}
    d = {i: (1 if i % 6 < 3 else -1) for i in range(12)}
    score = evaluate_from_dict(d)
    if score is not None and score < best_score:
        best_score = score
        best_poly = dict_to_list(d)
    
    # Pattern: c_i = 1 for i in {0,1,2,3,4,5,6,7,8,9,10,11} (all ones)
    d = {i: 1 for i in range(12)}
    score = evaluate_from_dict(d)
    if score is not None and score < best_score:
        best_score = score
        best_poly = dict_to_list(d)
    
    # Pattern: c_i = 2 for i in {0,1,2,3,4,5}, -1 for i in {6,7,8,9,10,11}
    d = {i: (2 if i < 6 else -1) for i in range(12)}
    score = evaluate_from_dict(d)
    if score is not None and score < best_score:
        best_score = score
        best_poly = dict_to_list(d)
    
    # --- Sub-strategy 8: Exhaustive search on small coefficient sets ---
    # For consecutive exponents 0..11, try all combinations of coefficients
    # from {-1, 0, 1} with exactly 12 nonzeros... that's just {-1, 1}^12 = 4096.
    # Already covered by the two-cluster search.
    
    # Try coefficients from {-2, -1, 1, 2} for exponents 0..11
    # That's 4^12 = 16M. Too many. But with FFT pre-filter, maybe.
    # Actually, let's just do it with the exact method since it's fast for 12 terms.
    
    for restart in range(100):
        if time.time() > deadline:
            break
        
        # Random walk in coefficient space
        curr = {}
        for e in range(12):
            curr[e] = random.choice([-2, -1, 1, 2])
        
        curr_score = exact_square_count_from_dict(curr)
        
        for step in range(500):
            if time.time() > deadline:
                break
            
            e = random.randint(0, 11)
            old_val = curr[e]
            new_val = random.choice([-3, -2, -1, 1, 2, 3])
            if new_val == old_val:
                continue
            
            curr[e] = new_val
            new_score = exact_square_count_from_dict(curr)
            
            if new_score <= curr_score:
                curr_score = new_score
                if curr_score < best_score:
                    best_score = curr_score
                    best_poly = dict_to_list(curr)
            else:
                curr[e] = old_val
    
    # --- Final fallback ---
    if best_poly is None:
        # Return the simplest valid polynomial: 12 terms with coefficient 1
        best_poly = [1] * 12
    
    # Final validation
    final_score = evaluate_from_dict({i: best_poly[i] for i in range(len(best_poly)) if best_poly[i] != 0})
    if final_score is None:
        # Fallback to known valid
        best_poly = [1] * 12
    
    return best_poly

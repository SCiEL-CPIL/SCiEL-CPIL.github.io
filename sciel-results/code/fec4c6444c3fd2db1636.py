import time
import itertools
import random
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s limit
    
    best_poly = None
    best_score = float('inf')
    
    def count_square_terms(coeffs_dict):
        """Count nonzero terms in square given sparse dict {exp: coeff}."""
        if not coeffs_dict:
            return 0
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
        n = max_exp + 1
        coeffs_list = [0] * n
        for e, c in coeffs_dict.items():
            coeffs_list[e] = c
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
    
    def poly_mul(dict_a, dict_b):
        """Multiply two sparse polynomials (dicts)."""
        result = {}
        for ea, ca in dict_a.items():
            for eb, cb in dict_b.items():
                key = ea + eb
                result[key] = result.get(key, 0) + ca * cb
        # Remove zeros
        result = {k: v for k, v in result.items() if v != 0}
        return result
    
    def poly_add(dict_a, dict_b):
        """Add two sparse polynomials (dicts)."""
        result = dict(dict_a)
        for k, v in dict_b.items():
            result[k] = result.get(k, 0) + v
        result = {k: v for k, v in result.items() if v != 0}
        return result
    
    def poly_sub(dict_a, dict_b):
        """Subtract two sparse polynomials (dicts)."""
        result = dict(dict_a)
        for k, v in dict_b.items():
            result[k] = result.get(k, 0) - v
        result = {k: v for k, v in result.items() if v != 0}
        return result
    
    def poly_scale(dict_a, scalar):
        """Scale a sparse polynomial by a scalar."""
        result = {k: v * scalar for k, v in dict_a.items() if v * scalar != 0}
        return result
    
    # ====================================================================
    # CORE STRATEGY: Cyclotomic / Factorization approach
    # 
    # Key insight: If P(x) = K(x) * G(x) where:
    #   - K(x) is a small sparse "kernel" with k terms
    #   - G(x) = 1 + x^s + x^{2s} + ... + x^{(m-1)s} (geometric series, m terms)
    #   - k * m = 12
    # 
    # Then P has at most k*m = 12 terms (exactly if no collisions).
    # 
    # P^2 = K(x)^2 * G(x)^2
    # 
    # G(x)^2 = sum_{i,j=0}^{m-1} x^{(i+j)s} = sum_{t=0}^{2(m-1)} (t+1)x^{ts} for t<m,
    #         then (2m-1-t)x^{ts} for t>=m.
    # This has 2m-1 distinct exponents (all multiples of s).
    # 
    # K(x)^2 has at most k(k+1)/2 distinct exponents.
    # 
    # The product K(x)^2 * G(x)^2 will have structured support.
    # 
    # BUT: we can do better. If K(x) is chosen such that K(x)K(x^{-1}) has
    # very few terms (i.e., K is "self-similar"), then K(x)^2 has cancellation.
    # 
    # Even better: use the identity
    #   (1 - x^n)^2 / (1-x)^2 = (1 + x + ... + x^{n-1})^2
    # and look for K such that K(x) = (1-x^a)(1-x^b) or similar.
    #
    # CONCRETE APPROACH:
    # Factor 12 = k * m where k, m in {1,2,3,4,6,12}
    # For each factorization, enumerate sparse kernels K with k terms,
    # choose step s so that max_degree <= 40, and compute P = K * G.
    # 
    # The key is to find K such that K(x)^2 has minimal support.
    # Kernels with palindromic structure or cyclotomic factors work well.
    # ====================================================================
    
    # Generate all sparse kernels with k terms from a coefficient pool
    # and exponent patterns, then test each.
    
    # Coefficient options for kernel terms
    coeff_pool = [-2, -1, 1, 2]
    
    # Exponent patterns for kernels: use small exponents
    # For k terms, exponents should be in a small range to keep total degree low
    
    def generate_kernels(k, max_exp_kernel=10):
        """Generate candidate sparse kernels with k terms."""
        kernels = []
        # Try various exponent patterns
        # Pattern 1: consecutive
        for base in range(0, 3):
            exps = [base + i for i in range(k)]
            if max(exps) > max_exp_kernel:
                continue
            for coeffs in itertools.product(coeff_pool, repeat=k):
                d = {}
                for e, c in zip(exps, coeffs):
                    if c != 0:
                        d[e] = d.get(e, 0) + c
                if len(d) == k:
                    kernels.append(d)
        
        # Pattern 2: powers of 2 (sparse)
        if k <= 4:
            exps = [0, 1, 2, 4, 8][:k]
            for coeffs in itertools.product(coeff_pool, repeat=k):
                d = {}
                for e, c in zip(exps, coeffs):
                    if c != 0:
                        d[e] = d.get(e, 0) + c
                if len(d) == k:
                    kernels.append(d)
        
        # Pattern 3: symmetric (palindromic)
        if k % 2 == 0:
            half = k // 2
            for base in range(0, 3):
                exps = [base + i for i in range(half)]
                # Mirror
                full_exps = exps + [exps[-1] - i for i in range(half, 0, -1)]
                # Actually let's do: exps = [base, base+1, ..., base+half-1, base+half, ...]
                # Simpler: just use consecutive and check palindromic coefficients
                pass
        
        # Pattern 4: arithmetic progression with step > 1
        for step in [2, 3]:
            for base in range(0, 3):
                exps = [base + i * step for i in range(k)]
                if max(exps) > max_exp_kernel:
                    continue
                for coeffs in itertools.product(coeff_pool, repeat=k):
                    d = {}
                    for e, c in zip(exps, coeffs):
                        if c != 0:
                            d[e] = d.get(e, 0) + c
                    if len(d) == k:
                        kernels.append(d)
        
        return kernels
    
    # For each factorization of 12
    factorizations = [(1, 12), (2, 6), (3, 4), (4, 3), (6, 2), (12, 1)]
    
    for k, m in factorizations:
        if k * m != 12:
            continue
        if time.time() > deadline:
            break
        
        # Maximum step s such that max_degree <= 40
        # P = K * G, max degree of K is maxK, max degree of G is (m-1)*s
        # Total max degree: maxK + (m-1)*s <= 40
        # So s <= (40 - maxK) / (m-1) if m > 1
        
        kernels = generate_kernels(k, max_exp_kernel=15)
        
        # Deduplicate kernels
        seen_kernels = set()
        unique_kernels = []
        for kern in kernels:
            key = tuple(sorted(kern.items()))
            if key not in seen_kernels:
                seen_kernels.add(key)
                unique_kernels.append(kern)
        
        for kern in unique_kernels:
            if time.time() > deadline:
                break
            
            maxK = max(kern.keys())
            
            if m == 1:
                # P = K itself, just 12 terms
                if len(kern) != 12:
                    continue
                score = evaluate_from_dict(kern)
                if score is not None and score < best_score:
                    best_score = score
                    best_poly = dict_to_list(kern)
                continue
            
            # Try various steps s
            max_s = (40 - maxK) // (m - 1)
            min_s = 1
            
            for s in range(min_s, max_s + 1):
                if time.time() > deadline:
                    break
                
                # Build G(x) = 1 + x^s + x^{2s} + ... + x^{(m-1)s}
                G = {}
                for i in range(m):
                    e = i * s
                    G[e] = G.get(e, 0) + 1
                
                # P = K * G
                P = poly_mul(kern, G)
                
                # Check exactly 12 nonzero terms
                if len(P) != 12:
                    continue
                
                # Check coefficient bounds
                valid = True
                for c in P.values():
                    if not (-64 <= c <= 64):
                        valid = False
                        break
                if not valid:
                    continue
                
                score = evaluate_from_dict(P)
                if score is not None and score < best_score:
                    best_score = score
                    best_poly = dict_to_list(P)
    
    # ====================================================================
    # STRATEGY 2: Difference of geometric series
    # P(x) = (1 + x^s + x^{2s} + ... + x^{(m-1)s}) - (1 + x^t + x^{2t} + ...)
    # This gives 2m - 2 terms if no overlap (subtracting common terms).
    # For 12 terms: 2m - 2 = 12 -> m = 7.
    # P = G_7(s) - G_7(t) where G_m(step) = sum_{i=0}^{m-1} x^{i*step}
    # ====================================================================
    
    for s in range(1, 10):
        if time.time() > deadline:
            break
        for t in range(s + 1, 20):
            if time.time() > deadline:
                break
            
            G_s = {i * s: 1 for i in range(7)}
            G_t = {i * t: 1 for i in range(7)}
            
            P = poly_sub(G_s, G_t)
            
            if len(P) != 12:
                continue
            
            max_exp = max(P.keys())
            if max_exp > 40:
                continue
            
            valid = True
            for c in P.values():
                if not (-64 <= c <= 64):
                    valid = False
                    break
            if not valid:
                continue
            
            score = evaluate_from_dict(P)
            if score is not None and score < best_score:
                best_score = score
                best_poly = dict_to_list(P)
    
    # ====================================================================
    # STRATEGY 3: Product of two sparse polynomials with overlapping support
    # P(x) = A(x) * B(x) where A has a terms, B has b terms, a*b >= 12
    # but with cancellations giving exactly 12 terms.
    # 
    # Try A(x) = 1 + x + x^2 + x^3 (4 terms), B(x) = 1 + x^s + x^{2s} + x^{3s} (4 terms)
    # P = A * B has up to 16 terms, but if s is small, overlaps reduce count.
    # For 12 terms: need exactly 4 overlaps.
    # ====================================================================
    
    # A = 1 + x + x^2 + x^3
    A_base = {0: 1, 1: 1, 2: 1, 3: 1}
    
    for s in range(1, 10):
        if time.time() > deadline:
            break
        B_base = {i * s: 1 for i in range(4)}
        P = poly_mul(A_base, B_base)
        if len(P) == 12:
            max_exp = max(P.keys())
            if max_exp <= 40:
                valid = all(-64 <= c <= 64 for c in P.values())
                if valid:
                    score = evaluate_from_dict(P)
                    if score is not None and score < best_score:
                        best_score = score
                        best_poly = dict_to_list(P)
    
    # Try A = 1 + x + x^2 + x^3 + x^4 (5 terms), B with 3 terms
    A_5 = {i: 1 for i in range(5)}
    for s in range(1, 10):
        if time.time() > deadline:
            break
        B_3 = {i * s: 1 for i in range(3)}
        P = poly_mul(A_5, B_3)
        if len(P) == 12:
            max_exp = max(P.keys())
            if max_exp <= 40:
                valid = all(-64 <= c <= 64 for c in P.values())
                if valid:
                    score = evaluate_from_dict(P)
                    if score is not None and score < best_score:
                        best_score = score
                        best_poly = dict_to_list(P)
    
    # ====================================================================
    # STRATEGY 4: Cyclotomic kernels - specific known good patterns
    # 
    # Phi_1(x) = x - 1
    # Phi_2(x) = x + 1
    # Phi_3(x) = x^2 + x + 1
    # Phi_4(x) = x^2 + 1
    # Phi_5(x) = x^4 + x^3 + x^2 + x + 1
    # Phi_6(x) = x^2 - x + 1
    # Phi_7(x) = x^6 + x^5 + ... + 1
    # Phi_8(x) = x^4 + 1
    # Phi_9(x) = x^6 + x^3 + 1
    # Phi_10(x) = x^4 - x^3 + x^2 - x + 1
    # Phi_12(x) = x^4 - x^2 + 1
    # 
    # Products of cyclotomic polynomials have structured squares.
    # ====================================================================
    
    cyclotomics = [
        {0: -1, 1: 1},           # x - 1
        {0: 1, 1: 1},            # x + 1
        {0: 1, 1: 1, 2: 1},      # x^2 + x + 1
        {0: 1, 2: 1},            # x^2 + 1
        {0: 1, 1: 1, 2: 1, 3: 1, 4: 1},  # x^4 + x^3 + x^2 + x + 1
        {0: 1, 1: -1, 2: 1},     # x^2 - x + 1
        {0: 1, 2: -1, 4: 1},     # x^4 - x^2 + 1
        {0: 1, 1: -1, 2: 1, 3: -1, 4: 1},  # x^4 - x^3 + x^2 - x + 1
    ]
    
    # Try products of 2 or 3 cyclotomic polynomials
    for r in range(2, 4):
        for combo in itertools.combinations_with_replacement(range(len(cyclotomics)), r):
            if time.time() > deadline:
                break
            P = {0: 1}
            for idx in combo:
                P = poly_mul(P, cyclotomics[idx])
            
            # P might have more or fewer than 12 terms
            # If it has <= 12 terms, we can try to extend with a geometric series
            if len(P) > 12:
                continue
            
            # Try multiplying by a geometric series to get exactly 12 terms
            current_terms = len(P)
            if current_terms == 0:
                continue
            
            # Need 12 / current_terms more terms from G
            if 12 % current_terms != 0:
                continue
            
            m_needed = 12 // current_terms
            maxP = max(P.keys()) if P else 0
            
            for s in range(1, 10):
                if time.time() > deadline:
                    break
                if (m_needed - 1) * s + maxP > 40:
                    break
                
                G = {i * s: 1 for i in range(m_needed)}
                P_ext = poly_mul(P, G)
                
                if len(P_ext) != 12:
                    continue
                
                max_exp = max(P_ext.keys())
                if max_exp > 40:
                    continue
                
                valid = all(-64 <= c <= 64 for c in P_ext.values())
                if not valid:
                    continue
                
                score = evaluate_from_dict(P_ext)
                if score is not None and score < best_score:
                    best_score = score
                    best_poly = dict_to_list(P_ext)
    
    # ====================================================================
    # STRATEGY 5: "Anti-palindromic" construction
    # P(x) = Q(x) - x^k * Q(x^{-1}) where Q has m terms
    # This gives at most 2m - 1 terms (if no overlap at the center).
    # For 12 terms: 2m - 1 = 12 -> not integer. 2m = 12 -> m = 6, no overlap.
    # Or 2m - 2 = 12 (one overlap) -> m = 7.
    # ====================================================================
    
    # P(x) = Q(x) - x^k * Q(x^{-1})
    # If Q = sum_{i=0}^{m-1} c_i x^i, then x^k * Q(x^{-1}) = sum c_i x^{k-i}
    # P has terms at exponents {0..m-1} union {k-m+1..k}
    # For no overlap: k >= m. Terms: 2m. For 12: m=6, k>=6.
    
    for m in [5, 6, 7]:
        if time.time() > deadline:
            break
        # Q has m terms, consecutive exponents 0..m-1
        # Try various coefficient patterns
        for coeffs in itertools.product([-1, 1], repeat=m):
            if time.time() > deadline:
                break
            
            Q = {i: coeffs[i] for i in range(m)}
            
            # k must be >= m for no overlap, or chosen for specific overlap
            for k in range(m, min(40 - m + 1, m + 10)):
                # x^k * Q(x^{-1}) = sum c_i x^{k-i}
                R = {k - i: coeffs[i] for i in range(m)}
                
                P = poly_sub(Q, R)
                
                if len(P) != 12:
                    continue
                
                max_exp = max(P.keys())
                if max_exp > 40:
                    continue
                
                valid = all(-64 <= c <= 64 for c in P.values())
                if not valid:
                    continue
                
                score = evaluate_from_dict(P)
                if score is not None and score < best_score:
                    best_score = score
                    best_poly = dict_to_list(P)
    
    # ====================================================================
    # STRATEGY 6: Targeted search on "sum-free" exponent sets
    # Choose 12 exponents such that all pairwise sums are distinct (Sidon set).
    # Then P^2 has exactly C(12,2) + 12 = 66 + 12 = 78 terms... no, that's max.
    # Actually for a Sidon set, all sums e_i + e_j (i<=j) are distinct,
    # so P^2 has exactly 12*13/2 = 78 terms. That's the MAXIMUM, not minimum.
    # We want the OPPOSITE: many collisions to cause cancellation.
    # 
    # So we want exponents with MANY repeated pairwise sums.
    # Consecutive exponents maximize collisions.
    # ====================================================================
    
    # For consecutive exponents 0..11, try more coefficient patterns
    # using a smarter search: coordinate descent with lookahead.
    
    def exact_square_count_from_dict(d):
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
    
    # Coordinate descent on consecutive exponents 0..11
    random.seed(123)
    
    for restart in range(200):
        if time.time() > deadline:
            break
        
        curr = {}
        for e in range(12):
            curr[e] = random.choice([-2, -1, 1, 2])
        
        curr_score = exact_square_count_from_dict(curr)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for e in range(12):
                old_val = curr[e]
                for new_val in [-3, -2, -1, 1, 2, 3]:
                    if new_val == old_val:
                        continue
                    curr[e] = new_val
                    new_score = exact_square_count_from_dict(curr)
                    if new_score < curr_score:
                        curr_score = new_score
                        improved = True
                        if curr_score < best_score:
                            best_score = curr_score
                            best_poly = dict_to_list(curr)
                    else:
                        curr[e] = old_val
    
    # ====================================================================
    # STRATEGY 7: Use the identity (a+b+c+d)^2 structure
    # If P = A + B where A and B are sparse with disjoint supports,
    # P^2 = A^2 + 2AB + B^2.
    # If A^2, B^2, and AB have overlapping supports, cancellation occurs.
    # 
    # Try: A = x^0 + x^3 + x^6 + x^9 + x^12 + x^15 (6 terms, step 3)
    #      B = x^1 + x^4 + x^7 + x^10 + x^13 + x^16 (6 terms, step 3)
    # Already analyzed: no overlap in P^2 supports. Bad.
    # 
    # Try: A = x^0 + x^2 + x^4 + x^6 + x^8 + x^10 (even, 6 terms)
    #      B = x^1 + x^3 + x^5 + x^7 + x^9 + x^11 (odd, 6 terms)
    # P = A + B = 1 - x + x^2 - x^3 + ... (alternating, 12 terms)
    # This is the alternating pattern. Let's compute its square.
    # ====================================================================
    
    # Compute square of alternating polynomial
    d_alt = {i: (1 if i % 2 == 0 else -1) for i in range(12)}
    score_alt = evaluate_from_dict(d_alt)
    if score_alt is not None and score_alt < best_score:
        best_score = score_alt
        best_poly = dict_to_list(d_alt)
    
    # ====================================================================
    # STRATEGY 8: "Perfect square minus small correction"
    # Find a 12-term P such that P^2 is close to a perfect square of a
    # lower-degree polynomial. If P^2 = Q^2 + R where R has few terms,
    # then P^2 has few terms.
    # 
    # Actually, we want P^2 to have few terms directly.
    # 
    # Try: P(x) = (1 + x + x^2)^2 - x^k * C(x)
    # (1+x+x^2)^2 = 1 + 2x + 3x^2 + 2x^3 + x^4 (5 terms)
    # Not 12 terms.
    # ====================================================================
    
    # ====================================================================
    # FINAL: If we haven't found anything, use the best from any strategy
    # ====================================================================
    
    if best_poly is None:
        # Fallback: 12 consecutive terms with alternating signs
        best_poly = [1 if i % 2 == 0 else -1 for i in range(12)]
    
    # Final validation
    final_dict = {i: best_poly[i] for i in range(len(best_poly)) if best_poly[i] != 0}
    final_score = evaluate_from_dict(final_dict)
    if final_score is None:
        best_poly = [1] * 12
    
    return best_poly

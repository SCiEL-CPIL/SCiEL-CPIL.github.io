import time
import itertools
from collections import defaultdict
import random

def _square_nonzero_count(exponents, coefficients):
    """Compute P(x)^2 from sparse representation and count nonzero coefficients."""
    n = len(exponents)
    square = defaultdict(int)
    for i in range(n):
        for j in range(i, n):
            e = exponents[i] + exponents[j]
            c = coefficients[i] * coefficients[j]
            if i != j:
                c *= 2
            square[e] += c
    return sum(1 for c in square.values() if c != 0)

def _build_coeff_list(exponents, coefficients, max_degree=40):
    """Build a coefficient list in ascending degree from sparse representation."""
    if not exponents:
        return None
    max_exp = max(exponents)
    if max_exp > max_degree:
        return None
    result = [0] * (max_exp + 1)
    for e, c in zip(exponents, coefficients):
        result[e] = c
    nonzero = sum(1 for x in result if x != 0)
    if nonzero != 12:
        return None
    if result[-1] == 0:
        return None
    if any(abs(x) > 64 for x in result):
        return None
    if len(result) > 41:
        return None
    return result

def _optimize_signs(exponents, initial_coeffs, max_iterations=50):
    """Optimize signs to minimize nonzero terms in square."""
    coeffs = list(initial_coeffs)
    current_score = _square_nonzero_count(exponents, coeffs)
    
    for _ in range(max_iterations):
        improved = False
        for i in range(len(coeffs)):
            old_coeff = coeffs[i]
            coeffs[i] = -coeffs[i]
            new_score = _square_nonzero_count(exponents, coeffs)
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                coeffs[i] = old_coeff
        if not improved:
            break
    
    return coeffs, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # Fallback: [1]*12 has 21 nonzero terms in square
    fallback = [1] * 12
    best_score = 21
    best_candidate = fallback
    
    # Strategy 1: Factorization-based construction
    # Generate small polynomials P and Q with known sparse squares,
    # then form P*Q to get a 12-term polynomial with structured cancellation
    
    # Generate candidate small polynomials (2-6 terms each)
    small_polys = []
    
    # Arithmetic progressions with various steps
    for num_terms in [2, 3, 4, 5, 6]:
        for start in range(0, 5):
            for step in range(1, 4):
                exps = tuple(start + i * step for i in range(num_terms))
                if max(exps) <= 30:
                    small_polys.append(exps)
    
    # Dense clusters
    for base in range(0, 8):
        for combo in itertools.combinations(range(0, 10), 5):
            exps = tuple(sorted([base] + list(combo)))
            if len(set(exps)) == 6 and max(exps) <= 30:
                small_polys.append(exps)
    
    # Try all pairs of small polynomials whose product has exactly 12 terms
    # P*Q has |P|*|Q| terms if no collisions, so we want |P|*|Q| = 12
    # Possible: (2,6), (3,4), (4,3), (6,2)
    
    candidate_pairs = []
    
    for P_exps in small_polys:
        P_len = len(P_exps)
        if P_len not in [2, 3, 4, 6]:
            continue
        
        for Q_exps in small_polys:
            Q_len = len(Q_exps)
            if P_len * Q_len != 12:
                continue
            
            # Compute P*Q
            product_exps = []
            for p in P_exps:
                for q in Q_exps:
                    product_exps.append(p + q)
            
            # Check if product has exactly 12 distinct exponents
            if len(set(product_exps)) != 12:
                continue
            
            product_exps = sorted(set(product_exps))
            if max(product_exps) > 40:
                continue
            
            # Check if coefficients can be assigned to stay in [-64, 64]
            # For now, assume all coefficients are ±1
            # P*Q with all ±1 coefficients will have coefficients in [-6, 6]
            # (max 6 terms in P times max 6 terms in Q, but actually bounded by min(|P|,|Q|))
            
            candidate_pairs.append((P_exps, Q_exps, product_exps))
    
    # Sort candidate pairs by some heuristic (e.g., sum of self-square densities)
    # For now, just iterate through them
    
    for P_exps, Q_exps, product_exps in candidate_pairs[:500]:  # Limit to first 500
        if time.time() > deadline:
            break
        
        # Try different sign assignments for P and Q
        # We want to minimize the number of nonzero terms in (P*Q)^2
        # This is equivalent to minimizing collisions in the square of the product
        
        # Start with all positive coefficients
        P_coeffs = [1] * len(P_exps)
        Q_coeffs = [1] * len(Q_exps)
        
        # Compute P*Q coefficients
        product_coeffs = defaultdict(int)
        for i, p in enumerate(P_exps):
            for j, q in enumerate(Q_exps):
                e = p + q
                product_coeffs[e] += P_coeffs[i] * Q_coeffs[j]
        
        # Convert to sparse representation
        prod_sparse_exps = sorted(product_coeffs.keys())
        prod_sparse_coeffs = [product_coeffs[e] for e in prod_sparse_exps]
        
        # Optimize signs of the product polynomial
        optimized_coeffs, score = _optimize_signs(prod_sparse_exps, prod_sparse_coeffs)
        
        if score < best_score:
            best_score = score
            candidate = _build_coeff_list(prod_sparse_exps, optimized_coeffs)
            if candidate is not None:
                best_candidate = candidate
    
    # Strategy 2: Direct search with structured exponent sets
    # Generate 12-term exponent sets with high collision potential
    
    # Method: Use two interleaved arithmetic progressions
    for step1 in range(1, 6):
        for step2 in range(1, 6):
            if step1 == step2:
                continue
            
            # Try to interleave two APs
            exps = []
            for i in range(6):
                exps.append(i * step1)
                if step2 != step1:
                    exps.append(i * step2)
            
            exps = sorted(set(exps))
            if len(exps) != 12:
                continue
            
            if max(exps) > 40:
                continue
            
            coeffs = [1] * 12
            optimized_coeffs, score = _optimize_signs(exps, coeffs)
            
            if score < best_score:
                best_score = score
                candidate = _build_coeff_list(exps, optimized_coeffs)
                if candidate is not None:
                    best_candidate = candidate
    
    # Strategy 3: Random search with sign optimization
    random.seed(42)
    for _ in range(2000):
        if time.time() > deadline:
            break
        
        # Random 12-term poly
        exps = random.sample(range(0, 41), 12)
        exps.sort()
        coeffs = [random.choice([-1, 1]) for _ in range(12)]
        
        optimized_coeffs, score = _optimize_signs(exps, coeffs)
        
        if score < best_score:
            best_score = score
            candidate = _build_coeff_list(exps, optimized_coeffs)
            if candidate is not None:
                best_candidate = candidate
    
    if best_score < 21:
        return best_candidate
    return fallback

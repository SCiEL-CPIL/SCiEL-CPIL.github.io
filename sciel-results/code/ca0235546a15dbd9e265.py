import time
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START

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

def construct():
    start_time = time.time()
    deadline = start_time + 80.0 # Safety margin for 90s limit

    # Fallback: [1]*12 has 21 nonzero terms in square
    fallback = [1] * 12
    best_score = 21
    best_candidate = fallback

    # 1. Generate candidate 6-term exponent sets
    # We want sets with "dense" self-squares to maximize collisions within A^2 and B^2
    candidate_6_exps = set()
    
    # Arithmetic progressions
    for start in range(0, 10):
        for step in range(1, 8):
            exps = tuple(start + i * step for i in range(6))
            if max(exps) <= 40:
                candidate_6_exps.add(exps)
                
    # Small clusters
    for base in range(0, 10):
        for s in itertools.combinations_with_replacement(range(0, 10), 5):
             exps = tuple(sorted([base] + [base + x for x in s]))
             if len(set(exps)) == 6 and max(exps) <= 40:
                 candidate_6_exps.add(exps)

    # 2. Generate pairs (A_exps, B_exps) with disjoint supports
    # To manage complexity, we iterate over A_exps and find compatible B_exps
    # We prioritize pairs where the "ranges" are separated or interlaced nicely
    # But strictly disjoint supports is the key for the structural argument.
    
    # Pre-compute self-square nonzero counts for all 6-term sets with all-1 coeffs
    # This is a static property of the exponent set
    self_sq_cache = {}
    for exps in candidate_6_exps:
        # Count nonzero in square of poly with coeffs [1,1,1,1,1,1]
        sq = defaultdict(int)
        for i in range(6):
            for j in range(i, 6):
                e = exps[i] + exps[j]
                c = 1
                if i != j:
                    c = 2
                sq[e] += c
        cnt = sum(1 for v in sq.values() if v != 0)
        self_sq_cache[exps] = cnt

    # Sort 6-term sets by self-square count (ascending)
    sorted_6_exps = sorted(candidate_6_exps, key=lambda x: self_sq_cache[x])
    
    # We only need to check top K sets to keep runtime low
    # Top sets are likely APs or dense clusters
    top_6_exps = sorted_6_exps[:200]
    
    best_found_in_structure = False
    
    # Iterate over pairs of 6-term sets
    # To avoid O(N^2) explosion with N=200, we can be smart or just limit iterations
    # 200 * 200 = 40,000 pairs. For each, we need to check disjointness and compute square.
    # Computing square of 12 terms is O(12^2) = 144 ops.
    # 40,000 * 144 = ~5.7M ops. This is fast in Python if optimized.
    
    for i in range(len(top_6_exps)):
        if time.time() > deadline:
            break
        A_exps = top_6_exps[i]
        A_set = set(A_exps)
        
        for j in range(i, len(top_6_exps)):
            if time.time() > deadline:
                break
            B_exps = top_6_exps[j]
            B_set = set(B_exps)
            
            # Check disjointness
            if A_set & B_set:
                continue
            
            # Combine
            # We assume coeffs are 1 for now to find structural minimum
            # P = A + B
            # P^2 = A^2 + 2AB + B^2
            # Since supports are disjoint, A^2 and B^2 supports are subsets of sums of A and sums of B.
            # 2AB support is sums of A+B.
            # If these three sets of exponents are disjoint, total nonzero = |A^2| + |B^2| + |2AB|
            # |2AB| is at most 36, but usually less due to collisions in A+B.
            
            # Let's just compute the actual square to be safe, as collisions might happen
            # between A^2 and 2AB if not carefully separated.
            
            all_exps = sorted(list(A_exps) + list(B_exps))
            all_coeffs = [1] * 12
            
            score = _square_nonzero_count(all_exps, all_coeffs)
            
            if score < best_score:
                best_score = score
                candidate = _build_coeff_list(all_exps, all_coeffs)
                if candidate is not None:
                    best_candidate = candidate
                    best_found_in_structure = True

    # 3. Refine the best structural candidate found (if any)
    # If we found a good structure, try to optimize coefficients (signs/magnitudes)
    # to reduce further via cancellation.
    
    if best_found_in_structure:
        # We need to recover the exponents/coeffs of best_candidate to optimize
        # But best_candidate is a list. Let's re-derive sparse form.
        exps = []
        coeffs = []
        for i, c in enumerate(best_candidate):
            if c != 0:
                exps.append(i)
                coeffs.append(c)
        
        # Simple sign flipping optimization
        current_score = best_score
        for _ in range(10): # Max 10 iterations of sign flips
            if time.time() > deadline:
                break
            improved = False
            for i in range(12):
                coeffs[i] *= -1
                new_score = _square_nonzero_count(exps, coeffs)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    coeffs[i] *= -1
            if not improved:
                break
        
        if current_score < best_score:
            best_score = current_score
            candidate = _build_coeff_list(exps, coeffs)
            if candidate is not None:
                best_candidate = candidate

    # 4. Fallback random search if structural search didn't beat fallback
    # This is a safety net.
    if best_score >= 21:
        import random
        random.seed(42)
        for _ in range(1000):
            if time.time() > deadline:
                break
            # Random 12-term poly
            exps = random.sample(range(0, 41), 12)
            exps.sort()
            coeffs = [random.choice([-1, 1]) for _ in range(12)]
            score = _square_nonzero_count(exps, coeffs)
            if score < best_score:
                best_score = score
                candidate = _build_coeff_list(exps, coeffs)
                if candidate is not None:
                    best_candidate = candidate

    if best_score < 21:
        return best_candidate
    return fallback

# EVOLVE-BLOCK-END

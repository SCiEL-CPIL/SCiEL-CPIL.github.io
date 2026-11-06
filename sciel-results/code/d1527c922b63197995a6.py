"""
Strategy: Structured Algebraic Construction + Stochastic Local Search
Instead of brute-forcing signs on fixed APs, we use known algebraic identities
(e.g., sum of squares, recursive products) to generate sparse squares, and then
refine them using local search on coefficients and exponents.
"""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _square_count(coeffs_list):
    """
    Calculate the number of non-zero terms in the square of a polynomial
    represented as a list of coefficients (index = degree).
    """
    if not coeffs_list:
        return 0
    # Use numpy convolve for speed
    sq = np.convolve(coeffs_list, coeffs_list)
    return np.count_nonzero(sq)

def _construct_from_ap(signs, coeffs_vals, start_exp=0, step=1):
    """
    Construct a polynomial list from a list of signs and coefficient magnitudes
    on an arithmetic progression of exponents.
    """
    n = len(signs)
    if n == 0:
        return []
    max_exp = start_exp + step * (n - 1)
    poly = [0] * (max_exp + 1)
    for i in range(n):
        exp = start_exp + step * i
        if exp < len(poly):
            poly[exp] = signs[i] * coeffs_vals[i]
    return poly

def _generate_identity_candidates():
    """
    Generate candidates based on known algebraic identities that yield sparse squares.
    1. (a + b)^2 = a^2 + 2ab + b^2 (3 terms)
    2. (a^2 + b^2)^2 = a^4 + 2a^2b^2 + b^4 (3 terms)
    3. Fermat's identity for sums of squares.
    4. Recursive construction: If P has sparse square, maybe P * (1+x^k) helps?
    
    We will generate a diverse set of starting points:
    - Random sparse polynomials
    - Polynomials with symmetric coefficients
    - Polynomials derived from (1+x)^k expansions with pruning
    """
    candidates = []
    
    # 1. Simple AP with random signs
    for _ in range(50):
        signs = [1 if random.random() > 0.5 else -1 for _ in range(12)]
        # Try different steps
        step = random.choice([1, 2, 3])
        # Coefficients mostly 1, occasionally 2
        coeffs = [1 if random.random() > 0.2 else 2 for _ in range(12)]
        poly = _construct_from_ap(signs, coeffs, 0, step)
        if len(poly) <= 41 and poly[-1] != 0:
             # Ensure exactly 12 non-zeros
             if sum(1 for x in poly if x != 0) == 12:
                 candidates.append(poly)

    # 2. Symmetric Polynomials
    # P(x) = sum c_i x^i where c_i = c_{n-i}
    for _ in range(50):
        n_terms = 12
        # Create symmetric coefficient vector
        # We need 12 non-zeros.
        # Let's place them symmetrically around a center.
        center = random.randint(0, 10)
        exps = sorted(random.sample(range(0, 41), 12))
        # Ensure symmetry? Hard to enforce exactly 12 terms with symmetry easily without duplicates.
        # Instead, try specific structures.
        
    # 3. "Clustered" Exponents
    # Group terms into clusters where intra-cluster cancellation is possible.
    # e.g., 3 clusters of 4 terms.
    for _ in range(50):
        clusters = [3, 4, 5] # Sum to 12
        random.shuffle(clusters)
        exps = []
        current_exp = 0
        for c_size in clusters:
            # Random gap between clusters
            gap = random.randint(1, 5)
            if current_exp > 0:
                current_exp += gap
            # Within cluster, use small steps (1 or 2)
            step = random.choice([1, 2])
            for _ in range(c_size):
                exps.append(current_exp)
                current_exp += step
        # Assign random signs and small coeffs
        signs = [1 if random.random() > 0.5 else -1 for _ in range(12)]
        coeffs = [1 if random.random() > 0.3 else 2 for _ in range(12)]
        poly = [0] * (max(exps) + 1)
        for i in range(12):
            poly[exps[i]] = signs[i] * coeffs[i]
        if poly[-1] != 0 and sum(1 for x in poly if x != 0) == 12:
            candidates.append(poly)

    # 4. Specific Algebraic Identities
    # (1 + x + x^2)^2 = 1 + 2x + 3x^2 + 2x^3 + x^4 (5 terms)
    # We need 12 terms in P, so P could be a product of such blocks?
    # P = (1+x+x^2)(1+x^3+x^6)...
    # Let's try P = (1+x)^11? No, that has 12 terms but square has 23.
    # Let's try P = (1+x^k) * Q where Q has 11 terms.
    
    # Add a few hardcoded good structures found in literature/heuristics
    # e.g. Alternating signs on AP often cancel well.
    for step in [1, 2, 3, 4]:
        signs = [1 if i % 2 == 0 else -1 for i in range(12)]
        poly = _construct_from_ap(signs, [1]*12, 0, step)
        if len(poly) <= 41 and poly[-1] != 0:
            candidates.append(poly)
            
    return candidates

def _local_search(poly, max_iters=5000, deadline=None):
    """
    Perform local search to minimize the number of non-zero terms in the square.
    Moves:
    1. Flip sign of a coefficient.
    2. Change magnitude of a coefficient (1 <-> 2 <-> -1 <-> -2).
    3. Shift exponent of a term (if space allows).
    """
    current_poly = poly[:]
    current_score = _square_count(current_poly)
    best_poly = current_poly[:]
    best_score = current_score
    
    # Pre-compute indices of non-zero terms
    nz_indices = [i for i, v in enumerate(current_poly) if v != 0]
    
    # To speed up, we can use a dictionary representation for the square calculation
    # But numpy convolve is fast enough for length <= 41.
    
    temperature = 1.0
    cooling_rate = 0.999
    
    for i in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        # Pick a random move
        move_type = random.random()
        
        if move_type < 0.5:
            # Flip sign or change value of a non-zero term
            if not nz_indices:
                break
            idx = random.choice(nz_indices)
            old_val = current_poly[idx]
            # Try other small values: -2, -1, 1, 2
            new_val = random.choice([-2, -1, 1, 2])
            if new_val == old_val:
                continue
            current_poly[idx] = new_val
            new_score = _square_count(current_poly)
            
            # Accept if better, or with prob exp(-delta/T)
            delta = new_score - current_score
            if delta < 0 or (temperature > 0 and random.random() < np.exp(-delta / temperature)):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_poly = current_poly[:]
            else:
                current_poly[idx] = old_val
                
        elif move_type < 0.8:
            # Shift exponent
            if not nz_indices:
                break
            idx = random.choice(nz_indices)
            val = current_poly[idx]
            # Try shifting left or right by 1
            shift = random.choice([-1, 1])
            new_idx = idx + shift
            if new_idx < 0 or new_idx >= len(current_poly):
                # If shifting out of bounds, we might need to extend the list?
                # Task says length <= 41. If we shift out, we might violate length or non-zero final.
                # Let's only allow shifts within current bounds to keep it simple.
                continue
            # Check if target is occupied
            if current_poly[new_idx] != 0:
                continue
            
            current_poly[idx] = 0
            current_poly[new_idx] = val
            new_score = _square_count(current_poly)
            
            delta = new_score - current_score
            if delta < 0 or (temperature > 0 and random.random() < np.exp(-delta / temperature)):
                current_score = new_score
                nz_indices.remove(idx)
                nz_indices.append(new_idx)
                if new_score < best_score:
                    best_score = new_score
                    best_poly = current_poly[:]
            else:
                current_poly[new_idx] = 0
                current_poly[idx] = val
                
        else:
            # Swap two non-zero terms' positions?
            if len(nz_indices) < 2:
                break
            i1, i2 = random.sample(nz_indices, 2)
            v1, v2 = current_poly[i1], current_poly[i2]
            current_poly[i1], current_poly[i2] = v2, v1
            new_score = _square_count(current_poly)
            
            delta = new_score - current_score
            if delta < 0 or (temperature > 0 and random.random() < np.exp(-delta / temperature)):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_poly = current_poly[:]
            else:
                current_poly[i1], current_poly[i2] = v1, v2
                
        temperature *= cooling_rate

    return best_poly, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # 1. Generate initial candidates
    candidates = _generate_identity_candidates()
    
    # Add some purely random ones
    for _ in range(20):
        # Random sparse poly
        exps = sorted(random.sample(range(0, 41), 12))
        signs = [1 if random.random() > 0.5 else -1 for _ in range(12)]
        coeffs = [1 if random.random() > 0.5 else 2 for _ in range(12)]
        poly = [0] * 41
        for i in range(12):
            poly[exps[i]] = signs[i] * coeffs[i]
        if poly[-1] != 0:
            candidates.append(poly)
    
    if not candidates:
        # Fallback: simple AP
        return [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]

    best_poly = None
    best_score = 1000
    
    # 2. Evaluate and refine each candidate
    for cand in candidates:
        if time.time() > deadline - 5.0:
            break
            
        # Basic validation
        if len(cand) > 41 or cand[-1] == 0:
            continue
        if sum(1 for x in cand if x != 0) != 12:
            continue
        if any(abs(x) > 64 for x in cand):
            continue
            
        # Quick score
        score = _square_count(cand)
        if score < best_score:
            best_score = score
            best_poly = cand[:]
            
        # Local search
        if time.time() < deadline - 10.0:
            # Allocate time based on potential
            # If score is already very good (e.g., < 15), spend more time?
            # Or just spend fixed time per candidate.
            # Let's spend up to 2 seconds per candidate for local search.
            cand_deadline = time.time() + 2.0
            refined_poly, refined_score = _local_search(cand, max_iters=2000, deadline=cand_deadline)
            
            if refined_score < best_score:
                best_score = refined_score
                best_poly = refined_poly[:]
                
        # If we found a very good score, we can stop early or do more
        if best_score <= 14:
             # 14 is very good. Minimum possible?
             # Square of 12 terms.
             # If P = sum c_i x^{e_i}, P^2 = sum c_i^2 x^{2e_i} + 2 sum_{i<j} c_i c_j x^{e_i+e_j}
             # To minimize terms, we want many collisions.
             # If all e_i are distinct, we have 12 diagonal terms.
             # Off-diagonal: 11*12/2 = 66 pairs.
             # If we can make all off-diagonal sums collide with each other or diagonals?
             # For AP 0..11, sums are 0..22.
             # Diagonals: 0, 2, 4, ..., 22 (12 terms).
             # Off-diagonals: 1, 3, ..., 21 (11 terms).
             # Total 23 terms if no cancellation.
             # With signs, we can cancel.
             # (1-x+x^2-x^3+...)^2
             # This is a known problem.
             # If we find 12, that's likely optimal or near-optimal.
             break

    if best_poly is None:
        # Fallback
        return [1] * 12
        
    return best_poly

# EVOLVE-BLOCK-END

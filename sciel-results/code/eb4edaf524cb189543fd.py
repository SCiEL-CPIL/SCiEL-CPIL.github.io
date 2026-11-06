import time
import numpy as np
import random

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    # Fallback: (1 + x + ... + x^11)^2 has 21 non-zero terms
    fallback = [1] * 12
    fallback_score = 21
    
    best_score = fallback_score
    best_candidate = fallback

    # Pre-define small coefficient values to try
    COEFF_VALUES = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]

    def compute_score(exps, coeffs):
        """
        Computes the number of non-zero terms in P(x)^2.
        exps: list of 12 exponents
        coeffs: list of 12 coefficients
        """
        if not exps or not coeffs:
            return float('inf')
        
        max_deg = max(exps)
        if max_deg > 40:
            return float('inf')
        if any(abs(c) > 64 for c in coeffs):
            return float('inf')
        if any(c == 0 for c in coeffs):
            return float('inf')
        if len(set(exps)) != 12:
            return float('inf')
            
        p = np.zeros(max_deg + 1, dtype=np.int64)
        for e, c in zip(exps, coeffs):
            p[e] = c
            
        q = np.convolve(p, p)
        count = np.count_nonzero(q)
        return count

    def _update_best(exps, coeffs):
        """Update the global best candidate dense list."""
        nonlocal best_score, best_candidate
        score = compute_score(exps, coeffs)
        if score < best_score:
            max_deg = max(exps)
            dense = [0] * (max_deg + 1)
            for e, c in zip(exps, coeffs):
                dense[e] = c
            if len(dense) <= 41 and dense[-1] != 0:
                best_score = score
                best_candidate = dense

    def local_search(exps, coeffs, max_iters=1000):
        """
        Local search to minimize non-zero terms in square.
        Moves:
        1. Change coefficient value.
        2. Shift an exponent by +/- 1 (if valid and no collision).
        """
        nonlocal best_score, best_candidate
        
        current_score = compute_score(exps, coeffs)
        _update_best(exps, coeffs)
        
        no_improve_count = 0
        iterations = 0
        
        # Sort exps and coeffs together initially
        sorted_idx = np.argsort(exps)
        exps = [exps[i] for i in sorted_idx]
        coeffs = [coeffs[i] for i in sorted_idx]
        
        while iterations < max_iters and (time.time() < deadline - 2):
            iterations += 1
            improved = False
            
            # 1. Try changing coefficients
            for i in range(12):
                for new_c in COEFF_VALUES:
                    if new_c == coeffs[i]:
                        continue
                    old_c = coeffs[i]
                    coeffs[i] = new_c
                    new_score = compute_score(exps, coeffs)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        no_improve_count = 0
                        _update_best(exps, coeffs)
                    else:
                        coeffs[i] = old_c
                        
            if improved:
                continue
                
            # 2. Try shifting exponents
            # We need to be careful with shifting because it changes the structure.
            # We iterate over a copy of indices to avoid modifying list size/structure during iteration
            # but since we just change values, it's fine.
            for i in range(12):
                for delta in [-1, 1]:
                    new_e = exps[i] + delta
                    if new_e < 0 or new_e > 40:
                        continue
                    if new_e in exps: # Collision check
                        continue
                        
                    old_e = exps[i]
                    exps[i] = new_e
                    
                    new_score = compute_score(exps, coeffs)
                    
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        no_improve_count = 0
                        _update_best(exps, coeffs)
                        # Re-sort state to keep it tidy for next iteration?
                        # If we re-sort, we need to move coeff too.
                        idx = np.argsort(exps)
                        exps = [exps[j] for j in idx]
                        coeffs = [coeffs[j] for j in idx]
                    else:
                        exps[i] = old_e
                        
            if not improved:
                no_improve_count += 1
                if no_improve_count > 50:
                    break
            else:
                no_improve_count = 0

    def split_search():
        """
        Structural strategy: P = A + B where A and B are 6-term polynomials.
        We enforce disjoint supports for A and B to simplify cross-term analysis,
        but here we just use it to generate diverse initial candidates and
        potentially constrain the search space for some restarts.
        Actually, the local search above is more powerful. 
        This function generates specific structured candidates based on the split idea.
        """
        candidates = []
        
        # Pattern: A has even exponents, B has odd exponents?
        # Or A has low exponents, B has high exponents.
        
        # Candidate 1: Low/High split
        # A: 0..5, B: 6..11
        # Try alternating signs to encourage cancellation in cross terms if they overlap?
        # But supports are disjoint, so cross terms 2AB are at exponents sum(a,b).
        # If A={0..5}, B={6..11}, cross terms are at 6..16.
        # A^2 is at 0..10. B^2 is at 12..22.
        # Overlaps: Cross(6..16) overlaps with A^2(6..10) and B^2(12..16).
        # So we can tune coefficients to cancel these.
        
        exps = list(range(12))
        # Initial guess: 
        # c0..c5 for A, c6..c11 for B
        # Try all 1s
        candidates.append((exps[:], [1]*12))
        
        # Try alternating signs
        candidates.append((exps[:], [1 if i%2==0 else -1 for i in range(12)]))
        
        # Try block signs
        candidates.append((exps[:], [1]*6 + [-1]*6))
        
        # Candidate 2: Interleaved
        # A: 0,2,4,6,8,10 ; B: 1,3,5,7,9,11
        exps = list(range(12))
        candidates.append((exps[:], [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]))
        
        # Candidate 3: Random sparse
        random.seed(123)
        for _ in range(10):
            # Pick 12 distinct exponents from 0..40
            exps = sorted(random.sample(range(0, 41), 12))
            coeffs = [random.choice([-1, 1]) for _ in range(12)]
            candidates.append((exps, coeffs))
            
        return candidates

    # --- Generate Initial Candidates ---
    
    candidates = []
    
    # 1. Consecutive exponents starting at k
    for start in range(0, 5):
        exps = [start + i for i in range(12)]
        if max(exps) <= 40:
            # Try various initial coefficient patterns
            for pattern in [
                [1]*12,
                [1 if i%2==0 else -1 for i in range(12)],
                [1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1], # Block pattern
                [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1], # Alternating
                [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1]  # Pairs
            ]:
                candidates.append((exps[:], pattern[:]))

    # 2. Gaps
    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
    exps_gap1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
    candidates.append((exps_gap1, [1]*12))
    candidates.append((exps_gap1, [1 if i%2==0 else -1 for i in range(12)]))
    
    # 3. Wider gaps
    exps_gap2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13]
    candidates.append((exps_gap2, [1]*12))
    
    # 4. Structured splits (from split_search idea)
    candidates.extend(split_search())

    # Run local search on each candidate
    for exps, coeffs in candidates:
        if time.time() > deadline - 5:
            break
        local_search(exps, coeffs)
        
        if best_score <= 12:
            break

    return best_candidate

# EVOLVE-BLOCK-END

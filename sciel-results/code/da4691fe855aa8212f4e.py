import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    best_score = float('inf')
    best_candidate = None

    def compute_square_nonzeros_fast(pairs):
        """
        pairs: list of 12 (exp, coeff) tuples.
        Returns number of non-zero terms in the square.
        """
        if len(pairs) != 12:
            return float('inf')
        
        max_deg = max(e for e, c in pairs)
        if max_deg > 40:
            return float('inf')
            
        p = np.zeros(max_deg + 1, dtype=np.int64)
        for e, c in pairs:
            if c == 0:
                return float('inf')
            if abs(c) > 64:
                return float('inf')
            p[e] = c
        
        square = np.convolve(p, p)
        return int(np.count_nonzero(square))

    def update_best(pairs, score):
        nonlocal best_score, best_candidate
        if score < best_score:
            best_score = score
            max_deg = max(e for e, c in pairs)
            dense = [0] * (max_deg + 1)
            for e, c in pairs:
                dense[e] = c
            if len(dense) <= 41 and dense[-1] != 0:
                best_candidate = dense

    def optimize_signs(exps):
        """
        For a fixed set of exponents, brute force all 2^12 sign combinations
        to find the one that minimizes the number of non-zero terms in the square.
        Returns the best score and the best pairs.
        """
        best_local_score = float('inf')
        best_local_pairs = None
        
        # Pre-compute the convolution structure?
        # Actually, just computing the square for 4096 cases is fast enough if done smartly.
        # But let's use a faster check.
        
        # We can represent the polynomial as a vector and use numpy.
        max_deg = max(exps)
        p_base = np.zeros(max_deg + 1, dtype=np.int64)
        for i, e in enumerate(exps):
            p_base[e] = 1 # Placeholder, we will flip signs
        
        # Iterate over all sign masks
        # 2^12 = 4096
        for mask in range(4096):
            p = p_base.copy()
            for i in range(12):
                if (mask >> i) & 1:
                    p[exps[i]] = -1
                else:
                    p[exps[i]] = 1
            
            square = np.convolve(p, p)
            score = int(np.count_nonzero(square))
            
            if score < best_local_score:
                best_local_score = score
                best_local_pairs = [(exps[i], -1 if (mask >> i) & 1 else 1) for i in range(12)]
                if best_local_score <= 12: # Can't be less than 12? 
                    # Actually, minimum is 12? 
                    # If P has 12 terms, P^2 has at least 12 terms? 
                    # No, if P = 1+x, P^2 = 1+2x+x^2 (3 terms). 
                    # If P = 1+x+x^2, P^2 = 1+2x+3x^2+2x^3+x^4 (5 terms).
                    # Generally, number of terms in P^2 >= number of terms in P?
                    # Yes, the diagonal terms (2i) always appear unless coefficient is 0, which it isn't.
                    # So min score is 12.
                    pass

        return best_local_score, best_local_pairs

    def optimize_exponents_local(pairs, max_iters=5000):
        """
        Local search on exponents, keeping signs optimized at each step? 
        Or fixing signs and moving exponents?
        Let's fix signs first, then move exponents, then re-optimize signs.
        """
        current_pairs = pairs[:]
        # First, optimize signs for the current exponents
        exps = [e for e, c in current_pairs]
        _, current_pairs = optimize_signs(exps)
        current_score = compute_square_nonzeros_fast(current_pairs)
        update_best(current_pairs, current_score)
        
        temp = 1.0
        cooling = 0.999
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
                
            new_pairs = current_pairs[:]
            idx = random.randint(0, 11)
            old_exp, old_coeff = new_pairs[idx]
            
            # Move exponent
            # Try small moves first
            if random.random() < 0.5:
                delta = random.choice([-2, -1, 1, 2])
            else:
                delta = random.randint(-5, 5)
            
            new_exp = old_exp + delta
            if new_exp < 0 or new_exp > 40:
                continue
            
            # Check collision
            other_exps = [e for j, (e, c) in enumerate(new_pairs) if j != idx]
            if new_exp in other_exps:
                continue
            
            new_pairs[idx] = (new_exp, old_coeff)
            
            # Re-optimize signs for this new configuration
            new_exps = [e for e, c in new_pairs]
            new_score, new_pairs_optimized = optimize_signs(new_exps)
            
            # Acceptance
            if new_score < current_score:
                current_pairs = new_pairs_optimized
                current_score = new_score
                update_best(current_pairs, current_score)
            else:
                delta_e = new_score - current_score
                if random.random() < np.exp(-delta_e / temp):
                    current_pairs = new_pairs_optimized
                    current_score = new_score
            
            temp *= cooling

    # --- Generate Initial Candidates ---
    
    candidates = []
    
    # 1. Arithmetic Progressions (very structured, often good for cancellation)
    for start in range(0, 10):
        for step in range(1, 8):
            exps = [start + i * step for i in range(12)]
            if max(exps) <= 40:
                candidates.append(exps)
                
    # 2. Geometric Progressions
    for base in range(2, 5):
        exps = [base**i for i in range(12)]
        if max(exps) <= 40:
            candidates.append(exps)
            
    # 3. Random sets with spacing
    for _ in range(100):
        if time.time() > deadline - 20: break
        # Generate 12 distinct exponents
        # Try to space them out to reduce collisions
        exps = sorted(random.sample(range(0, 41), 12))
        candidates.append(exps)
        
    # 4. Structured "Product" patterns
    # A(x) * B(x) where A has 3 terms, B has 4 terms
    for _ in range(50):
        if time.time() > deadline - 20: break
        a_exps = sorted(random.sample(range(0, 15), 3))
        b_exps = sorted(random.sample(range(0, 15), 4))
        prod_exps = set()
        for a in a_exps:
            for b in b_exps:
                prod_exps.add(a + b)
        if len(prod_exps) == 12:
            candidates.append(sorted(prod_exps))
        elif len(prod_exps) > 12:
            # Sample 12
            candidates.append(sorted(random.sample(list(prod_exps), 12)))
        elif len(prod_exps) < 12:
            # Pad with random
            remaining = [x for x in range(41) if x not in prod_exps]
            if len(remaining) >= 12 - len(prod_exps):
                candidates.append(sorted(list(prod_exps) + random.sample(remaining, 12 - len(prod_exps))))

    # Deduplicate
    seen = set()
    unique_candidates = []
    for c in candidates:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)

    # Run optimization on each candidate
    for exps in unique_candidates:
        if time.time() > deadline:
            break
        # First, get the best signs for these exponents
        score, pairs = optimize_signs(exps)
        update_best(pairs, score)
        
        # Then, do local search on exponents
        optimize_exponents_local(pairs)

    # If no valid candidate found, return a fallback
    if best_candidate is None:
        # Fallback: 1 + x + ... + x^11
        best_candidate = [1] * 12

    return best_candidate

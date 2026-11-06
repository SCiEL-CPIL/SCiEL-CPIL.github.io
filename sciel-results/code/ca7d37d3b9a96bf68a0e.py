import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    # Fallback
    fallback = [1] * 12
    best_score = 21
    best_candidate = fallback

    def count_nonzero_square(exps, coeffs):
        """Count nonzero terms in P(x)^2 for sparse polynomial."""
        max_deg = max(exps)
        if max_deg > 40:
            return float('inf')
        if any(abs(c) > 64 for c in coeffs):
            return float('inf')
        if any(c == 0 for c in coeffs):
            return float('inf')
        if len(set(exps)) != len(exps):
            return float('inf')
        
        p = np.zeros(max_deg + 1, dtype=np.int64)
        for e, c in zip(exps, coeffs):
            p[e] = c
        q = np.convolve(p, p)
        return int(np.count_nonzero(q))

    def make_dense(exps, coeffs):
        max_deg = max(exps)
        dense = [0] * (max_deg + 1)
        for e, c in zip(exps, coeffs):
            dense[e] = c
        return dense

    def update_best(exps, coeffs, score):
        nonlocal best_score, best_candidate
        if score < best_score:
            dense = make_dense(exps, coeffs)
            if len(dense) <= 41 and dense[-1] != 0:
                best_score = score
                best_candidate = dense

    def exhaustive_signs(exps, mags):
        """Try all 2^12 sign patterns for given exponents and magnitudes."""
        best_local = float('inf')
        best_coeffs = None
        
        # Generate all sign patterns
        for mask in range(4096):  # 2^12
            if time.time() > deadline - 2:
                break
            coeffs = []
            valid = True
            for j in range(12):
                sign = 1 if (mask >> j) & 1 else -1
                c = sign * mags[j]
                if abs(c) > 64 or c == 0:
                    valid = False
                    break
                coeffs.append(c)
            if not valid:
                continue
            
            score = count_nonzero_square(exps, coeffs)
            if score < best_local:
                best_local = score
                best_coeffs = coeffs[:]
        
        if best_coeffs is not None:
            return best_local, exps[:], best_coeffs
        return None

    def local_search(exps, coeffs):
        """Local search: try flipping signs, changing magnitudes, moving exponents."""
        current_score = count_nonzero_square(exps, coeffs)
        improved = True
        iterations = 0
        
        while improved and iterations < 200:
            improved = False
            iterations += 1
            if time.time() > deadline - 3:
                break
            
            # Try flipping each sign
            for i in range(12):
                new_coeffs = coeffs[:]
                new_coeffs[i] = -new_coeffs[i]
                if abs(new_coeffs[i]) <= 64 and new_coeffs[i] != 0:
                    s = count_nonzero_square(exps, new_coeffs)
                    if s < current_score:
                        coeffs = new_coeffs
                        current_score = s
                        improved = True
            
            # Try changing magnitudes
            for i in range(12):
                for mag in [1, 2, 3, 4, 5, 8, 16, 32, 64]:
                    new_coeffs = coeffs[:]
                    sign = 1 if coeffs[i] > 0 else -1
                    new_coeffs[i] = sign * mag
                    if abs(new_coeffs[i]) <= 64 and new_coeffs[i] != 0:
                        s = count_nonzero_square(exps, new_coeffs)
                        if s < current_score:
                            coeffs = new_coeffs
                            current_score = s
                            improved = True
            
            # Try moving one exponent
            for i in range(12):
                for delta in [-2, -1, 1, 2]:
                    new_exps = exps[:]
                    new_exps[i] += delta
                    if new_exps[i] < 0 or new_exps[i] > 40:
                        continue
                    if len(set(new_exps)) != 12:
                        continue
                    s = count_nonzero_square(new_exps, coeffs)
                    if s < current_score:
                        exps = new_exps
                        current_score = s
                        improved = True
        
        return exps, coeffs, current_score

    # Phase 1: Try structured exponent sets with exhaustive sign search
    candidate_exps = []
    
    # Arithmetic progressions
    for a in range(0, 8):
        for d in range(1, 5):
            exps = [a + i * d for i in range(12)]
            if max(exps) <= 40 and len(set(exps)) == 12:
                candidate_exps.append(exps)
    
    # Two blocks
    for k in range(2, 11):
        for a in range(0, 8):
            block1 = [a + i for i in range(k)]
            for b in range(a + k, min(a + k + 10, 41)):
                block2 = [b + i for i in range(12 - k)]
                if max(block2) <= 40:
                    exps = sorted(block1 + block2)
                    if len(set(exps)) == 12:
                        candidate_exps.append(exps)
    
    # Three blocks
    for k1 in range(3, 6):
        for k2 in range(3, 6):
            k3 = 12 - k1 - k2
            if k3 < 2:
                continue
            for a in range(0, 5):
                b1 = [a + i for i in range(k1)]
                for b in range(a + k1 + 2, min(a + k1 + 8, 30)):
                    b2 = [b + i for i in range(k2)]
                    for c in range(b + k2 + 2, min(b + k2 + 8, 41)):
                        b3 = [c + i for i in range(k3)]
                        if max(b3) <= 40:
                            exps = sorted(b1 + b2 + b3)
                            if len(set(exps)) == 12:
                                candidate_exps.append(exps)
    
    # Random sets
    random.seed(42)
    for _ in range(100):
        exps = sorted(random.sample(range(0, 41), 12))
        candidate_exps.append(exps)
    
    # Deduplicate
    seen = set()
    unique_exps = []
    for exps in candidate_exps:
        key = tuple(exps)
        if key not in seen:
            seen.add(key)
            unique_exps.append(exps)
    
    # Evaluate with sign search + local refinement
    mags_all_ones = [1] * 12
    
    for exps in unique_exps:
        if time.time() > deadline - 5:
            break
        
        result = exhaustive_signs(exps, mags_all_ones)
        if result is not None:
            score, e, c = result
            # Local search to refine
            e2, c2, score2 = local_search(e, c)
            if score2 < best_score:
                update_best(e2, c2, score2)
                if best_score <= 13:
                    return best_candidate
    
    # Phase 2: More aggressive local search from random starts
    if best_score > 13 and time.time() < deadline - 10:
        random.seed(123)
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            exps = sorted(random.sample(range(0, 41), 12))
            coeffs = [random.choice([-1, 1]) for _ in range(12)]
            e2, c2, score2 = local_search(exps, coeffs)
            if score2 < best_score:
                update_best(e2, c2, score2)
                if best_score <= 13:
                    return best_candidate
    
    # Phase 3: Try with varied magnitudes
    if best_score > 13 and time.time() < deadline - 5:
        mag_patterns = [
            [1]*12,
            [1,1,1,1,1,1,2,2,2,2,2,2],
            [1,2,1,2,1,2,1,2,1,2,1,2],
            [1,1,2,2,1,1,2,2,1,1,2,2],
        ]
        test_exps = [
            list(range(12)),
            [0,1,2,3,4,5,10,11,12,13,14,15],
            [0,2,4,6,8,10,12,14,16,18,20,22],
        ]
        for exps in test_exps:
            if time.time() > deadline - 3:
                break
            for mag in mag_patterns:
                result = exhaustive_signs(exps, mag)
                if result is not None:
                    score, e, c = result
                    e2, c2, score2 = local_search(e, c)
                    if score2 < best_score:
                        update_best(e2, c2, score2)
                        if best_score <= 13:
                            return best_candidate
    
    return best_candidate

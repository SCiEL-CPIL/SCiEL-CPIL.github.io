import time
import random
import math
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    best_score = 21
    best_candidate = [1] * 12  # fallback: (1+x+...+x^11)^2 has 21 nonzero terms

    def compute_square_nonzeros(exps, coeffs):
        """Compute number of nonzero terms in P(x)^2."""
        square = [0] * 81
        n = len(exps)
        for i in range(n):
            ei = exps[i]
            ci = coeffs[i]
            square[2 * ei] += ci * ci
            for j in range(i + 1, n):
                ej = exps[j]
                cj = coeffs[j]
                square[ei + ej] += 2 * ci * cj
        return sum(1 for val in square if val != 0)

    def get_dense(exps, coeffs):
        max_deg = max(exps)
        dense = [0] * (max_deg + 1)
        for e, c in zip(exps, coeffs):
            dense[e] = c
        if dense[-1] != 0:
            return dense
        return None

    def try_update(exps, coeffs):
        nonlocal best_score, best_candidate
        if len(exps) != 12 or len(coeffs) != 12:
            return
        if len(set(exps)) != 12:
            return
        if max(exps) > 40 or min(exps) < 0:
            return
        if any(abs(c) > 64 or c == 0 for c in coeffs):
            return
        
        score = compute_square_nonzeros(exps, coeffs)
        if score < best_score:
            best_score = score
            cand = get_dense(exps, coeffs)
            if cand:
                best_candidate = cand

    def optimize_signs_for_exps(exps, time_limit):
        """For a fixed set of exponents, optimize signs to minimize nonzero terms in square."""
        # For each pair (i,j) with i<j, the cross term 2*c_i*c_j appears at degree e_i+e_j
        # If multiple pairs map to the same degree, their contributions must cancel
        # With c_i in {-1, 1}, cross terms are ±2
        # We need to assign signs to maximize cancellations
        
        # Build a graph: nodes are indices 0..11, edges are pairs that share a sum
        # Actually, let's think of it as: for each degree d in the square, 
        # the coefficient is sum of c_i*c_j over all pairs (i,j) with e_i+e_j=d
        # (including i=j which gives c_i^2=1)
        
        # For diagonal terms (i=i): always +1 at degree 2*e_i
        # For off-diagonal: 2*c_i*c_j at degree e_i+e_j
        
        # Strategy: try all 2^12 sign assignments? That's 4096, feasible.
        # But we can do better with local search.
        
        best_local = compute_square_nonzeros(exps, [1]*12)
        best_signs = [1] * 12
        
        # Start with random signs
        random.seed(12345)
        for attempt in range(50):
            if time.time() > time_limit:
                break
            signs = [random.choice([-1, 1]) for _ in range(12)]
            current_score = compute_square_nonzeros(exps, signs)
            if current_score < best_local:
                best_local = current_score
                best_signs = list(signs)
            
            # Local search: flip one sign at a time
            improved = True
            while improved and time.time() < time_limit:
                improved = False
                for i in range(12):
                    signs[i] = -signs[i]
                    new_score = compute_square_nonzeros(exps, signs)
                    if new_score < current_score:
                        current_score = new_score
                        if current_score < best_local:
                            best_local = current_score
                            best_signs = list(signs)
                        improved = True
                    else:
                        signs[i] = -signs[i]  # revert
        
        return best_local, best_signs

    def optimize_coeff_magnitudes(exps, signs, time_limit):
        """Given exponents and signs, try to optimize coefficient magnitudes."""
        # Coefficients are sign_i * magnitude_i where magnitude_i in {1,2,3,...}
        # For cancellation, we need 2*c_i*c_j = -2*c_k*c_l at same degree
        # i.e., sign_i*mag_i*sign_j*mag_j = -sign_k*mag_k*sign_l*mag_l
        
        best_score = compute_square_nonzeros(exps, [s * 1 for s in signs])
        best_coeffs = [s * 1 for s in signs]
        
        # Try small magnitude changes
        for _ in range(200):
            if time.time() > time_limit:
                break
            i = random.randint(0, 11)
            old_mag = abs(best_coeffs[i])
            new_mag = random.choice([1, 2, 3, 4, 5, 6, 8, 10, 12, 16])
            if new_mag == old_mag:
                continue
            if new_mag > 64:
                continue
            new_coeffs = list(best_coeffs)
            new_coeffs[i] = signs[i] * new_mag
            new_score = compute_square_nonzeros(exps, new_coeffs)
            if new_score < best_score:
                best_score = new_score
                best_coeffs = new_coeffs
        
        return best_score, best_coeffs

    def structured_search(exps, time_limit):
        """Try to find good signs and magnitudes for given exponents."""
        score, signs = optimize_signs_for_exps(exps, time_limit)
        if time.time() > time_limit:
            return score, signs
        score, coeffs = optimize_coeff_magnitudes(exps, signs, time_limit)
        return score, coeffs

    # Generate candidate exponent sets with structure that promotes cancellation
    candidate_exps = []
    
    # 1. Pairs that create sum collisions
    # If e_i + e_j = e_k + e_l, we get cancellation opportunities
    # Try sets with many such collisions
    
    # Arithmetic progressions
    for start in range(0, 20):
        for step in range(1, 5):
            exps = [start + i * step for i in range(12)]
            if max(exps) <= 40:
                candidate_exps.append(exps)
    
    # 2. Two interleaved arithmetic progressions
    for start1 in range(0, 10):
        for step1 in range(1, 4):
            for start2 in range(0, 10):
                for step2 in range(1, 4):
                    exps1 = [start1 + i * step1 for i in range(6)]
                    exps2 = [start2 + i * step2 for i in range(6)]
                    if max(exps1) <= 40 and max(exps2) <= 40:
                        exps = sorted(exps1 + exps2)
                        if len(set(exps)) == 12 and max(exps) <= 40:
                            candidate_exps.append(exps)
    
    # 3. Random sets with bias towards creating sum collisions
    random.seed(42)
    for _ in range(200):
        # Start with a random set and try to adjust to create collisions
        exps = sorted(random.sample(range(0, 41), 12))
        # Try to modify to create more sum collisions
        for _ in range(10):
            i = random.randint(0, 11)
            j = random.randint(0, 11)
            if i == j:
                continue
            # Try to set exps[i] + exps[j] equal to some other pair's sum
            target = exps[random.randint(0, 11)] + exps[random.randint(0, 11)]
            new_val = target - exps[j]
            if 0 <= new_val <= 40 and new_val not in exps:
                exps[i] = new_val
                exps = sorted(exps)
        candidate_exps.append(exps)
    
    # 4. Symmetric sets (e_i + e_{11-i} = constant)
    for _ in range(100):
        left = sorted(random.sample(range(0, 21), 6))
        right = sorted(40 - x for x in left)
        exps = sorted(left + right)
        if len(set(exps)) == 12 and max(exps) <= 40:
            candidate_exps.append(exps)
    
    # 5. Sets with many duplicate sums
    # Generate by picking 6 pairs with equal sums
    for _ in range(100):
        random.seed(random.randint(0, 10000))
        # Pick a target sum
        target = random.randint(10, 35)
        pairs = []
        used = set()
        for _ in range(6):
            a = random.randint(0, target)
            b = target - a
            if a < b and a not in used and b not in used:
                pairs.append((a, b))
                used.add(a)
                used.add(b)
        if len(pairs) == 6:
            exps = sorted([x for pair in pairs for x in pair])
            if len(set(exps)) == 12 and max(exps) <= 40:
                candidate_exps.append(exps)
    
    # Remove duplicates
    seen = set()
    unique_candidates = []
    for exps in candidate_exps:
        key = tuple(exps)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(exps)
    
    # Sort by number of sum collisions (more collisions = more cancellation potential)
    def count_collisions(exps):
        sums = {}
        for i in range(12):
            for j in range(i+1, 12):
                s = exps[i] + exps[j]
                sums[s] = sums.get(s, 0) + 1
        return sum(1 for v in sums.values() if v > 1)
    
    unique_candidates.sort(key=count_collisions, reverse=True)
    
    # Process top candidates
    processed = 0
    for exps in unique_candidates:
        if time.time() > deadline - 5:
            break
        if processed >= 100:  # Limit number of candidates to process
            break
        
        time_per_candidate = min(2.0, (deadline - 5 - time.time()) / max(1, 100 - processed))
        score, coeffs = structured_search(exps, time.time() + time_per_candidate)
        
        if score < best_score:
            try_update(exps, coeffs)
        
        processed += 1
    
    # Final polish: local search on best candidate
    if best_score < 21:
        best_exps = [i for i, c in enumerate(best_candidate) if c != 0]
        best_coeffs = [best_candidate[i] for i in best_exps]
        
        polish_deadline = time.time() + min(10.0, deadline - time.time() - 2)
        
        # Try flipping signs
        for _ in range(1000):
            if time.time() > polish_deadline:
                break
            i = random.randint(0, 11)
            best_coeffs[i] = -best_coeffs[i]
            s = compute_square_nonzeros(best_exps, best_coeffs)
            if s < best_score:
                best_score = s
            else:
                best_coeffs[i] = -best_coeffs[i]
        
        # Try changing magnitudes
        for _ in range(1000):
            if time.time() > polish_deadline:
                break
            i = random.randint(0, 11)
            old_c = best_coeffs[i]
            sign = 1 if old_c > 0 else -1
            mag = abs(old_c)
            new_mag = random.choice([1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64])
            if new_mag == mag:
                continue
            new_c = sign * new_mag
            if abs(new_c) > 64:
                continue
            best_coeffs[i] = new_c
            s = compute_square_nonzeros(best_exps, best_coeffs)
            if s < best_score:
                best_score = s
            else:
                best_coeffs[i] = old_c
        
        # Try moving exponents
        for _ in range(500):
            if time.time() > polish_deadline:
                break
            i = random.randint(0, 11)
            old_e = best_exps[i]
            new_e = old_e + random.choice([-2, -1, 1, 2])
            if new_e < 0 or new_e > 40:
                continue
            if new_e in best_exps:
                if best_exps.index(new_e) != i:
                    continue
            best_exps[i] = new_e
            paired = sorted(zip(best_exps, best_coeffs))
            best_exps = [p[0] for p in paired]
            best_coeffs = [p[1] for p in paired]
            s = compute_square_nonzeros(best_exps, best_coeffs)
            if s < best_score:
                best_score = s
            else:
                # Revert
                best_exps[i] = old_e
                paired = sorted(zip(best_exps, best_coeffs))
                best_exps = [p[0] for p in paired]
                best_coeffs = [p[1] for p in paired]
        
        if best_score < 21:
            cand = get_dense(best_exps, best_coeffs)
            if cand:
                best_candidate = cand
    
    return best_candidate

# EVOLVE-BLOCK-END

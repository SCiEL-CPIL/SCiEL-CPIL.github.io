import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin

    best_score = 21
    best_candidate = [1] * 12  # fallback

    def square_nonzeros(exps, coeffs):
        """Compute number of nonzero terms in P(x)^2."""
        n = len(exps)
        # Use dict for sparse representation
        sq = {}
        for i in range(n):
            ei = exps[i]
            ci = coeffs[i]
            # Self term
            key = 2 * ei
            sq[key] = sq.get(key, 0) + ci * ci
            # Cross terms
            for j in range(i + 1, n):
                ej = exps[j]
                cj = coeffs[j]
                key = ei + ej
                sq[key] = sq.get(key, 0) + 2 * ci * cj
        return sum(1 for v in sq.values() if v != 0)

    def square_nonzeros_arr(exps, coeffs):
        """Faster: use array if degree is small enough."""
        max_deg = 2 * max(exps)
        if max_deg > 100:
            return square_nonzeros(exps, coeffs)
        sq = [0] * (max_deg + 1)
        n = len(exps)
        for i in range(n):
            ei = exps[i]
            ci = coeffs[i]
            sq[2 * ei] += ci * ci
            for j in range(i + 1, n):
                ej = exps[j]
                cj = coeffs[j]
                sq[ei + ej] += 2 * ci * cj
        return sum(1 for v in sq if v != 0)

    def try_candidate(exps, coeffs):
        nonlocal best_score, best_candidate
        if len(exps) != 12 or len(coeffs) != 12:
            return
        if len(set(exps)) != 12:
            return
        if max(exps) > 40 or min(exps) < 0:
            return
        if any(abs(c) > 64 or c == 0 for c in coeffs):
            return
        score = square_nonzeros_arr(exps, coeffs)
        if score < best_score:
            best_score = score
            max_deg = max(exps)
            dense = [0] * (max_deg + 1)
            for e, c in zip(exps, coeffs):
                dense[e] = c
            if dense[-1] != 0:
                best_candidate = dense

    def sign_optimize(exps, max_iters=2000):
        """Given fixed exponents, find best signs using simulated annealing."""
        n = 12
        best_signs = [1] * n
        best_local = square_nonzeros_arr(exps, best_signs)
        
        # Simulated annealing
        current = [1] * n
        current_score = best_local
        temp = 5.0
        cooling = 0.999
        
        for it in range(max_iters):
            if time.time() > deadline - 2:
                break
            # Pick random position to flip
            i = random.randint(0, n - 1)
            old_sign = current[i]
            current[i] = -old_sign
            new_score = square_nonzeros_arr(exps, current)
            
            if new_score < current_score:
                current_score = new_score
                if new_score < best_local:
                    best_local = new_score
                    best_signs = current[:]
            else:
                # Accept with probability exp(-delta/temp)
                import math
                delta = new_score - current_score
                if random.random() < math.exp(-delta / temp):
                    current_score = new_score
                else:
                    current[i] = old_sign
            temp *= cooling
        
        return best_signs, best_local

    def small_coeff_optimize(exps, max_iters=5000):
        """Optimize small integer coefficients {-2,-1,1,2} using SA."""
        n = 12
        vals = [-2, -1, 1, 2]
        best_coeffs = [1] * n
        best_local = square_nonzeros_arr(exps, best_coeffs)
        
        current = [1] * n
        current_score = best_local
        temp = 3.0
        cooling = 0.9995
        
        for it in range(max_iters):
            if time.time() > deadline - 2:
                break
            i = random.randint(0, n - 1)
            old_val = current[i]
            new_val = random.choice(vals)
            if new_val == old_val:
                continue
            current[i] = new_val
            new_score = square_nonzeros_arr(exps, current)
            
            if new_score < current_score:
                current_score = new_score
                if new_score < best_local:
                    best_local = new_score
                    best_coeffs = current[:]
            else:
                import math
                delta = new_score - current_score
                if random.random() < math.exp(-delta / temp):
                    current_score = new_score
                else:
                    current[i] = old_val
            temp *= cooling
        
        return best_coeffs, best_local

    # === Phase 1: Structured exponent generation ===
    random.seed(42)
    
    candidate_exps = set()
    
    # 1. Arithmetic progressions
    for start in range(0, 13):
        for step in range(1, 5):
            exps = tuple(start + i * step for i in range(12))
            if max(exps) <= 40:
                candidate_exps.add(exps)
    
    # 2. Two interleaved APs (6+6)
    for s1 in range(0, 8):
        for d1 in range(1, 4):
            for s2 in range(5, 22):
                for d2 in range(1, 4):
                    exps1 = [s1 + i * d1 for i in range(6)]
                    exps2 = [s2 + i * d2 for i in range(6)]
                    all_e = sorted(exps1 + exps2)
                    if len(set(all_e)) == 12 and max(all_e) <= 40 and min(all_e) >= 0:
                        candidate_exps.add(tuple(all_e))
    
    # 3. Symmetric patterns
    for D in range(20, 41):
        for d1 in range(1, 4):
            for d2 in range(1, 4):
                left = [d1 * i for i in range(6)]
                right = [D - d2 * i for i in range(6)]
                all_e = sorted(left + right)
                if len(set(all_e)) == 12 and max(all_e) <= 40 and min(all_e) >= 0:
                    candidate_exps.add(tuple(all_e))
    
    # 4. Random exponent sets
    for _ in range(2000):
        exps = tuple(sorted(random.sample(range(0, 41), 12)))
        candidate_exps.add(exps)
    
    # 5. "Clustered" exponents: groups close together
    for _ in range(1000):
        # Generate 12 exponents with some clustering
        base = random.randint(0, 10)
        exps = []
        for i in range(12):
            if i < 6:
                e = base + random.randint(0, 5)
            else:
                e = base + random.randint(8, 15)
            exps.append(e)
        exps = sorted(set(exps))
        while len(exps) < 12:
            e = random.randint(0, 40)
            if e not in exps:
                exps.append(e)
        exps = sorted(exps[:12])
        if max(exps) <= 40:
            candidate_exps.add(tuple(exps))
    
    # Sort by span (smaller span = more potential for collisions)
    exps_list = sorted(candidate_exps, key=lambda e: max(e) - min(e))
    
    # === Phase 2: For each exponent set, optimize coefficients ===
    processed = 0
    for exps in exps_list:
        if time.time() > deadline - 5:
            break
        exps = list(exps)
        
        # Try all-ones
        try_candidate(exps, [1] * 12)
        # Try alternating signs
        try_candidate(exps, [1 if i % 2 == 0 else -1 for i in range(12)])
        
        # Sign optimization
        signs, s_score = sign_optimize(exps, max_iters=1500)
        try_candidate(exps, signs)
        
        # Small coefficient optimization (only for promising spans)
        span = max(exps) - min(exps)
        if span < 25:
            coeffs, c_score = small_coeff_optimize(exps, max_iters=3000)
            try_candidate(exps, coeffs)
        
        processed += 1
    
    # === Phase 3: Local search on best candidates found ===
    if time.time() < deadline - 10:
        # Re-examine best with more thorough local search
        if best_score < 21:
            # Find the exponents of best candidate
            nz_indices = [i for i, c in enumerate(best_candidate) if c != 0]
            if len(nz_indices) == 12:
                exps = nz_indices
                coeffs = [best_candidate[i] for i in nz_indices]
                # More thorough SA
                signs, _ = sign_optimize(exps, max_iters=5000)
                try_candidate(exps, signs)
                if time.time() < deadline - 5:
                    coeffs2, _ = small_coeff_optimize(exps, max_iters=10000)
                    try_candidate(exps, coeffs2)
    
    return best_candidate

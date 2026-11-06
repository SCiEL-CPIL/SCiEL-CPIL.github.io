import time
import random
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    # Fallback: Simple polynomial, score is likely 21
    # (1 + x + ... + x^11)^2 has 21 non-zero terms
    fallback = [1] * 12
    fallback_score = 21

    best_score = fallback_score
    best_candidate = fallback

    def compute_square_nonzeros_fast(exps, coeffs):
        """
        Fast computation of non-zero terms in P(x)^2 using NumPy.
        """
        n = len(exps)
        exps_arr = np.array(exps, dtype=np.int64)
        coeffs_arr = np.array(coeffs, dtype=np.int64)
        
        # Build sparse representation
        max_deg = int(np.max(exps_arr) * 2)
        square = np.zeros(max_deg + 1, dtype=np.int64)
        
        for i in range(n):
            ei = exps_arr[i]
            ci = coeffs_arr[i]
            square[2 * ei] += ci * ci
            for j in range(i + 1, n):
                ej = exps_arr[j]
                cj = coeffs_arr[j]
                square[ei + ej] += 2 * ci * cj
        
        return int(np.count_nonzero(square))

    def try_candidate(exps, coeffs):
        """
        Check if candidate is valid and update best if better.
        """
        nonlocal best_score, best_candidate
        
        # Validation
        if len(exps) != 12 or len(coeffs) != 12:
            return
        if len(set(exps)) != 12:
            return
        if max(exps) > 40:
            return
        if any(abs(c) > 64 for c in coeffs):
            return
        if any(c == 0 for c in coeffs):
            return
        
        score = compute_square_nonzeros_fast(exps, coeffs)
        
        if score < best_score:
            best_score = score
            max_deg = max(exps)
            dense = [0] * (max_deg + 1)
            for e, c in zip(exps, coeffs):
                dense[e] = c
            if dense[-1] != 0:
                best_candidate = dense

    def optimize_signs(exps, max_iter=200):
        """
        Local search for best sign assignment with multi-flip moves.
        """
        if time.time() > deadline:
            return
        
        random.seed(42)
        
        best_local_score = float('inf')
        best_local_signs = [1] * 12

        for restart in range(8):
            if time.time() > deadline:
                break
            
            if restart == 0:
                signs = [1] * 12
            else:
                signs = [1 if random.random() > 0.5 else -1 for _ in range(12)]
            
            current_score = compute_square_nonzeros_fast(exps, signs)
            
            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                
                improved = False
                
                # Single flip moves
                for i in range(12):
                    old = signs[i]
                    signs[i] = -signs[i]
                    new_score = compute_square_nonzeros_fast(exps, signs)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                    else:
                        signs[i] = old
                
                # Double flip moves (escape local optima)
                if not improved and restart > 0:
                    for i in range(12):
                        for j in range(i + 1, 12):
                            old_i, old_j = signs[i], signs[j]
                            signs[i] = -signs[i]
                            signs[j] = -signs[j]
                            new_score = compute_square_nonzeros_fast(exps, signs)
                            if new_score < current_score:
                                current_score = new_score
                                improved = True
                            else:
                                signs[i], signs[j] = old_i, old_j
                            if improved:
                                break
                        if improved:
                            break
                
                if not improved:
                    break
            
            if current_score < best_local_score:
                best_local_score = current_score
                best_local_signs = signs[:]
        
        if best_local_score < best_score:
            try_candidate(exps, best_local_signs)

    def optimize_coeffs(exps, max_iter=100):
        """
        Local search for small integer coefficients with structured moves.
        """
        if time.time() > deadline:
            return
        
        random.seed(99)
        
        best_local_score = float('inf')
        best_local_coeffs = [1] * 12
        
        for restart in range(5):
            if time.time() > deadline:
                break
            
            if restart == 0:
                coeffs = [1 if i % 2 == 0 else -1 for i in range(12)]
            else:
                coeffs = [random.choice([-3, -2, -1, 1, 2, 3]) for _ in range(12)]
            
            current_score = compute_square_nonzeros_fast(exps, coeffs)
            
            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                
                i = random.randint(0, 11)
                old_val = coeffs[i]
                new_val = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
                if new_val == old_val:
                    continue
                
                coeffs[i] = new_val
                new_score = compute_square_nonzeros_fast(exps, coeffs)
                
                if new_score < current_score:
                    current_score = new_score
                else:
                    coeffs[i] = old_val
            
            if current_score < best_local_score:
                best_local_score = current_score
                best_local_coeffs = coeffs[:]
        
        if best_local_score < best_score:
            try_candidate(exps, best_local_coeffs)

    def try_AB_decomposition():
        """
        Model P = A + B where A and B are 6-term polynomials with disjoint
        exponent sets. Enforce structural constraints to minimize collisions.
        """
        if time.time() > deadline:
            return
        
        # Generate A and B exponent sets that are well-separated
        for a_start in range(0, 6):
            for a_step in range(1, 4):
                a_exps = [a_start + i * a_step for i in range(6)]
                if max(a_exps) > 40:
                    continue
                
                for b_start in range(0, 41):
                    for b_step in range(1, 4):
                        b_exps = [b_start + i * b_step for i in range(6)]
                        if max(b_exps) > 40:
                            continue
                        
                        all_exps = sorted(a_exps + b_exps)
                        if len(set(all_exps)) != 12:
                            continue
                        
                        # Check separation: min gap between A and B exponents
                        a_set = set(a_exps)
                        b_set = set(b_exps)
                        if a_set & b_set:
                            continue
                        
                        # Try various coefficient patterns
                        for a_sign_pattern in [[1,1,1,1,1,1], [1,-1,1,-1,1,-1], [1,1,-1,-1,1,1], [1,-1,-1,1,1,-1]]:
                            for b_sign_pattern in [[1,1,1,1,1,1], [1,-1,1,-1,1,-1], [1,1,-1,-1,1,1], [1,-1,-1,1,1,-1]]:
                                if time.time() > deadline:
                                    return
                                
                                coeffs = a_sign_pattern + b_sign_pattern
                                try_candidate(all_exps, coeffs)
                                
                                # Also try scaled versions
                                for scale_a in [1, 2]:
                                    for scale_b in [1, 2]:
                                        if scale_a * scale_b > 64:
                                            continue
                                        scaled_coeffs = [c * scale_a for c in a_sign_pattern] + [c * scale_b for c in b_sign_pattern]
                                        if any(abs(c) > 64 for c in scaled_coeffs):
                                            continue
                                        try_candidate(all_exps, scaled_coeffs)
                        
                        # Symmetric A/B patterns: A exponents low, B exponents high
                        # This naturally reduces cross-term collisions
                        if min(b_exps) - max(a_exps) >= 5:
                            # Good separation, try more coefficient variations
                            for a_coeff in [[1,1,1,1,1,1], [1,-1,1,-1,1,-1], [2,1,1,1,1,2], [1,2,1,1,2,1]]:
                                for b_coeff in [[1,1,1,1,1,1], [1,-1,1,-1,1,-1], [2,1,1,1,1,2], [1,2,1,1,2,1]]:
                                    if time.time() > deadline:
                                        return
                                    coeffs = a_coeff + b_coeff
                                    if any(abs(c) > 64 for c in coeffs):
                                        continue
                                    try_candidate(all_exps, coeffs)

    def try_geometric_patterns():
        """
        Try geometric and other structured exponent patterns.
        """
        if time.time() > deadline:
            return
        
        # Geometric sequences
        for base in [2, 3]:
            exps = [base ** i for i in range(12)]
            if max(exps) <= 40:
                optimize_signs(exps)
        
        # Quadratic: e_i = i^2
        exps = [i * i for i in range(12)]
        if max(exps) <= 40:
            optimize_signs(exps)
        
        # Mixed: first 6 linear, next 6 quadratic
        for start in range(0, 5):
            exps1 = [start + i for i in range(6)]
            exps2 = [start + 10 + i * i for i in range(6)]
            all_exps = sorted(exps1 + exps2)
            if len(set(all_exps)) == 12 and max(all_exps) <= 40:
                optimize_signs(all_exps)
        
        # Symmetric around center: e_i = center ± offsets
        for center in range(10, 25):
            for offsets in [[1,2,3,4,5,6], [1,2,4,5,7,8], [1,3,4,6,7,9], [2,3,5,6,8,10]]:
                exps = [center - o for o in offsets] + [center + o for o in offsets]
                if len(set(exps)) == 12 and min(exps) >= 0 and max(exps) <= 40:
                    optimize_signs(sorted(exps))

    def try_interleaved_patterns():
        """
        Try interleaved patterns with controlled collision structure.
        """
        if time.time() > deadline:
            return
        
        # Two interleaved APs with controlled spacing
        for d1 in [1, 2, 3]:
            for d2 in [1, 2, 3]:
                for offset in range(0, 5):
                    exps1 = [i * d1 for i in range(6)]
                    exps2 = [offset + i * d2 for i in range(6)]
                    all_exps = sorted(exps1 + exps2)
                    if len(set(all_exps)) == 12 and max(all_exps) <= 40:
                        optimize_signs(all_exps)
        
        # Three interleaved groups of 4
        for d in [1, 2]:
            for base in range(0, 4):
                exps = []
                for g in range(3):
                    start = base + g * 4 * d
                    for i in range(4):
                        exps.append(start + i * d)
                if len(set(exps)) == 12 and max(exps) <= 40:
                    optimize_signs(sorted(exps))

    # --- Main search strategy ---
    
    # Phase 1: Quick structured searches
    try_AB_decomposition()
    
    if time.time() > deadline:
        return best_candidate
    
    # Phase 2: Geometric and symmetric patterns
    try_geometric_patterns()
    
    if time.time() > deadline:
        return best_candidate
    
    # Phase 3: Interleaved patterns
    try_interleaved_patterns()
    
    if time.time() > deadline:
        return best_candidate
    
    # Phase 4: Systematic AP search with optimization
    seen = set()
    for start in range(0, 10):
        for step in range(1, 5):
            exps = [start + i * step for i in range(12)]
            if max(exps) > 40:
                continue
            key = tuple(exps)
            if key in seen:
                continue
            seen.add(key)
            
            if time.time() > deadline:
                return best_candidate
            
            optimize_signs(exps)
            
            if best_score <= 16 and time.time() < deadline - 15:
                optimize_coeffs(exps)
            
            if best_score <= 13:
                break
    
    # Phase 5: Final polish with more iterations on best candidates
    if time.time() < deadline - 5 and best_score > 13:
        # Re-optimize the best structure found
        seen2 = set()
        for start in range(0, 8):
            for step in range(1, 4):
                exps = [start + i * step for i in range(12)]
                if max(exps) > 40:
                    continue
                key = tuple(exps)
                if key in seen2:
                    continue
                seen2.add(key)
                
                if time.time() > deadline:
                    return best_candidate
                
                # More aggressive optimization
                random.seed(123)
                best_local = float('inf')
                best_local_coeffs = [1] * 12
                
                for _ in range(10):
                    coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
                    score = compute_square_nonzeros_fast(exps, coeffs)
                    if score < best_local:
                        best_local = score
                        best_local_coeffs = coeffs[:]
                
                if best_local < best_score:
                    try_candidate(exps, best_local_coeffs)
    
    return best_candidate

# EVOLVE-BLOCK-END

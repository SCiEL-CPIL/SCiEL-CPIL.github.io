import random
import itertools
import numpy as np

def construct():
    best_count = float('inf')
    best_coeffs = [1]*12
    
    def count_nonzero(coeffs):
        n = len(coeffs)
        square = [0]*(2*n-1)
        for i in range(n):
            for j in range(n):
                square[i+j] += coeffs[i]*coeffs[j]
        return sum(1 for x in square if x != 0)

    def evaluate_p(b_coeffs, e_b, k, sign):
        # Construct P
        # B has terms at e_b with coeffs b_coeffs
        # P = sign * x^k B + B
        # Exponents in P: e_b and [x+k for x in e_b]
        # Check if distinct
        exps = sorted(e_b + [x+k for x in e_b])
        if len(exps) != len(set(exps)):
            return None, None
        
        # Map exponents to coeffs
        p_coeffs_map = {}
        for i, e in enumerate(e_b):
            p_coeffs_map[e] = p_coeffs_map.get(e, 0) + b_coeffs[i]
            p_coeffs_map[e+k] = p_coeffs_map.get(e+k, 0) + sign * b_coeffs[i]
        
        # Convert to list
        max_exp = max(exps)
        p_list = [0]*(max_exp+1)
        for e, c in p_coeffs_map.items():
            p_list[e] = c
            
        # Check constraints: exactly 12 nonzeros, coeffs in [-64, 64]
        non_zeros = [c for c in p_list if c != 0]
        if len(non_zeros) != 12:
            return None, None
        if any(abs(c) > 64 for c in non_zeros):
            return None, None
            
        cnt = count_nonzero(p_list)
        return cnt, p_list

    # Search over k and B structures
    random.seed(42)
    
    # Try different k values
    for k in range(6, 40):
        # Try different exponent sets for B (6 terms)
        # e_b must be < k to avoid overlap with shifted B if we want distinct?
        # Actually overlap is handled by summing, but we need exactly 12 nonzeros in P.
        # If e_i == e_j + k, they merge. We want 12 distinct non-zeros.
        # So we should avoid e_i in e_b being equal to e_j + k.
        
        # Generate random B exponents
        for trial in range(2000):
            e_b = sorted(random.sample(range(0, k), 6))
            # Ensure e_b[0] is 0? Not strictly necessary, but helps normalization.
            # Let's shift e_b so min is 0
            min_e = min(e_b)
            e_b = [x - min_e for x in e_b]
            
            for sign in [1, -1]:
                # Initialize b_coeffs randomly
                b_coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(6)]
                
                current_cnt, current_p = evaluate_p(b_coeffs, e_b, k, sign)
                if current_p is None:
                    continue
                    
                if current_cnt < best_count:
                    best_count = current_cnt
                    best_coeffs = current_p[:]
                    
                # Local search on b_coeffs
                improved = True
                while improved and current_cnt < best_count:
                    improved = False
                    for i in range(6):
                        for delta in [-2, -1, 1, 2]:
                            new_b = b_coeffs[i] + delta
                            if -64 <= new_b <= 64:
                                old_b = b_coeffs[i]
                                b_coeffs[i] = new_b
                                new_cnt, new_p = evaluate_p(b_coeffs, e_b, k, sign)
                                if new_p is not None:
                                    if new_cnt < current_cnt:
                                        current_cnt = new_cnt
                                        improved = True
                                        if current_cnt < best_count:
                                            best_count = current_cnt
                                            best_coeffs = new_p[:]
                                    else:
                                        b_coeffs[i] = old_b
                                else:
                                    b_coeffs[i] = old_b
            break # Only do a few trials per k? Or all?
            # To save time, limit trials per k
            if trial > 50: break

    # Fallback to original random search if nothing found or to improve
    if best_count > 20: # If our structured search didn't find something good
        # Run original logic briefly
        for trial in range(5000):
            coeffs = [random.choice([-2,-1,1,2]) for _ in range(12)]
            for i in range(12):
                if coeffs[i] == 0:
                    coeffs[i] = 1
            current = count_nonzero(coeffs)
            if current < best_count:
                best_count = current
                best_coeffs = coeffs[:]
            # Simple local search
            improved = True
            while improved and current < best_count:
                improved = False
                for i in range(12):
                    for delta in [-1, 1]:
                        new_c = coeffs[i] + delta
                        if -64 <= new_c <= 64 and new_c != 0:
                            old = coeffs[i]
                            coeffs[i] = new_c
                            new_count = count_nonzero(coeffs)
                            if new_count < current:
                                current = new_count
                                improved = True
                                if current < best_count:
                                    best_count = current
                                    best_coeffs = coeffs[:]
                            else:
                                coeffs[i] = old

    return best_coeffs

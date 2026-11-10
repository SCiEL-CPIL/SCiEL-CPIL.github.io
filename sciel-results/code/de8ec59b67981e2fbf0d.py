import numpy as np
import time
import random
from itertools import product, combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit

    def count_nonzero(poly):
        return np.count_nonzero(poly)

    def poly_square(coeffs):
        return np.convolve(coeffs, coeffs)

    def evaluate(coeffs):
        sq = poly_square(coeffs)
        return count_nonzero(sq)

    def validate(coeffs):
        if len(coeffs) > 41:
            return False
        if len(coeffs) == 0:
            return False
        if coeffs[-1] == 0:
            return False
        if not np.all((coeffs >= -64) & (coeffs <= 64)):
            return False
        if count_nonzero(coeffs) != 12:
            return False
        return True

    best_score = float('inf')
    best_poly = None

    # Strategy: Enumerate factor pairs A*B where |A|*|B| = 12
    # Factorizations: (1,12), (2,6), (3,4), (4,3), (6,2), (12,1)
    # For each, search over support positions and coefficient values

    def search_factor_pair(size_a, size_b, max_coeffs=6, time_budget=30):
        """Search for sparse square using factorization A*B"""
        nonlocal best_score, best_poly
        
        # Generate all possible support sets for A and B
        # A has size_a terms, B has size_b terms
        # Total degree of P = max_deg_A + max_deg_B <= 40
        
        # Try various support configurations
        results = []
        
        # For efficiency, limit the search space
        # A supports: choose size_a positions from 0..max_deg_a
        # B supports: choose size_b positions from 0..max_deg_b
        
        max_deg_a = 15
        max_deg_b = 15
        
        # Generate candidate supports
        a_supports = list(combinations(range(max_deg_a + 1), size_a))
        b_supports = list(combinations(range(max_deg_b + 1), size_b))
        
        # Limit to reasonable number of combinations
        max_a_supports = min(len(a_supports), 200)
        max_b_supports = min(len(b_supports), 200)
        
        a_supports = a_supports[:max_a_supports]
        b_supports = b_supports[:max_b_supports]
        
        coeff_range = list(range(-max_coeffs, max_coeffs + 1))
        # Filter out zero for non-constant terms
        coeff_range_nonzero = [c for c in coeff_range if c != 0]
        
        for a_sup in a_supports:
            if time.time() > deadline - 5:
                break
            for b_sup in b_supports:
                if time.time() > deadline - 5:
                    break
                
                # Build A and B with random coefficients, then optimize
                a_coeffs = np.zeros(max_deg_a + 1, dtype=np.int64)
                b_coeffs = np.zeros(max_deg_b + 1, dtype=np.int64)
                
                for pos in a_sup:
                    a_coeffs[pos] = random.choice(coeff_range_nonzero)
                for pos in b_sup:
                    b_coeffs[pos] = random.choice(coeff_range_nonzero)
                
                # Compute P = A * B
                p = np.convolve(a_coeffs, b_coeffs)
                
                # Check if P has exactly 12 nonzero terms
                if count_nonzero(p) != 12:
                    continue
                
                # Trim trailing zeros
                while len(p) > 0 and p[-1] == 0:
                    p = p[:-1]
                
                if len(p) > 41:
                    continue
                
                # Evaluate the square
                score = evaluate(p)
                
                if score < best_score:
                    best_score = score
                    best_poly = p.copy()
                    results.append((score, p.copy()))
                
                # Local search: perturb coefficients of A and B
                curr_p = p.copy()
                curr_score = score
                no_improve = 0
                max_no_improve = 50
                temperature = 1.5
                
                while no_improve < max_no_improve and time.time() < deadline - 5:
                    # Perturb one coefficient in A or B
                    if random.random() < 0.5:
                        # Perturb A
                        idx = random.choice(a_sup)
                        old_val = a_coeffs[idx]
                        new_val = random.choice(coeff_range_nonzero)
                        if new_val == old_val:
                            continue
                        a_coeffs[idx] = new_val
                        new_p = np.convolve(a_coeffs, b_coeffs)
                    else:
                        # Perturb B
                        idx = random.choice(b_sup)
                        old_val = b_coeffs[idx]
                        new_val = random.choice(coeff_range_nonzero)
                        if new_val == old_val:
                            continue
                        b_coeffs[idx] = new_val
                        new_p = np.convolve(a_coeffs, b_coeffs)
                    
                    # Check if new P has exactly 12 terms
                    if count_nonzero(new_p) != 12:
                        # Revert
                        if random.random() < 0.5:
                            a_coeffs[a_sup[random.randint(0, len(a_sup)-1)]] = old_val if random.random() < 0.5 else b_coeffs[b_sup[random.randint(0, len(b_sup)-1)]]
                        no_improve += 1
                        continue
                    
                    # Trim
                    while len(new_p) > 0 and new_p[-1] == 0:
                        new_p = new_p[:-1]
                    
                    if len(new_p) > 41:
                        no_improve += 1
                        continue
                    
                    new_score = evaluate(new_p)
                    
                    if new_score < curr_score:
                        curr_score = new_score
                        curr_p = new_p.copy()
                        no_improve = 0
                        if curr_score < best_score:
                            best_score = curr_score
                            best_poly = curr_p.copy()
                    else:
                        # Simulated annealing
                        delta = new_score - curr_score
                        if random.random() < np.exp(-delta / max(temperature, 0.1)):
                            curr_score = new_score
                            curr_p = new_p.copy()
                            no_improve = 0
                            if curr_score < best_score:
                                best_score = curr_score
                                best_poly = curr_p.copy()
                        else:
                            # Revert
                            if random.random() < 0.5:
                                a_coeffs[a_sup[random.randint(0, len(a_sup)-1)]] = old_val
                            else:
                                b_coeffs[b_sup[random.randint(0, len(b_sup)-1)]] = old_val
                            no_improve += 1
                    
                    temperature *= 0.999
        
        return results

    # Try different factorizations
    random.seed(42)
    
    # Factorization (3, 4) - good balance
    search_factor_pair(3, 4, max_coeffs=5, time_budget=20)
    
    if time.time() < deadline - 10:
        # Factorization (4, 3)
        search_factor_pair(4, 3, max_coeffs=5, time_budget=20)
    
    if time.time() < deadline - 5:
        # Factorization (2, 6)
        search_factor_pair(2, 6, max_coeffs=4, time_budget=15)
    
    if time.time() < deadline - 5:
        # Factorization (6, 2)
        search_factor_pair(6, 2, max_coeffs=4, time_budget=15)
    
    # Also try direct search on 12-term polynomials with structured patterns
    if time.time() < deadline - 10:
        # Try patterns that are known to have sparse squares
        # e.g., (1+x+x^2+x^3+x^4+x^5)^2 has 11 terms
        # But we need 12 terms in P
        
        # Try: P = (1+x+x^2+x^3+x^4+x^5) * (1+x) 
        # This gives 12 terms
        base1 = np.array([1,1,1,1,1,1], dtype=np.int64)
        base2 = np.array([1,1], dtype=np.int64)
        p = np.convolve(base1, base2)
        if count_nonzero(p) == 12 and len(p) <= 41:
            score = evaluate(p)
            if score < best_score:
                best_score = score
                best_poly = p.copy()
        
        # Try more structured patterns
        patterns = [
            [1,1,1,1,1,1,1,1,1,1,1,-1],
            [1,1,1,1,1,1,1,1,1,1,-1,-1],
            [1,1,1,1,1,1,-1,-1,-1,-1,-1,-1],
            [1,1,1,1,-1,-1,-1,-1,1,1,1,1],
            [1,1,-1,-1,1,1,-1,-1,1,1,-1,-1],
            [1,2,1,1,2,1,1,2,1,1,2,1],
            [1,-1,1,-1,1,-1,1,-1,1,-1,1,-1],
            [1,1,1,-1,-1,-1,1,1,1,-1,-1,-1],
            [1,1,-1,1,-1,1,1,-1,1,-1,1,1],
            [2,1,-1,-2,1,-1,2,-1,1,-2,1,-1],
        ]
        
        for pat in patterns:
            p = np.array(pat, dtype=np.int64)
            if validate(p):
                score = evaluate(p)
                if score < best_score:
                    best_score = score
                    best_poly = p.copy()
        
        # Local search on best found
        if best_poly is not None:
            curr_c = best_poly.copy()
            curr_score = evaluate(curr_c)
            
            no_improve = 0
            max_no_improve = 2000
            temperature = 2.0
            
            while no_improve < max_no_improve and time.time() < deadline - 3:
                idx = random.randint(0, 11)
                new_val = random.randint(-8, 8)
                if new_val == 0:
                    new_val = 1 if random.random() < 0.5 else -1
                
                if curr_c[idx] == new_val:
                    continue
                
                old_val = curr_c[idx]
                curr_c[idx] = new_val
                
                # Check if still valid (12 nonzero terms)
                if count_nonzero(curr_c) != 12:
                    curr_c[idx] = old_val
                    no_improve += 1
                    continue
                
                new_score = evaluate(curr_c)
                
                if new_score < curr_score:
                    curr_score = new_score
                    no_improve = 0
                    if curr_score < best_score:
                        best_score = curr_score
                        best_poly = curr_c.copy()
                        if best_score <= 12:
                            break
                else:
                    delta = new_score - curr_score
                    if random.random() < np.exp(-delta / max(temperature, 0.1)):
                        curr_score = new_score
                        no_improve = 0
                        if curr_score < best_score:
                            best_score = curr_score
                            best_poly = curr_c.copy()
                    else:
                        curr_c[idx] = old_val
                        no_improve += 1
                
                temperature *= 0.9995
    
    # Final polish: try to improve the best solution with more iterations
    if best_poly is not None and time.time() < deadline - 2:
        curr_c = best_poly.copy()
        curr_score = evaluate(curr_c)
        
        # Try modifying 2 coefficients at a time
        no_improve = 0
        max_no_improve = 1000
        
        while no_improve < max_no_improve and time.time() < deadline - 2:
            indices = random.sample(range(12), 2)
            old_vals = [curr_c[i] for i in indices]
            
            for i in indices:
                val = random.randint(-6, 6)
                if val == 0:
                    val = 1 if random.random() < 0.5 else -1
                curr_c[i] = val
            
            if count_nonzero(curr_c) != 12:
                for i, v in zip(indices, old_vals):
                    curr_c[i] = v
                no_improve += 1
                continue
            
            new_score = evaluate(curr_c)
            
            if new_score < curr_score:
                curr_score = new_score
                no_improve = 0
                if curr_score < best_score:
                    best_score = curr_score
                    best_poly = curr_c.copy()
            else:
                for i, v in zip(indices, old_vals):
                    curr_c[i] = v
                no_improve += 1
    
    # Ensure final coefficient is nonzero
    if best_poly is None:
        best_poly = np.array([1]*12, dtype=np.int64)
    
    while len(best_poly) > 0 and best_poly[-1] == 0:
        best_poly = best_poly[:-1]
    
    # Final validation
    if not validate(best_poly):
        # Fallback to a valid polynomial
        best_poly = np.array([1]*12, dtype=np.int64)
    
    return best_poly.tolist()

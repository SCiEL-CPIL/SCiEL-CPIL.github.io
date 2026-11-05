import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s kill

    best_count = float('inf')
    best_coeffs = None

    def eval_square(coeffs_list):
        """
        coeffs_list: list of (exp, coef) with len 12, coef != 0
        Returns: (count_nonzero_in_square, square_list, max_exp)
        """
        if not coeffs_list:
            return 0, [0], 0
        max_e = max(e for e, c in coeffs_list)
        # Max degree in square is 2 * max_e
        square = [0] * (2 * max_e + 1)
        
        # Optimization: Sort by exponent
        cl = sorted(coeffs_list)
        
        for i in range(12):
            ei, ci = cl[i]
            # Square term
            square[2 * ei] += ci * ci
            # Cross terms
            for j in range(i + 1, 12):
                ej, cj = cl[j]
                square[ei + ej] += 2 * ci * cj
        
        count = sum(1 for x in square if x != 0)
        return count, square, max_e

    def eval_square_fast(coeffs_list):
        """
        Fast evaluation using dict for sparse representation.
        coeffs_list: list of (exp, coef) with len 12, coef != 0
        Returns: count_nonzero_in_square
        """
        cl = sorted(coeffs_list)
        square_dict = defaultdict(int)
        
        for i in range(12):
            ei, ci = cl[i]
            square_dict[2 * ei] += ci * ci
            for j in range(i + 1, 12):
                ej, cj = cl[j]
                square_dict[ei + ej] += 2 * ci * cj
        
        return sum(1 for v in square_dict.values() if v != 0)

    def to_full(coeffs_list):
        if not coeffs_list:
            return None
        max_e = max(e for e, c in coeffs_list)
        if max_e >= 41: # Length must be <= 41, so max index 40
            return None
        
        full = [0] * (max_e + 1)
        for e, c in coeffs_list:
            full[e] = c
        
        # Verify constraints
        if len(full) > 41:
            return None
        if full[-1] == 0:
            return None
        
        nz = sum(1 for x in full if x != 0)
        if nz != 12:
            return None
        if any(abs(x) > 64 for x in full):
            return None
            
        return full

    def try_cancellation_search(start_cl, time_budget=5.0):
        """
        Targeted search for coefficient patterns that create cancellations.
        Key insight: if ci*cj + ck*cl = 0 for some pairs, cross terms cancel.
        """
        nonlocal best_count, best_coeffs
        t0 = time.time()
        cl = [tuple(x) for x in start_cl]
        
        count = eval_square_fast(cl)
        try_update(count, cl)
        
        # Try specific coefficient patterns that promote cancellation
        # Pattern: pairs with opposite signs and equal magnitudes
        base_patterns = [
            [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
            [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
            [1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1],
            [2, -1, 1, -2, 2, -1, 1, -2, 2, -1, 1, -2],
            [1, 2, -1, -2, 1, 2, -1, -2, 1, 2, -1, -2],
        ]
        
        for pattern in base_patterns:
            if time.time() - t0 > time_budget:
                break
            cl2 = [(cl[i][0], pattern[i]) for i in range(12)]
            c2 = eval_square_fast(cl2)
            if c2 < count:
                cl = cl2
                count = c2
                try_update(count, cl)
        
        # Local search: try changing individual coefficients to promote cancellation
        vals = [-4, -3, -2, -1, 1, 2, 3, 4, -8, 8, -16, 16, -32, 32]
        
        for _ in range(200):
            if time.time() - t0 > time_budget:
                break
            if best_count <= 13:
                break
            
            idx = random.randrange(12)
            e, c = cl[idx]
            new_c = random.choice(vals)
            if new_c != c and abs(new_c) <= 64:
                cl2 = list(cl)
                cl2[idx] = (e, new_c)
                c2 = eval_square_fast(cl2)
                if c2 <= count:
                    cl = cl2
                    count = c2
                    try_update(count, cl)
        
        return cl, count

    def try_update(count, coeffs_list):
        nonlocal best_count, best_coeffs
        if count < best_count:
            full = to_full(coeffs_list)
            if full is not None:
                best_count = count
                best_coeffs = full
                return True
        return False

    def simulated_annealing(start_cl, time_budget=5.0):
        nonlocal best_count, best_coeffs
        t0 = time.time()
        cl = [tuple(x) for x in start_cl]
        if len(cl) != 12:
            return cl, float('inf')
        
        count = eval_square_fast(cl)
        try_update(count, cl)
        
        current_temp = 1.0
        cooling = 0.999
        min_temp = 0.01
        
        vals = [-4, -3, -2, -1, 1, 2, 3, 4, -8, 8, -16, 16, -32, 32]
        
        while time.time() - t0 < time_budget and current_temp > min_temp:
            # Pick a move
            move_type = random.random()
            
            if move_type < 0.5:
                # Change coefficient
                idx = random.randrange(12)
                e, c = cl[idx]
                new_c = random.choice(vals)
                if new_c != c and abs(new_c) <= 64:
                    cl2 = list(cl)
                    cl2[idx] = (e, new_c)
                    c2 = eval_square_fast(cl2)
                    delta = c2 - count
                    if delta < 0 or random.random() < np.exp(-delta / current_temp):
                        cl = cl2
                        count = c2
                        try_update(count, cl)
            elif move_type < 0.8:
                # Shift exponent
                idx = random.randrange(12)
                e, c = cl[idx]
                shift = random.choice([-3, -2, -1, 1, 2, 3])
                new_e = e + shift
                if 0 <= new_e <= 40:
                    exps = [cl[i][0] for i in range(12)]
                    exps.remove(e)
                    if new_e not in exps:
                        cl2 = list(cl)
                        cl2[idx] = (new_e, c)
                        c2 = eval_square_fast(cl2)
                        delta = c2 - count
                        if delta < 0 or random.random() < np.exp(-delta / current_temp):
                            cl = cl2
                            count = c2
                            try_update(count, cl)
            elif move_type < 0.95:
                # Swap coefficients of two terms
                i, j = random.sample(range(12), 2)
                e_i, c_i = cl[i]
                e_j, c_j = cl[j]
                if c_i != c_j:
                    cl2 = list(cl)
                    cl2[i] = (e_i, c_j)
                    cl2[j] = (e_j, c_i)
                    c2 = eval_square_fast(cl2)
                    delta = c2 - count
                    if delta < 0 or random.random() < np.exp(-delta / current_temp):
                        cl = cl2
                        count = c2
                        try_update(count, cl)
            else:
                # Pair coefficient change: change two coefficients simultaneously
                i, j = random.sample(range(12), 2)
                e_i, c_i = cl[i]
                e_j, c_j = cl[j]
                new_c_i = random.choice(vals)
                new_c_j = random.choice(vals)
                if (new_c_i != c_i or new_c_j != c_j) and abs(new_c_i) <= 64 and abs(new_c_j) <= 64:
                    cl2 = list(cl)
                    cl2[i] = (e_i, new_c_i)
                    cl2[j] = (e_j, new_c_j)
                    c2 = eval_square_fast(cl2)
                    delta = c2 - count
                    if delta < 0 or random.random() < np.exp(-delta / current_temp):
                        cl = cl2
                        count = c2
                        try_update(count, cl)
                        
            current_temp *= cooling
            
            if best_count <= 12:
                break
                
        return cl, count

    # =====================================================================
    # Phase 1: Structural Search via Factorization (Retained from original)
    # =====================================================================
    
    def convolve_sparse(poly1, poly2):
        res = {}
        for e1, c1 in poly1:
            for e2, c2 in poly2:
                e = e1 + e2
                res[e] = res.get(e, 0) + c1 * c2
        return [(e, c) for e, c in res.items() if c != 0]
    
    candidate_factors_pool = []
    for k in range(1, 11):
        for a in [-2, -1, 1, 2]:
            for b in [-2, -1, 1, 2]:
                if a == 0 or b == 0: continue
                candidate_factors_pool.append([(0, a), (k, b)])
                
    for k in range(1, 8):
        for a in [-2, -1, 1, 2]:
            for b in [-2, -1, 1, 2]:
                for c in [-2, -1, 1, 2]:
                    if a==0 or b==0 or c==0: continue
                    candidate_factors_pool.append([(0, a), (k, b), (2*k, c)])
                    if k + 1 <= 15:
                        candidate_factors_pool.append([(0, a), (k, b), (k+1, c)])
    
    pool_2 = [f for f in candidate_factors_pool if len(f) == 2]
    pool_3 = [f for f in candidate_factors_pool if len(f) == 3]
    
    pair_products = []
    for i in range(len(pool_2)):
        for j in range(i, len(pool_2)):
            prod = convolve_sparse(pool_2[i], pool_2[j])
            if len(prod) <= 4:
                pair_products.append(prod)
                
    seen_polys = set()
    
    for pp in pair_products:
        if time.time() > deadline - 15:
            break
        for c in pool_3:
            max_e_pp = max(e for e, _ in pp)
            max_e_c = max(e for e, _ in c)
            if max_e_pp + max_e_c > 40:
                continue
            
            p_poly = convolve_sparse(pp, c)
            if len(p_poly) != 12:
                continue
            
            p_sorted = sorted(p_poly)
            key = tuple(p_sorted)
            
            if key in seen_polys:
                continue
            seen_polys.add(key)
            
            count, _, _ = eval_square(p_sorted)
            try_update(count, p_sorted)
            
            if best_count == 12:
                return best_coeffs

    # =====================================================================
    # Phase 2: Simulated Annealing from Structural Seeds and Random Starts
    # =====================================================================
    
    random.seed(42)
    
    # Start from best found so far
    if best_coeffs is not None:
        start_cl = [(i, c) for i, c in enumerate(best_coeffs) if c != 0]
        simulated_annealing(start_cl, time_budget=10.0)
        if best_count == 12:
            return best_coeffs
    
    # Start from random 12-term polynomials
    vals = [-4, -3, -2, -1, 1, 2, 3, 4]
    
    attempts = 0
    while time.time() < deadline - 10:
        attempts += 1
        
        # Diverse initialization strategies
        r = random.random()
        if r < 0.3:
            # Consecutive
            exps = list(range(12))
        elif r < 0.6:
            # Arithmetic Progression
            step = random.randint(1, 3)
            if 11 * step > 40:
                step = 3
            exps = [i * step for i in range(12)]
        else:
            # Random
            exps = sorted(random.sample(range(41), 12))
        
        cl = [(exps[i], random.choice(vals)) for i in range(12)]
        
        # Run SA for a short time
        simulated_annealing(cl, time_budget=2.0)
        if best_count == 12:
            return best_coeffs

    # =====================================================================
    # Phase 3: Final Polish
    # =====================================================================
    if best_coeffs is not None:
        start_cl = [(i, c) for i, c in enumerate(best_coeffs) if c != 0]
        simulated_annealing(start_cl, time_budget=10.0)
        if best_count == 12:
            return best_coeffs

    # Fallback
    if best_coeffs is None:
        best_coeffs = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1] + [0]*29
        if best_coeffs[-1] == 0:
            for i in range(len(best_coeffs)-1, -1, -1):
                if best_coeffs[i] != 0:
                    best_coeffs = best_coeffs[:i+1]
                    break
            if not best_coeffs:
                best_coeffs = [1]
        
        nz = [i for i, c in enumerate(best_coeffs) if c != 0]
        if len(nz) != 12:
            best_coeffs = [0]*41
            for i in range(12):
                best_coeffs[i] = 1
            
    return best_coeffs

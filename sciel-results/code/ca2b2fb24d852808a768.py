"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    best_square_count = 100
    best_poly = [0] * 41
    for i in range(12):
        best_poly[i] = 1

    def count_nonzero_square(poly):
        p = np.array(poly, dtype=int)
        sq = np.convolve(p, p)
        return int(np.count_nonzero(sq))

    def evaluate(poly):
        if len(poly) != 41:
            return 1000, False
        if any(c < -64 or c > 64 for c in poly):
            return 1000, False
        nz_count = sum(1 for c in poly if c != 0)
        if nz_count != 12:
            return 1000, False
        sq_nz = count_nonzero_square(poly)
        return sq_nz, True

    best_square_count, _ = evaluate(best_poly)

    def local_search_factors(params):
        """
        params: dict with keys 'a0', 'a1', 'u1', 'b0', 'b1', 'b2', 'u2', 'u3', 'c0', 'c1', 'u4'
        Returns best_score, best_poly, best_params
        """
        current_params = params.copy()
        current_poly = [0]*41
        # Build poly from factors
        def build_poly(p):
            a = np.zeros(41, dtype=int)
            a[0] = p['a0']
            a[p['u1']] = p['a1']
            
            b = np.zeros(41, dtype=int)
            b[0] = p['b0']
            b[p['u2']] = p['b1']
            b[p['u3']] = p['b2']
            
            c = np.zeros(41, dtype=int)
            c[0] = p['c0']
            c[p['u4']] = p['c1']
            
            ab = np.convolve(a, b)
            abc = np.convolve(ab, c)
            # Trim to 41
            if len(abc) > 41:
                abc = abc[:41]
            else:
                abc = np.pad(abc, (0, 41-len(abc)), 'constant')
            return abc

        current_poly = build_poly(current_params)
        current_score, valid = evaluate(current_poly)
        if not valid:
            # Try to fix by adjusting exponents to be distinct and non-overlapping
            # Or just return invalid
            return 1000, [0]*41, current_params

        best_score = current_score
        best_poly = current_poly
        best_params = current_params.copy()

        temp = 5.0
        cooling_rate = 0.99
        no_improve = 0
        max_iters = 2000

        for i in range(max_iters):
            if time.time() > deadline:
                break
                
            new_params = current_params.copy()
            # Perturb one parameter
            keys = list(new_params.keys())
            key = random.choice(keys)
            
            if key.startswith('u'):
                # Exponent
                new_val = max(0, min(40, new_params[key] + random.randint(-3, 3)))
                new_params[key] = new_val
            else:
                # Coefficient
                new_val = random.randint(-5, 5)
                if new_val == 0:
                    new_val = 1
                new_params[key] = new_val
            
            new_poly = build_poly(new_params)
            new_score, new_valid = evaluate(new_poly)
            
            if new_valid:
                if new_score < current_score:
                    current_params = new_params
                    current_poly = new_poly
                    current_score = new_score
                    no_improve = 0
                    if current_score < best_score:
                        best_score = current_score
                        best_poly = current_poly
                        best_params = new_params.copy()
                else:
                    delta = new_score - current_score
                    if random.random() < np.exp(-delta / temp):
                        current_params = new_params
                        current_poly = new_poly
                        current_score = new_score
                    no_improve += 1
            else:
                no_improve += 1
                
            temp *= cooling_rate
            if temp < 0.01:
                temp = 0.01
                
            if no_improve > 100:
                break

        return best_score, best_poly, best_params

    # Initialize random factors
    def random_factors():
        return {
            'a0': random.choice([-2, -1, 1, 2]),
            'a1': random.choice([-2, -1, 1, 2]),
            'u1': random.randint(1, 10),
            'b0': random.choice([-2, -1, 1, 2]),
            'b1': random.choice([-2, -1, 1, 2]),
            'b2': random.choice([-2, -1, 1, 2]),
            'u2': random.randint(1, 10),
            'u3': random.randint(1, 10),
            'c0': random.choice([-2, -1, 1, 2]),
            'c1': random.choice([-2, -1, 1, 2]),
            'u4': random.randint(1, 10)
        }

    for restart in range(50):
        if time.time() > deadline:
            break
            
        params = random_factors()
        # Ensure exponents are distinct to avoid immediate zero terms in factors
        while params['u1'] == params['u2'] or params['u1'] == params['u3'] or params['u2'] == params['u3'] or \
              params['u1'] == params['u4'] or params['u2'] == params['u4'] or params['u3'] == params['u4']:
            params['u1'] = random.randint(1, 10)
            params['u2'] = random.randint(1, 10)
            params['u3'] = random.randint(1, 10)
            params['u4'] = random.randint(1, 10)
            
        score, poly, _ = local_search_factors(params)
        
        if score < best_square_count:
            best_square_count = score
            best_poly = poly

    # Final validation
    nz_count = sum(1 for c in best_poly if c != 0)
    if nz_count != 12 or any(c < -64 or c > 64 for c in best_poly):
        best_poly = [0] * 41
        for i in range(12):
            best_poly[i] = 1
        best_square_count = count_nonzero_square(best_poly)

    result = best_poly[:]
    while len(result) > 1 and result[-1] == 0:
        result.pop()
        
    if not result:
        result = [1]
        
    return result
# EVOLVE-BLOCK-END

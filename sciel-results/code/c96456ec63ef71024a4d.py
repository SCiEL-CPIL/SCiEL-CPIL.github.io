import numpy as np
import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    def count_nonzero(poly):
        return int(np.count_nonzero(poly))
    
    def poly_square(coeffs):
        return np.convolve(coeffs, coeffs)
    
    def evaluate(coeffs):
        sq = poly_square(coeffs)
        return count_nonzero(sq)
    
    def validate(coeffs):
        if len(coeffs) > 41 or len(coeffs) == 0:
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
    
    def update_best(c, score):
        nonlocal best_score, best_poly
        if score < best_score:
            best_score = score
            best_poly = c.copy()
            return best_score <= 12
    
    # Generate diverse initial candidates
    candidates = []
    
    # Known good patterns
    candidates.append([1]*12)
    candidates.append([1 if i%2==0 else -1 for i in range(12)])
    candidates.append([1]*6 + [-1]*6)
    candidates.append([1,1,1,1,1,1,-1,-1,-1,-1,-1,-1])
    candidates.append([1,1,1,1,1,-1,1,1,1,1,1,-1])
    
    # Factor-based: P = A * B where A, B are sparse
    # A has 2 terms, B has 6 terms -> P can have up to 12 terms
    for a_pos in range(0, 12, 3):
        for b_base in range(0, 4):
            A = np.zeros(12, dtype=np.int64)
            A[0] = 1
            A[a_pos] = random.choice([-1, 1])
            B = np.zeros(12, dtype=np.int64)
            for j in range(6):
                B[b_base + j] = random.choice([-1, 1])
            P = np.convolve(A, B)[:12]
            if count_nonzero(P) == 12 and P[-1] != 0:
                candidates.append(P.tolist())
    
    # More factor-based
    for _ in range(50):
        # P = (1 + a*x^i) * (b0 + b1*x + ... + b5*x^5)
        i = random.randint(1, 11)
        a = random.choice([-2, -1, 1, 2])
        B = [random.choice([-2, -1, 1, 2]) for _ in range(6)]
        A = np.zeros(12, dtype=np.int64)
        A[0] = 1
        A[i] = a
        B_arr = np.array(B, dtype=np.int64)
        P = np.convolve(A, B_arr)[:12]
        if count_nonzero(P) == 12 and P[-1] != 0 and np.all((P >= -64) & (P <= 64)):
            candidates.append(P.tolist())
    
    # Structured patterns
    random.seed(42)
    for _ in range(100):
        c = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
        if c[-1] == 0:
            c[-1] = 1
        candidates.append(c)
    
    # Symmetric and antisymmetric
    for _ in range(50):
        base = [random.choice([-2, -1, 1, 2]) for _ in range(6)]
        c = base + base[::-1]
        if c[-1] == 0:
            c[-1] = 1
        candidates.append(c)
    for _ in range(50):
        base = [random.choice([-2, -1, 1, 2]) for _ in range(6)]
        c = base + [-x for x in base[::-1]]
        if c[-1] == 0:
            c[-1] = 1
        candidates.append(c)
    
    # Evaluate all candidates
    for init_c in candidates:
        if time.time() > deadline:
            break
        curr_c = np.array(init_c, dtype=np.int64)
        if not validate(curr_c):
            continue
        curr_score = evaluate(curr_c)
        if update_best(curr_c, curr_score):
            break
    
    if best_poly is None:
        best_poly = np.array([1]*12, dtype=np.int64)
        best_score = evaluate(best_poly)
    
    # Constraint-guided local search: try to zero out specific square coefficients
    def targeted_local_search(curr_c, curr_score, max_iters=2000):
        nonlocal best_score, best_poly
        no_improvement = 0
        temp = 3.0
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Get current square
            sq = poly_square(curr_c)
            
            # Find nonzero square coefficients
            nz_sq = np.where(sq != 0)[0]
            
            if len(nz_sq) == 0:
                break
            
            # Pick a target: try to zero out a specific square coefficient
            # Or pick a random polynomial coefficient to modify
            if random.random() < 0.5:
                # Random modification
                idx = random.randint(0, 11)
            else:
                # Targeted: pick a square coefficient that's nonzero and try to zero it
                target_deg = random.choice(nz_sq)
                # Find which polynomial coefficients affect this degree
                # coeff at degree k in square = sum_{i+j=k} a_i*a_j
                # For k <= 11: pairs are (0,k), (1,k-1), ..., (k,0)
                # For k > 11: pairs are (k-11, 11), (k-10, 10), ..., (11, k-11)
                if target_deg <= 11:
                    affecting = list(range(0, target_deg + 1))
                else:
                    affecting = list(range(target_deg - 11, 12))
                
                if not affecting:
                    idx = random.randint(0, 11)
                else:
                    idx = random.choice(affecting)
            
            # Generate new value
            new_val = random.randint(-4, 4)
            if new_val == 0:
                new_val = random.choice([-1, 1])
            
            if curr_c[idx] == new_val:
                continue
            
            old_val = curr_c[idx]
            curr_c[idx] = new_val
            
            new_score = evaluate(curr_c)
            
            if new_score < curr_score:
                curr_score = new_score
                no_improvement = 0
                if update_best(curr_c, curr_score):
                    return True
            else:
                delta = new_score - curr_score
                if random.random() < np.exp(-delta / max(temp, 0.5)):
                    curr_score = new_score
                    no_improvement = 0
                    if curr_score < best_score:
                        update_best(curr_c, curr_score)
                else:
                    curr_c[idx] = old_val
                    no_improvement += 1
            
            temp *= 0.999
            if no_improvement > 500:
                break
        
        return best_score <= 12
    
    # Run targeted local search from best and several other starting points
    starting_points = [best_poly.copy()]
    
    # Add some diverse starting points
    for _ in range(10):
        c = [random.choice([-3, -2, -1, 1, 2, 3]) for _ in range(12)]
        if c[-1] == 0:
            c[-1] = 1
        c_arr = np.array(c, dtype=np.int64)
        if validate(c_arr):
            starting_points.append(c_arr)
    
    for sp in starting_points:
        if time.time() > deadline - 5:
            break
        if targeted_local_search(sp.copy(), evaluate(sp), max_iters=1500):
            break
    
    # Multi-point mutation search
    if time.time() < deadline - 10:
        curr_c = best_poly.copy()
        curr_score = evaluate(curr_c)
        no_improvement = 0
        
        while time.time() < deadline - 5 and no_improvement < 3000:
            # Modify 1-3 coefficients at once
            num_mods = random.choice([1, 1, 2, 2, 3])
            indices = random.sample(range(12), num_mods)
            old_vals = [curr_c[i] for i in indices]
            
            for i in indices:
                val = random.randint(-5, 5)
                if val == 0:
                    val = random.choice([-1, 1])
                curr_c[i] = val
            
            new_score = evaluate(curr_c)
            
            if new_score < curr_score:
                curr_score = new_score
                no_improvement = 0
                if update_best(curr_c, curr_score):
                    break
            else:
                for i, val in zip(indices, old_vals):
                    curr_c[i] = val
                no_improvement += 1
    
    # Final: try to refine with very small perturbations
    if time.time() < deadline - 3 and best_poly is not None:
        curr_c = best_poly.copy()
        curr_score = evaluate(curr_c)
        
        for _ in range(500):
            if time.time() > deadline - 2:
                break
            idx = random.randint(0, 11)
            delta_val = random.choice([-2, -1, 1, 2])
            new_val = curr_c[idx] + delta_val
            if new_val == 0:
                new_val = 1
            if not (-64 <= new_val <= 64):
                continue
            
            old_val = curr_c[idx]
            curr_c[idx] = new_val
            new_score = evaluate(curr_c)
            
            if new_score < curr_score:
                curr_score = new_score
                if update_best(curr_c, curr_score):
                    break
            else:
                curr_c[idx] = old_val
    
    if best_poly is None:
        best_poly = np.array([1]*12, dtype=np.int64)
    
    while len(best_poly) > 0 and best_poly[-1] == 0:
        best_poly = best_poly[:-1]
    
    return best_poly.tolist()

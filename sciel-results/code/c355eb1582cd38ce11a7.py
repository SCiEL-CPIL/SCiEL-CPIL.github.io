import random
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    best_score = 1000  # Max possible is 2*40+1 = 81, but we start high
    best_poly = None
    
    # Helper to calculate score
    def get_score(poly):
        # poly is list of (exp, coeff)
        # Square it
        # Use dict for convolution
        square = {}
        for i in range(len(poly)):
            e1, c1 = poly[i]
            for j in range(len(poly)):
                e2, c2 = poly[j]
                e_sum = e1 + e2
                c_prod = c1 * c2
                if e_sum in square:
                    square[e_sum] += c_prod
                else:
                    square[e_sum] = c_prod
        
        # Count nonzero
        count = 0
        for v in square.values():
            if v != 0:
                count += 1
        return count

    def random_poly():
        # 12 distinct exponents from 0 to 40
        exps = random.sample(range(41), 12)
        # Coefficients from small set to encourage cancellation?
        # Or just +/- 1?
        # Let's try coefficients from {-2, -1, 1, 2}
        coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
        # Sort by exponent
        pairs = sorted(zip(exps, coeffs))
        return pairs

    def neighbor(poly):
        p = [x for x in poly]
        move_type = random.randint(0, 3)
        
        if move_type == 0:
            # Change exponent of one term
            idx = random.randint(0, 11)
            old_exp = p[idx][0]
            new_exp = random.randint(0, 40)
            # Ensure distinct
            if new_exp != old_exp and all(p[i][0] != new_exp for i in range(12) if i != idx):
                p[idx] = (new_exp, p[idx][1])
                p.sort()
        elif move_type == 1:
            # Change coefficient of one term
            idx = random.randint(0, 11)
            new_coeff = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
            if new_coeff != p[idx][1]:
                p[idx] = (p[idx][0], new_coeff)
        elif move_type == 2:
            # Swap two exponents
            i, j = random.sample(range(12), 2)
            e_i, c_i = p[i]
            e_j, c_j = p[j]
            p[i] = (e_j, c_i)
            p[j] = (e_i, c_j)
            p.sort()
        elif move_type == 3:
            # Swap two coefficients
            i, j = random.sample(range(12), 2)
            e_i, c_i = p[i]
            e_j, c_j = p[j]
            p[i] = (e_i, c_j)
            p[j] = (e_j, c_i)
            p.sort()
            
        return p

    # Initial population
    current_poly = random_poly()
    current_score = get_score(current_poly)
    
    if current_score < best_score:
        best_score = current_score
        best_poly = current_poly

    # Simulated Annealing / Hill Climbing
    # We don't strictly need annealing if we do many random restarts,
    # but simple hill climbing with random restarts is often sufficient for this small space.
    
    restarts = 0
    while time.time() < deadline:
        # Local search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for _ in range(100): # Fixed steps per local search phase
                if time.time() > deadline:
                    break
                new_poly = neighbor(current_poly)
                new_score = get_score(new_poly)
                
                # Accept if better, or with small probability if equal (to escape plateaus)
                if new_score < current_score or (new_score == current_score and random.random() < 0.1):
                    current_poly = new_poly
                    current_score = new_score
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_poly = current_poly
                        if best_score <= 20: # Good enough?
                             # 23 is AP. Can we beat 23?
                             # If we find 22, that's a jump.
                             pass

        # Random restart
        current_poly = random_poly()
        current_score = get_score(current_poly)
        if current_score < best_score:
            best_score = current_score
            best_poly = current_poly
        restarts += 1
        
        # If we found a very good score, we can stop or keep going
        # Theoretical min for 12 terms is likely around 23, but cancellations might reduce it.
        # If best_score is 23, we might want to keep searching for 22.
        
    # Convert best_poly to list format
    # Must return coefficient list in ascending degree, length <= 41
    # Nonzero final coefficient
    # Entries integers in [-64, 64]
    
    if best_poly is None:
        return [1] * 12 # Fallback

    max_exp = max(e for e, c in best_poly)
    result = [0] * (max_exp + 1)
    for e, c in best_poly:
        result[e] = c
        
    # Ensure final coefficient is nonzero (it is, by construction)
    # Ensure length <= 41 (max_exp <= 40)
    
    return result

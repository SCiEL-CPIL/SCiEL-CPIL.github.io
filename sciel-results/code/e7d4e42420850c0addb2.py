import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Initial seed
    best_poly = [0] * 41
    for i in range(12):
        best_poly[i] = 1
        
    def count_nz_square(poly):
        # poly is list of coeffs
        n = len(poly)
        max_deg = 2 * (n - 1)
        sq = [0] * (max_deg + 1)
        # Optimization: only iterate non-zeros
        nz_indices = [i for i in range(n) if poly[i] != 0]
        for i in nz_indices:
            ci = poly[i]
            for j in nz_indices:
                cj = poly[j]
                sq[i + j] += ci * cj
        return sum(1 for c in sq if c != 0)
        
    def get_minimal_list(poly):
        # Trim trailing zeros
        i = len(poly) - 1
        while i >= 0 and poly[i] == 0:
            i -= 1
        if i < 0:
            return [0]
        return poly[:i+1]
        
    best_val = count_nz_square(best_poly)
    best_result = get_minimal_list(best_poly)
    
    max_deg = 40
    
    while time.time() < deadline:
        # Random restart
        # Generate 12 random unique exponents
        exps = random.sample(range(max_deg + 1), 12)
        exps.sort()
        coefs = [random.choice([-1, 1]) * random.randint(1, 64) for _ in range(12)]
        
        # Build poly
        poly = [0] * (max_deg + 1)
        for e, c in zip(exps, coefs):
            poly[e] = c
            
        current_val = count_nz_square(poly)
        
        if current_val < best_val:
            best_val = current_val
            best_result = get_minimal_list(poly[:])
            
        # Local Search
        T = 2.0
        cooling = 0.9995
        min_T = 0.01
        
        # Limit steps to avoid long single runs, allow more restarts
        # But we have 85s, so we can do a lot.
        # Let's do up to 2000 steps or until T < min_T
        for step in range(2000):
            if time.time() > deadline - 2.0:
                break
            if T < min_T:
                break
                
            # Propose move on (exps, coefs)
            # We need to maintain consistency.
            # Let's work with the list of (exp, coef) pairs directly.
            # items = list(zip(exps, coefs))
            
            items = list(zip(exps, coefs))
            idx = random.randint(0, 11)
            
            move_type = random.randint(0, 2)
            
            if move_type == 0:
                # Change coefficient
                new_c = random.choice([-1, 1]) * random.randint(1, 64)
                items[idx] = (items[idx][0], new_c)
            elif move_type == 1:
                # Change exponent
                new_e = random.randint(0, max_deg)
                # Check if new_e is already in other items
                valid = True
                for k in range(12):
                    if k != idx and items[k][0] == new_e:
                        valid = False
                        break
                if valid:
                    items[idx] = (new_e, items[idx][1])
                else:
                    continue # Skip this move
            else:
                # Swap exponents between two random items
                j = random.randint(0, 11)
                if j == idx:
                    continue
                e_i, c_i = items[idx]
                e_j, c_j = items[j]
                items[idx] = (e_j, c_i)
                items[j] = (e_i, c_j)
                
            # Rebuild poly to calculate cost
            # This is O(12) to build, O(12^2) to square.
            # We can optimize cost calculation by only updating affected terms, 
            # but rebuilding is simpler and likely fast enough for 12 terms.
            new_poly = [0] * (max_deg + 1)
            for e, c in items:
                new_poly[e] = c
                
            new_val = count_nz_square(new_poly)
            
            # Accept
            if new_val < current_val:
                current_val = new_val
                exps, coefs = zip(*items)
                exps = list(exps)
                coefs = list(coefs)
                
                if current_val < best_val:
                    best_val = current_val
                    best_result = get_minimal_list(new_poly)
            else:
                # Simulated Annealing
                # delta = new_val - current_val (positive)
                # prob = exp(-delta / T)
                if random.random() < pow(2.718281828, -(new_val - current_val) / T):
                    current_val = new_val
                    exps, coefs = zip(*items)
                    exps = list(exps)
                    coefs = list(coefs)
                    
            T *= cooling
            
    return best_result

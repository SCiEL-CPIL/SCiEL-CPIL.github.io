import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    n = 12
    m = 40
    N = 1 << n  # 4096
    
    def is_comp(a, b):
        if a == b:
            return False
        if (a & b) == a:
            return True
        if (a & b) == b:
            return True
        return False

    best_score = -1
    best_family = None
    
    # We will run multiple restarts of simulated annealing
    while time.time() < deadline:
        # Initialize family with random distinct masks
        family = []
        present = [False] * N
        while len(family) < m:
            mask = random.randint(0, N - 1)
            if not present[mask]:
                family.append(mask)
                present[mask] = True
        
        # Calculate initial C array and score
        C = [0] * m
        score = 0
        for i in range(m):
            for j in range(i + 1, m):
                if is_comp(family[i], family[j]):
                    C[i] += 1
                    C[j] += 1
                    score += 1
        
        if score > best_score:
            best_score = score
            best_family = family.copy()
            
        # Local Search with Simulated Annealing
        T = 1.0
        T_min = 0.01
        cooling = 0.995
        
        # Run until temperature is low or time is up
        while T > T_min and time.time() < deadline - 1:
            i = random.randint(0, m - 1)
            move_type = random.random()
            
            if move_type < 0.5:
                # Flip a bit
                bit = random.randint(0, n - 1)
                new_mask = family[i] ^ (1 << bit)
                if present[new_mask]:
                    continue
            else:
                # Replace with random new mask
                attempts = 0
                new_mask = -1
                while attempts < 10:
                    cand = random.randint(0, N - 1)
                    if not present[cand]:
                        new_mask = cand
                        break
                    attempts += 1
                if new_mask == -1:
                    continue
            
            old_mask = family[i]
            
            # Calculate delta score and deltas for C array
            new_c_i = 0
            deltas = [0] * m
            
            for j in range(m):
                if j == i:
                    continue
                mj = family[j]
                old_comp = 1 if is_comp(old_mask, mj) else 0
                new_comp = 1 if is_comp(new_mask, mj) else 0
                
                if old_comp != new_comp:
                    if new_comp:
                        new_c_i += 1
                        deltas[j] += 1
                    else:
                        deltas[j] -= 1
                        # new_c_i does not increase, but old C[i] had this contribution
                        # So delta_score will be lower.
                        # We just need new_c_i to be the sum of new comps.
                        # The logic above:
                        # if new_comp is 1, new_c_i += 1.
                        # if new_comp is 0, new_c_i += 0.
                        # This is correct.
            
            delta_score = new_c_i - C[i]
            
            # Acceptance criterion
            if delta_score > 0 or (T > 0 and random.random() < pow(2.71828, delta_score / T)):
                # Accept move
                present[old_mask] = False
                present[new_mask] = True
                family[i] = new_mask
                score += delta_score
                C[i] = new_c_i
                for j in range(m):
                    if j != i:
                        C[j] += deltas[j]
                        
                if score > best_score:
                    best_score = score
                    best_family = family.copy()
            
            T *= cooling

    # Fallback if no valid family was found (should not happen)
    if best_family is None:
        return [[b for b in range(n) if (mask >> b) & 1] for mask in range(m)]
    
    # Convert best_family (list of masks) to list of lists of ints
    result = []
    for mask in best_family:
        lst = []
        for b in range(n):
            if (mask >> b) & 1:
                lst.append(b)
        result.append(lst)
    
    return result

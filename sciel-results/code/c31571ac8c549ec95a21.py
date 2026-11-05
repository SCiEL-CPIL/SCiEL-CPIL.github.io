# EVOLVE-BLOCK-START
from fractions import Fraction
import random
import itertools

def construct():
    # Set a seed for reproducibility
    random.seed(42)
    
    best_sum = None
    best_coeff = None
    
    # Precompute reciprocals
    recips = [Fraction(1, i + 1) for i in range(30)]
    
    def calc_sum(coeff):
        s = Fraction(0, 1)
        for i, c in enumerate(coeff):
            if c != 0:
                s += c * recips[i]
        return s

    def local_search(coeff):
        current_coeff = coeff[:]
        current_sum = calc_sum(current_coeff)
        
        if current_sum == 0:
            # Perturb to get out of zero if we start there (shouldn't happen with good init, but safe)
            idx = random.randint(0, 29)
            current_coeff[idx] = 1 if current_coeff[idx] != 1 else -1
            current_sum = calc_sum(current_coeff)
            if current_sum == 0:
                current_coeff[idx] = 0
                current_sum = calc_sum(current_coeff)
                if current_sum == 0:
                    return None, None

        improved = True
        max_iters = 50
        iters = 0
        
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            # Phase 1: Steepest Ascent 1-flips
            best_move_1 = None
            best_val_1 = None
            
            for i in range(30):
                old_val = current_coeff[i]
                for new_val in [-1, 0, 1]:
                    if new_val == old_val:
                        continue
                    delta = (new_val - old_val) * recips[i]
                    new_sum = current_sum + delta
                    if new_sum == 0:
                        continue
                    if best_val_1 is None or abs(new_sum) < abs(best_val_1):
                        best_val_1 = new_sum
                        best_move_1 = (i, new_val)
            
            if best_move_1 is not None:
                i, new_val = best_move_1
                current_coeff[i] = new_val
                current_sum = best_val_1
                improved = True
                continue # Restart to check again immediately
                
            # Phase 2: Steepest Ascent 2-flips
            # To limit complexity, we only try 2-flips if 1-flip didn't improve
            best_move_2 = None
            best_val_2 = None
            
            # Sample a subset of pairs to keep time low, or all pairs if fast enough
            # 30*29/2 = 435 pairs. For each pair, 4 value combos. ~1700 checks.
            # This is fast.
            for i in range(30):
                old_i = current_coeff[i]
                for j in range(i + 1, 30):
                    old_j = current_coeff[j]
                    # Try all combinations of changes for i and j
                    # We can just iterate new_i, new_j
                    for new_i in [-1, 0, 1]:
                        if new_i == old_i:
                            continue
                        for new_j in [-1, 0, 1]:
                            if new_j == old_j:
                                continue
                            delta = (new_i - old_i) * recips[i] + (new_j - old_j) * recips[j]
                            new_sum = current_sum + delta
                            if new_sum == 0:
                                continue
                            if best_val_2 is None or abs(new_sum) < abs(best_val_2):
                                best_val_2 = new_sum
                                best_move_2 = (i, new_i, j, new_j)
            
            if best_move_2 is not None:
                i, new_i, j, new_j = best_move_2
                current_coeff[i] = new_i
                current_coeff[j] = new_j
                current_sum = best_val_2
                improved = True
                continue # Restart

            # No improvement found
            break

        if current_sum == 0:
            return None, None
            
        return current_coeff, abs(current_sum)

    # Try multiple random starts
    starts = []
    
    # Greedy start 1: Fill from high to low (smaller denominators first? No, i+1 is denom)
    # High i means small fraction. 
    c_greedy = [0] * 30
    s_greedy = Fraction(0, 1)
    for i in range(29, -1, -1):
        best_c = 0
        best_s = s_greedy
        for c in [-1, 0, 1]:
            s_new = s_greedy + c * recips[i]
            if abs(s_new) < abs(best_s):
                best_s = s_new
                best_c = c
        c_greedy[i] = best_c
        s_greedy = best_s
    starts.append(c_greedy)
    
    # Greedy start 2: Fill from low to high
    c_greedy2 = [0] * 30
    s_greedy2 = Fraction(0, 1)
    for i in range(30):
        best_c = 0
        best_s = s_greedy2
        for c in [-1, 0, 1]:
            s_new = s_greedy2 + c * recips[i]
            if abs(s_new) < abs(best_s):
                best_s = s_new
                best_c = c
        c_greedy2[i] = best_c
        s_greedy2 = best_s
    starts.append(c_greedy2)
    
    # Random starts
    for _ in range(300):
        coeff = [0] * 30
        # Sparse random
        num_nonzero = random.randint(1, 15)
        indices = random.sample(range(30), num_nonzero)
        for idx in indices:
            coeff[idx] = random.choice([-1, 1])
        starts.append(coeff)
        
    # Dense random
    for _ in range(100):
        coeff = [random.choice([-1, 0, 1]) for _ in range(30)]
        starts.append(coeff)
        
    # Balanced random (equal number of +1 and -1)
    for _ in range(100):
        coeff = [0] * 30
        num_pos = random.randint(5, 15)
        num_neg = num_pos
        indices = random.sample(range(30), num_pos + num_neg)
        for k in range(num_pos):
            coeff[indices[k]] = 1
        for k in range(num_neg):
            coeff[indices[num_pos + k]] = -1
        starts.append(coeff)

    for init_coeff in starts:
        res_coeff, res_sum = local_search(init_coeff)
        
        if res_coeff is not None:
            if best_sum is None or res_sum < best_sum:
                best_sum = res_sum
                best_coeff = res_coeff

    # Fallback
    if best_coeff is None:
        best_coeff = [0] * 30
        best_coeff[28] = 1
        best_coeff[29] = -1
        
    return best_coeff
# EVOLVE-BLOCK-END

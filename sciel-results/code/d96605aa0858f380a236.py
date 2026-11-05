"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

def _set_sizes(A):
    """Compute |A+A| and |A*A| for set A (with equal elements allowed)."""
    n = len(A)
    sums = set()
    prods = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            sums.add(ai + aj)
            prods.add(ai * aj)
    return len(sums), len(prods)

def _objective(A):
    """Return max(|A+A|, |A*A|)."""
    s, p = _set_sizes(A)
    return max(s, p)

# EVOLVE-BLOCK-START
import math

def _fast_set_sizes(A):
    """
    Compute |A+A| and |A*A| for set A.
    Optimized: Uses a boolean array for sums (max sum 20000) and a set for products.
    """
    n = len(A)
    # Max sum is 10000 + 10000 = 20000
    sum_arr = [0] * 20001
    sum_count = 0
    prod_set = set()
    
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            s = ai + aj
            if sum_arr[s] == 0:
                sum_arr[s] = 1
                sum_count += 1
            prod_set.add(ai * aj)
            
    return sum_count, len(prod_set)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Helper to validate and compute objective
    def get_obj(A):
        if len(A) != 20:
            return None
        if len(set(A)) != 20:
            return None
        if any(x < 1 or x > 10000 for x in A):
            return None
        s, p = _fast_set_sizes(A)
        return max(s, p)

    best_set = None
    best_obj = float('inf')
    
    # Candidate generators
    def gen_ap():
        return list(range(1, 21))
        
    def gen_gp():
        # Powers of 2, scaled to fit? 
        # 2^19 is 524288, too big.
        # We need 20 distinct integers <= 10000.
        # Max ratio: 10000.
        # 10000^(1/19) approx 1.8.
        # So base 2 is too big for 20 terms if start is 1.
        # 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096. (13 terms)
        # We can mix.
        # Let's just use random small integers.
        return sorted(random.sample(range(1, 100), 20))
        
    def gen_random_small():
        return sorted(random.sample(range(1, 200), 20))
        
    def gen_random_med():
        return sorted(random.sample(range(1, 1000), 20))

    # Initial candidates
    candidates = [
        gen_ap(),
        gen_gp(),
        gen_random_small(),
        gen_random_med(),
        list(range(2, 22)),
        list(range(1, 21, 2)) + list(range(20, 40, 2))[:0] # Just APs
    ]
    
    # Add some structured randoms
    for _ in range(50):
        # Perturb AP
        ap = list(range(1, 21))
        for _ in range(random.randint(1, 5)):
            idx = random.randint(0, 19)
            ap[idx] += random.randint(-5, 5)
        if len(set(ap)) == 20 and all(1 <= x <= 10000 for x in ap):
            candidates.append(sorted(ap))
            
    # Evaluate initial candidates
    for cand in candidates:
        obj = get_obj(cand)
        if obj is not None and obj < best_obj:
            best_obj = obj
            best_set = cand[:]
            
    if best_set is None:
        best_set = list(range(1, 21))
        best_obj = get_obj(best_set)
        
    current_set = best_set[:]
    current_obj = best_obj
    
    # Simulated Annealing
    T = 50.0
    T_min = 0.1
    cooling = 0.99
    
    while time.time() < deadline and T > T_min:
        # Adaptive steps per temp
        # If we have time, do more steps
        steps = 100
        for _ in range(steps):
            if time.time() > deadline:
                break
                
            # Generate neighbor
            # Strategy: Replace 1 element
            idx = random.randint(0, 19)
            old_val = current_set[idx]
            
            # Generate new value
            # Bias towards local changes
            if random.random() < 0.7:
                # Local
                delta = random.randint(-10, 10)
                if delta == 0: delta = 1
                new_val = old_val + delta
            else:
                # Global
                new_val = random.randint(1, 10000)
                
            if new_val < 1 or new_val > 10000:
                continue
                
            # Check distinctness
            if new_val in current_set and current_set.index(new_val) != idx:
                continue
                
            new_set = current_set[:]
            new_set[idx] = new_val
            new_set.sort()
            
            new_obj = get_obj(new_set)
            if new_obj is None:
                continue
                
            delta_obj = new_obj - current_obj
            
            # Acceptance
            if delta_obj < 0 or random.random() < math.exp(-delta_obj / T):
                current_set = new_set
                current_obj = new_obj
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_set = current_set[:]
                    
        T *= cooling
        
        # Optional: Restart if stuck?
        # If no improvement for a while, try a random restart from best?
        # For now, just cool.
        
    return best_set
# EVOLVE-BLOCK-END

import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    # Generate 5-smooth numbers <= 10000
    pool = set()
    a = 1
    while a <= 10000:
        b = a
        while b <= 10000:
            c = b
            while c <= 10000:
                pool.add(c)
                c *= 5
            b *= 3
        a *= 2
    
    pool = sorted(list(pool))
    
    if len(pool) < 20:
        return list(range(1, 21))

    # Objective function
    def calc_score(A_list):
        sums = set()
        prods = set()
        for i in range(len(A_list)):
            ai = A_list[i]
            for j in range(i, len(A_list)):
                aj = A_list[j]
                sums.add(ai + aj)
                prods.add(ai * aj)
        return max(len(sums), len(prods))

    # Initialize with smallest 20
    A = pool[:20]
    A_set = set(A)
    
    current_score = calc_score(A)
    best_score = current_score
    best_A = list(A)
    
    # Local Search
    # We will perform multiple passes of hill climbing.
    # If no improvement in a pass, we perturb and try again.
    
    passes = 0
    max_passes = 200 # Safety limit on passes, though time is the real limit
    
    while time.time() < deadline and passes < max_passes:
        improved = False
        passes += 1
        
        # Shuffle pool to avoid bias? 
        # Actually, deterministic order is fine for hill climbing, 
        # but random order helps escape symmetric local minima.
        # Let's use a fixed random seed for reproducibility in logic, 
        # but randomize the swap order each pass.
        random.shuffle(pool)
        
        for i in range(20):
            if time.time() > deadline:
                break
            for new_val in pool:
                if new_val in A_set:
                    continue
                
                old_val = A[i]
                if old_val == new_val:
                    continue
                
                # Try swap
                A[i] = new_val
                A_set.remove(old_val)
                A_set.add(new_val)
                
                new_score = calc_score(A)
                
                if new_score < current_score:
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_A = list(A)
                    improved = True
                else:
                    # Revert
                    A[i] = old_val
                    A_set.add(old_val)
                    A_set.remove(new_val)
        
        if not improved:
            # Perturbation: replace 3 random elements
            for _ in range(3):
                idx = random.randint(0, 19)
                new_val = random.choice(pool)
                if new_val not in A_set:
                    old_val = A[idx]
                    A[idx] = new_val
                    A_set.remove(old_val)
                    A_set.add(new_val)
            current_score = calc_score(A)
            # Don't update best_score here, only on improvement in next pass
            # Or update if this perturbation happened to be good? 
            # Usually perturbation worsens it.
            
    return best_A

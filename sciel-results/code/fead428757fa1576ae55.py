import time
import random
import math

def generate_smooth_pool(max_val=10000, primes=(2, 3, 5, 7)):
    """Generate numbers with prime factors in 'primes' <= max_val."""
    smooth = set([1])
    # Simple sieve-like generation or recursive
    # Given max_val=10000, recursion depth is small.
    def gen(current, idx):
        if current > max_val:
            return
        smooth.add(current)
        if idx < len(primes):
            p = primes[idx]
            next_val = current
            while next_val <= max_val:
                gen(next_val, idx)
                next_val *= p
            gen(current, idx + 1)
    
    gen(1, 0)
    # Add small integers to ensure coverage
    for i in range(1, 101):
        smooth.add(i)
    return sorted([x for x in smooth if x <= max_val])

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    pool = generate_smooth_pool(10000)
    # Limit pool size to keep swap selection fast? 
    # random.choice(pool) is O(1). 
    # But if pool is huge, we might miss good local swaps?
    # Let's keep pool around 500-1000 elements.
    if len(pool) > 800:
        # Keep smallest 800? Or random sample?
        # Smaller numbers have more collisions.
        pool = pool[:800]
    
    n = 20
    
    def compute_initial_metrics(A):
        sum_counts = {}
        prod_counts = {}
        for i in range(n):
            a = A[i]
            for j in range(i, n):
                b = A[j]
                s = a + b
                p = a * b
                sum_counts[s] = sum_counts.get(s, 0) + 1
                prod_counts[p] = prod_counts.get(p, 0) + 1
        return sum_counts, prod_counts, max(len(sum_counts), len(prod_counts))

    def get_score(sum_counts, prod_counts):
        return max(len(sum_counts), len(prod_counts))

    # Helper to update counts for a swap
    # Returns new sum_counts, prod_counts, new_score
    # We will mutate the dicts to save time, then rollback if rejected?
    # Or copy? Copying dicts of size ~200 is fast.
    # Let's mutate and rollback.
    
    def apply_swap(A, idx_old, val_old, idx_new, val_new, sum_counts, prod_counts):
        """
        Replaces A[idx_old] (val_old) with val_new.
        Assumes val_new is not in A.
        Updates sum_counts and prod_counts in place.
        Returns new score.
        """
        # 1. Remove val_old contributions
        # Pair (val_old, val_old)
        s_oo = 2 * val_old
        p_oo = val_old * val_old
        sum_counts[s_oo] -= 1
        if sum_counts[s_oo] == 0: del sum_counts[s_oo]
        prod_counts[p_oo] -= 1
        if prod_counts[p_oo] == 0: del prod_counts[p_oo]
        
        # Pairs (val_old, A[j]) for j != idx_old
        for j in range(n):
            if j == idx_old: continue
            a_j = A[j]
            s = val_old + a_j
            p = val_old * a_j
            sum_counts[s] -= 1
            if sum_counts[s] == 0: del sum_counts[s]
            prod_counts[p] -= 1
            if prod_counts[p] == 0: del prod_counts[p]
            
        # 2. Update A
        A[idx_old] = val_new
        
        # 3. Add val_new contributions
        # Pair (val_new, val_new)
        s_nn = 2 * val_new
        p_nn = val_new * val_new
        sum_counts[s_nn] = sum_counts.get(s_nn, 0) + 1
        prod_counts[p_nn] = prod_counts.get(p_nn, 0) + 1
        
        # Pairs (val_new, A[j]) for j != idx_old
        for j in range(n):
            if j == idx_old: continue
            a_j = A[j]
            s = val_new + a_j
            p = val_new * a_j
            sum_counts[s] = sum_counts.get(s, 0) + 1
            prod_counts[p] = prod_counts.get(p, 0) + 1
            
        return get_score(sum_counts, prod_counts)

    def rollback_swap(A, idx_old, val_old, val_new, sum_counts, prod_counts):
        """
        Reverts the swap: A[idx_old] is currently val_new, revert to val_old.
        """
        # 1. Remove val_new contributions
        s_nn = 2 * val_new
        p_nn = val_new * val_new
        sum_counts[s_nn] -= 1
        if sum_counts[s_nn] == 0: del sum_counts[s_nn]
        prod_counts[p_nn] -= 1
        if prod_counts[p_nn] == 0: del prod_counts[p_nn]
        
        for j in range(n):
            if j == idx_old: continue
            a_j = A[j]
            s = val_new + a_j
            p = val_new * a_j
            sum_counts[s] -= 1
            if sum_counts[s] == 0: del sum_counts[s]
            prod_counts[p] -= 1
            if prod_counts[p] == 0: del prod_counts[p]
            
        # 2. Update A
        A[idx_old] = val_old
        
        # 3. Add val_old contributions
        s_oo = 2 * val_old
        p_oo = val_old * val_old
        sum_counts[s_oo] = sum_counts.get(s_oo, 0) + 1
        prod_counts[p_oo] = prod_counts.get(p_oo, 0) + 1
        
        for j in range(n):
            if j == idx_old: continue
            a_j = A[j]
            s = val_old + a_j
            p = val_old * a_j
            sum_counts[s] = sum_counts.get(s, 0) + 1
            prod_counts[p] = prod_counts.get(p, 0) + 1

    # Main Search Loop
    best_set = None
    best_score = float('inf')
    
    random.seed(42)
    
    num_restarts = 10
    time_per_restart = (deadline - start_time) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 2.0:
            break
        
        # Initialize current state
        # Strategy: Mix of random and greedy-like starts
        if restart == 0:
            # Start with first 20 smooth numbers
            current_set = pool[:20]
        elif restart % 2 == 0:
            # Random from small part of pool
            current_set = sorted(random.sample(pool[:200], 20))
        else:
            # Random from full pool
            current_set = sorted(random.sample(pool, 20))
        
        # Ensure distinct and valid
        if len(set(current_set)) != 20 or any(x < 1 or x > 10000 for x in current_set):
            current_set = list(range(1, 21))
            
        sum_counts, prod_counts, current_score = compute_initial_metrics(current_set)
        
        if current_score < best_score:
            best_score = current_score
            best_set = current_set[:]
            best_sum_counts = sum_counts.copy()
            best_prod_counts = prod_counts.copy()
        else:
            # If not better, we still use this state for SA
            pass
        
        # Simulated Annealing
        T_start = 10.0
        T_end = 0.01
        max_steps = 500000 # Increased due to speedup
        
        step = 0
        while step < max_steps:
            if time.time() > deadline - 2.0:
                break
            
            step += 1
            T = T_start * (T_end / T_start) ** (step / max_steps)
            
            # Single Swap
            idx = random.randint(0, n - 1)
            old_val = current_set[idx]
            new_val = random.choice(pool)
            
            if new_val == old_val or new_val in current_set:
                continue
            
            # Apply swap
            new_score = apply_swap(current_set, idx, old_val, new_val, sum_counts, prod_counts)
            
            delta = new_score - current_score
            
            if delta <= 0 or random.random() < math.exp(-delta / T):
                # Accept
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_set = current_set[:]
                    best_sum_counts = sum_counts.copy()
                    best_prod_counts = prod_counts.copy()
            else:
                # Rollback
                rollback_swap(current_set, idx, old_val, new_val, sum_counts, prod_counts)
        
        # Local Search Polish (Hill Climbing)
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            for i in range(n):
                old_val = current_set[i]
                # Try a few random swaps for polish
                for _ in range(10):
                    new_val = random.choice(pool)
                    if new_val == old_val or new_val in current_set:
                        continue
                    
                    new_score = apply_swap(current_set, i, old_val, new_val, sum_counts, prod_counts)
                    
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        if new_score < best_score:
                            best_score = new_score
                            best_set = current_set[:]
                            best_sum_counts = sum_counts.copy()
                            best_prod_counts = prod_counts.copy()
                    else:
                        rollback_swap(current_set, i, old_val, new_val, sum_counts, prod_counts)

    return sorted(best_set)

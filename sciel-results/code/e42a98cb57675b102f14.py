import time
import random
import math
import numpy as np

def compute_metrics(A):
    """
    Computes the maximum of the size of the sumset A+A and product set A*A.
    A is a list of distinct positive integers.
    """
    A = np.array(A, dtype=np.int64)
    n = len(A)
    sums = A[:, None] + A[None, :]
    prods = A[:, None] * A[None, :]
    sum_set = len(set(sums.ravel().tolist()))
    prod_set = len(set(prods.ravel().tolist()))
    return max(sum_set, prod_set)

def generate_structured_pool(max_val=10000, primes=(2, 3, 5, 7), max_exp=8):
    """
    Generate numbers that are products of primes^e where e <= max_exp.
    This creates a pool with rich multiplicative structure.
    """
    pool = set()
    def gen(idx, current):
        if idx == len(primes):
            if 1 <= current <= max_val:
                pool.add(current)
            return
        p = primes[idx]
        val = current
        for _ in range(max_exp + 1):
            if val > max_val:
                break
            gen(idx + 1, val)
            val *= p
    gen(0, 1)
    return sorted(pool)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Generate structured pool with multiplicative richness
    pool = generate_structured_pool(10000, (2, 3, 5, 7), max_exp=8)
    pool_set = set(pool)
    
    # Also include some small integers for sumset balance
    small_ints = list(range(1, 51))
    combined_pool = sorted(set(pool) | set(small_ints))
    combined_set = set(combined_pool)
    
    # Multiple diverse initial seeds
    seeds = []
    
    # Seed 1: Small integers (good sumset)
    seeds.append(list(range(1, 21)))
    
    # Seed 2: Powers of 2 and 3 mixed (good product set)
    p23 = [2**a * 3**b for a in range(0, 14) for b in range(0, 14) if 2**a * 3**b <= 10000]
    p23 = sorted(set(p23))
    if len(p23) >= 20:
        # Pick 20 spread across the range
        step = len(p23) // 20
        seed2 = [p23[i * step] for i in range(20)]
        if len(set(seed2)) == 20:
            seeds.append(seed2)
    
    # Seed 3: Smooth numbers from pool, spread out
    if len(pool) >= 20:
        step = len(pool) // 20
        seed3 = [pool[i * step] for i in range(20)]
        if len(set(seed3)) == 20:
            seeds.append(seed3)
    
    # Seed 4: Mix of small and structured
    seed4 = list(range(1, 11))
    remaining_pool = [x for x in pool if x > 10]
    if len(remaining_pool) >= 10:
        step = len(remaining_pool) // 10
        seed4 += [remaining_pool[i * step] for i in range(10)]
        seed4 = sorted(set(seed4))
        if len(seed4) == 20:
            seeds.append(seed4)
    
    # Seed 5: Geometric progression in log space
    try:
        log_min = math.log(2)
        log_max = math.log(10000)
        log_step = (log_max - log_min) / 19
        seed5 = []
        for i in range(20):
            val = int(round(math.exp(log_min + i * log_step)))
            val = max(1, min(10000, val))
            seed5.append(val)
        seed5 = sorted(set(seed5))
        if len(seed5) == 20:
            seeds.append(seed5)
    except:
        pass
    
    # Evaluate all seeds
    best_A = None
    best_score = float('inf')
    
    for seed in seeds:
        if len(set(seed)) != 20 or any(x < 1 or x > 10000 for x in seed):
            continue
        score = compute_metrics(seed)
        if score < best_score:
            best_score = score
            best_A = seed[:]
    
    if best_A is None:
        best_A = list(range(1, 21))
        best_score = compute_metrics(best_A)
    
    # Swap-based local search with structured candidates
    A = best_A[:]
    current_score = best_score
    A_set = set(A)
    
    # Pre-compute neighbor candidates for each position
    # For each element, generate replacement candidates from the structured pool
    candidate_list = combined_pool  # All possible replacements
    
    # Simulated annealing with swap moves
    T = 50.0
    T_min = 0.01
    alpha = 0.9995
    
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline and T > T_min:
        # Choose two indices to swap values
        i = random.randint(0, 19)
        j = random.randint(0, 19)
        if i == j:
            j = (i + 1) % 20
        
        old_vi = A[i]
        old_vj = A[j]
        
        # Generate new candidates for positions i and j
        # Bias towards structured pool numbers
        r = random.random()
        if r < 0.7:
            new_vi = random.choice(candidate_list)
            new_vj = random.choice(candidate_list)
        elif r < 0.85:
            # Neighbor in value space
            delta_i = random.choice([-5, -3, -2, -1, 1, 2, 3, 5])
            delta_j = random.choice([-5, -3, -2, -1, 1, 2, 3, 5])
            new_vi = max(1, min(10000, old_vi + delta_i))
            new_vj = max(1, min(10000, old_vj + delta_j))
        else:
            # Random
            new_vi = random.randint(1, 10000)
            new_vj = random.randint(1, 10000)
        
        # Check distinctness
        if new_vi == new_vj:
            continue
        if new_vi in A_set and new_vi not in (old_vi, old_vj):
            continue
        if new_vj in A_set and new_vj not in (old_vi, old_vj):
            continue
        
        # Apply swap
        A[i] = new_vi
        A[j] = new_vj
        A_set.discard(old_vi)
        A_set.discard(old_vj)
        A_set.add(new_vi)
        A_set.add(new_vj)
        
        new_score = compute_metrics(A)
        
        # Acceptance
        delta = new_score - current_score
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_A = A[:]
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # Revert
            A[i] = old_vi
            A[j] = old_vj
            A_set.discard(new_vi)
            A_set.discard(new_vj)
            A_set.add(old_vi)
            A_set.add(old_vj)
            no_improve_count += 1
        
        T *= alpha
        
        # Restart if stuck too long
        if no_improve_count > max_no_improve:
            no_improve_count = 0
            # Perturb: replace 3 random elements with structured pool values
            for _ in range(3):
                idx = random.randint(0, 19)
                old_val = A[idx]
                new_val = random.choice(candidate_list)
                if new_val != old_val and new_val not in A_set:
                    A[idx] = new_val
                    A_set.discard(old_val)
                    A_set.add(new_val)
            current_score = compute_metrics(A)
            if current_score < best_score:
                best_score = current_score
                best_A = A[:]
        
        # Time check
        if time.time() > deadline - 5.0:
            break
    
    return best_A

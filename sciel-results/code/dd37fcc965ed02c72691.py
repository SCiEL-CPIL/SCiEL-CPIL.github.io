import time
import random
import math
import numpy as np

def compute_metrics(A):
    """
    Computes the maximum of the size of the sumset A+A and product set A*A.
    A is a list of distinct positive integers.
    """
    arr = np.array(A, dtype=np.int64)
    # Compute all pairwise sums
    sums = arr[:, None] + arr[None, :]
    # Compute all pairwise products
    prods = arr[:, None] * arr[None, :]
    
    sum_set_size = len(np.unique(sums))
    prod_set_size = len(np.unique(prods))
    
    return max(sum_set_size, prod_set_size)

def generate_smooth_candidates(limit=10000, max_exponent=10):
    """
    Generate a pool of 'smooth' numbers (composed of small primes) <= limit.
    These numbers are more likely to create product collisions than random large primes.
    """
    candidates = set([1])
    primes = [2, 3, 5, 7, 11, 13]
    
    def generate(idx, current_val):
        if current_val > limit:
            return
        if idx == len(primes):
            candidates.add(current_val)
            return
        
        p = primes[idx]
        # Multiply by p^k
        val = current_val
        for _ in range(max_exponent):
            if val > limit:
                break
            candidates.add(val)
            generate(idx + 1, val)
            val *= p
        # Also skip this prime and go to next
        generate(idx + 1, current_val)

    generate(0, 1)
    # Filter valid candidates
    return [x for x in candidates if 1 <= x <= limit]

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin: 55s out of 60s limit
    
    # 1. Generate a pool of structurally relevant candidates
    # Smooth numbers are better for product set minimization.
    # We also include small integers 1..100 to help with sum set minimization.
    smooth_pool = generate_smooth_candidates(10000, max_exponent=15)
    small_ints = list(range(1, 101))
    
    # Combine and unique
    candidate_pool = list(set(smooth_pool) | set(small_ints))
    random.shuffle(candidate_pool)
    
    # 2. Initial Seeds
    # Seed 1: 1..20 (minimizes sumset)
    seed1 = list(range(1, 21))
    
    # Seed 2: Powers of 2 mixed with 3s (minimizes product set structure)
    # 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 (14 nums)
    # Add 3, 6, 12, 24, 48, 96? 
    # Let's just use a random sample from the smooth pool for a second seed
    random.shuffle(smooth_pool)
    seed2 = []
    used = set()
    for x in smooth_pool:
        if x not in used:
            seed2.append(x)
            used.add(x)
        if len(seed2) == 20:
            break
    
    # Ensure seeds are valid
    if len(set(seed2)) != 20 or any(x > 10000 or x < 1 for x in seed2):
        seed2 = seed1[:] # Fallback
    
    # Seed 3: Mix of small numbers and powers of 2
    seed3 = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 16, 18, 20, 24, 32, 36, 40, 48, 64, 72]
    
    best_A = None
    best_score = float('inf')
    
    for seed in [seed1, seed2, seed3]:
        score = compute_metrics(seed)
        if score < best_score:
            best_score = score
            best_A = seed[:]
            
    # 3. Multi-restart Simulated Annealing
    A = best_A[:]
    current_score = best_score
    
    # Parameters
    T_start = 15.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    num_restarts = 3
    restarts_used = 0
    
    while time.time() < deadline:
        # Check time frequently to avoid overshoot
        if time.time() > deadline - 1.0:
            break
            
        idx = random.randint(0, 19)
        old_val = A[idx]
        
        # Pick a new value from the candidate pool
        # Bias towards values not already in A
        # To speed up, just pick a random candidate and check validity
        new_val = random.choice(candidate_pool)
        
        if new_val == old_val or new_val in A:
            continue
            
        # Ensure bounds (pool is already bounded, but just in case)
        if new_val < 1 or new_val > 10000:
            continue
            
        A[idx] = new_val
        new_score = compute_metrics(A)
        delta = new_score - current_score
        
        # Metropolis criterion
        if delta < 0:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_A = A[:]
        else:
            # Accept with probability exp(-delta / T)
            # delta is integer, T is float
            # If T is small, exp(-large/T) is ~0
            if T_start > 0:
                prob = math.exp(-delta / T_start)
                if random.random() < prob:
                    current_score = new_score
                else:
                    # Reject, revert
                    A[idx] = old_val
            
        # Cool down
        T_start *= cooling_rate
        if T_start < T_min:
            # Restart with new temperature
            restarts_used += 1
            if restarts_used < num_restarts:
                T_start = T_start * 0.5  # Lower initial temp for next restart
                A = best_A[:]
                current_score = best_score
            else:
                T_start = T_min
    
    # 4. Final polish: small perturbations with low temperature
    if time.time() < deadline - 0.5:
        polish_T = 0.1
        polish_deadline = min(deadline - 0.1, time.time() + 5.0)
        
        A = best_A[:]
        current_score = best_score
        
        while time.time() < polish_deadline:
            idx = random.randint(0, 19)
            old_val = A[idx]
            
            # Try small perturbations
            perturbation = random.randint(-5, 5)
            new_val = old_val + perturbation
            
            if new_val == old_val or new_val in A or new_val < 1 or new_val > 10000:
                continue
                
            A[idx] = new_val
            new_score = compute_metrics(A)
            delta = new_score - current_score
            
            if delta < 0:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_A = A[:]
            else:
                prob = math.exp(-delta / polish_T)
                if random.random() < prob:
                    current_score = new_score
                else:
                    A[idx] = old_val
    
    return best_A

# Keep the function name and signature exactly as requested.
# The function above is the implementation.
# Just to be safe, define construct as the entry point clearly.
# The code above defines construct().

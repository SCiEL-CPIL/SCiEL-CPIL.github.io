import time
import random
import math

MAX_N = 20
MAX_VAL = 10000
TIME_LIMIT = 85.0

def _objective(A):
    """Compute max(|A+A|, |A*A|) for a set A of distinct positive integers."""
    A_sorted = sorted(A)
    n = len(A_sorted)
    
    # Sumset with equal summands allowed
    sums = set()
    for i in range(n):
        ai = A_sorted[i]
        for j in range(i, n):
            sums.add(ai + A_sorted[j])
    
    # Product set with equal factors allowed
    products = set()
    for i in range(n):
        ai = A_sorted[i]
        for j in range(i, n):
            products.add(ai * A_sorted[j])
    
    return max(len(sums), len(products))

def _is_valid(A):
    if len(A) != 20:
        return False
    if len(set(A)) != 20:
        return False
    for x in A:
        if not isinstance(x, int) or x < 1 or x > 10000:
            return False
    return True

def _generate_gp(ratio, start_val):
    """Generate a geometric progression set, filling gaps if needed."""
    A = set()
    current = start_val
    while len(A) < 20 and current <= MAX_VAL:
        A.add(current)
        current = int(current * ratio)
        if current <= start_val:
            break
    
    if len(A) < 20:
        # Fill remaining with random small integers to maintain distinctness
        # Try to keep them close to the GP structure or just random
        candidates = list(range(1, MAX_VAL + 1))
        random.shuffle(candidates)
        for v in candidates:
            if len(A) >= 20:
                break
            if v not in A:
                A.add(v)
    return list(A)

def _generate_ap(start_val, step):
    """Generate an arithmetic progression."""
    A = set()
    current = start_val
    while len(A) < 20 and current <= MAX_VAL:
        A.add(current)
        current += step
    if len(A) < 20:
        candidates = list(range(1, MAX_VAL + 1))
        random.shuffle(candidates)
        for v in candidates:
            if len(A) >= 20:
                break
            if v not in A:
                A.add(v)
    return list(A)

def _local_search(A, time_limit, rng, mode='sa'):
    """Local search with simulated annealing."""
    A = sorted(A)
    current_obj = _objective(A)
    best_A = list(A)
    best_obj = current_obj
    
    start_time = time.time()
    
    if mode == 'sa':
        temperature = 10.0
        cooling_rate = 0.99
        min_temperature = 0.1
    else:
        # Hill climbing
        temperature = 0.0
        cooling_rate = 1.0
        min_temperature = 0.0

    while temperature > min_temperature:
        if time.time() - start_time > time_limit:
            break
        
        idx = rng.randint(0, 19)
        old_val = A[idx]
        
        # Generate a new value
        # Bias towards small values and values close to current
        if rng.random() < 0.5:
            # Small random step
            new_val = old_val + rng.randint(-10, 10)
        else:
            # Random value, biased towards low end
            new_val = rng.randint(1, min(MAX_VAL, int(old_val * 2) + 10))
        
        if new_val < 1 or new_val > MAX_VAL:
            continue
        if new_val in A:
            continue
        
        A[idx] = new_val
        new_obj = _objective(A)
        
        delta = new_obj - current_obj
        
        if delta <= 0 or rng.random() < math.exp(-delta / (temperature + 1e-10)):
            current_obj = new_obj
            if new_obj < best_obj:
                best_obj = new_obj
                best_A = list(A)
        else:
            A[idx] = old_val
            
        temperature *= cooling_rate
        
    return best_A, best_obj

def _multi_start_search(time_budget):
    rng = random.Random(42)
    best_A = None
    best_obj = float('inf')
    
    start_time = time.time()
    
    # 1. Start with various GPs
    ratios = [1.1, 1.2, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0]
    starts = [1, 2, 3, 5]
    
    for r in ratios:
        for s in starts:
            if time.time() - start_time > time_budget:
                break
            A = _generate_gp(r, s)
            if _is_valid(A):
                # Give each start a small time slice
                slice_time = max(0.5, (time_budget - (time.time() - start_time)) / 10)
                A_sol, obj = _local_search(A, slice_time, rng)
                if obj < best_obj:
                    best_obj = obj
                    best_A = A_sol
        if time.time() - start_time > time_budget:
            break
            
    # 2. Start with APs
    for step in [1, 2, 3, 5]:
        if time.time() - start_time > time_budget:
            break
        A = _generate_ap(1, step)
        if _is_valid(A):
            slice_time = max(0.5, (time_budget - (time.time() - start_time)) / 10)
            A_sol, obj = _local_search(A, slice_time, rng)
            if obj < best_obj:
                best_obj = obj
                best_A = A_sol
                
    # 3. Random starts
    remaining = time_budget - (time.time() - start_time)
    if remaining > 2:
        num_starts = int(remaining / 2) + 1
        for _ in range(num_starts):
            if time.time() - start_time > time_budget:
                break
            A = random.sample(range(1, 100), 20) # Start small
            if _is_valid(A):
                slice_time = max(0.5, (time_budget - (time.time() - start_time)) / 10)
                A_sol, obj = _local_search(A, slice_time, rng)
                if obj < best_obj:
                    best_obj = obj
                    best_A = A_sol
                    
    if best_A is None:
        best_A = list(range(1, 21))
        best_obj = _objective(best_A)
        
    return best_A, best_obj

def construct():
    start_time = time.time()
    time_budget = TIME_LIMIT - 2.0 # Safety margin
    
    try:
        best_A, best_obj = _multi_start_search(time_budget)
    except Exception:
        best_A = list(range(1, 21))
        best_obj = _objective(best_A)
        
    # Final refinement with remaining time
    remaining = TIME_LIMIT - 2.0 - (time.time() - start_time)
    if remaining > 1:
        try:
            refined_A, refined_obj = _local_search(best_A, remaining, random.Random(99))
            if refined_obj < best_obj:
                best_A = refined_A
        except Exception:
            pass
            
    if not _is_valid(best_A):
        best_A = list(range(1, 21))
        
    return sorted(best_A)

# EVOLVE-BLOCK-END

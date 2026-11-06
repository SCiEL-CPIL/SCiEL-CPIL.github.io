"""Integer sum-product extremal sets - minimize max(|A+A|, |A*A|)"""
import time
import random
import math
import numpy as np

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

def _objective_np(A):
    """Fast objective using numpy for vectorized computation."""
    A_arr = np.array(A, dtype=np.int64)
    n = len(A_arr)
    
    # Sumset: all pairs (i,j) with i <= j
    i_idx, j_idx = np.triu_indices(n, k=0)
    sums = A_arr[i_idx] + A_arr[j_idx]
    n_sums = len(np.unique(sums))
    
    # Product set: all pairs (i,j) with i <= j
    products = A_arr[i_idx] * A_arr[j_idx]
    n_products = len(np.unique(products))
    
    return max(n_sums, n_products)

def _is_valid(A):
    """Check if A has 20 distinct positive integers <= 10000."""
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
    while len(A) < 20 and current <= 10000:
        A.add(current)
        current = int(current * ratio)
        if current <= start_val:
            break
    
    if len(A) < 20:
        # Fill remaining with random small integers to maintain distinctness
        candidates = list(range(1, 10001))
        random.shuffle(candidates)
        for v in candidates:
            if len(A) >= 20:
                break
            if v not in A:
                A.add(v)
    return list(A)

def _generate_rational_gp(p, q, start_val):
    """Generate a geometric progression with rational ratio p/q."""
    A = set()
    current = start_val
    while len(A) < 20 and current <= 10000:
        A.add(current)
        new_val = (current * p) // q
        if new_val <= current:
            break
        current = new_val
    
    if len(A) < 20:
        # Fill remaining with random values
        candidates = list(range(1, 10001))
        random.shuffle(candidates)
        for v in candidates:
            if len(A) >= 20:
                break
            if v not in A:
                A.add(v)
    return list(A)

def _local_search(A, time_limit, rng):
    """Local search with simulated annealing to minimize the objective."""
    A = sorted(A)
    current_obj = _objective_np(A)
    best_A = list(A)
    best_obj = current_obj
    
    start_time = time.time()
    temperature = 100.0
    cooling_rate = 0.995
    min_temperature = 0.01
    A_set = set(A)
    
    while temperature > min_temperature:
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random element to modify
        idx = rng.randint(0, 19)
        old_val = A[idx]
        
        # Try a few random changes
        found_improvement = False
        for _ in range(50):
            if time.time() - start_time > time_limit:
                break
            
            # Random delta
            if rng.random() < 0.3:
                new_val = old_val + rng.randint(-50, 50)
            else:
                new_val = old_val + rng.randint(-200, 200)
            
            if new_val < 1 or new_val > 10000:
                continue
            if new_val in A_set:
                continue
            
            # Test the change
            A[idx] = new_val
            A_set.discard(old_val)
            A_set.add(new_val)
            new_obj = _objective_np(A)
            A[idx] = old_val
            A_set.discard(new_val)
            A_set.add(old_val)
            
            delta = new_obj - current_obj
            if delta <= 0 or rng.random() < min(1.0, max(0.0, -delta / (temperature + 1e-10))):
                A[idx] = new_val
                A_set.discard(old_val)
                A_set.add(new_val)
                current_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_A = list(A)
                found_improvement = True
                break
        
        temperature *= cooling_rate
    
    return best_A, best_obj

def _multi_start_local_search(time_budget):
    """Run multiple local searches from different starting points."""
    rng = random.Random()
    best_A = None
    best_obj = float('inf')
    
    start_time = time.time()
    
    # Start with geometric progressions with various ratios
    for ratio in [1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5]:
        for start_val in [1, 2, 3, 5, 7, 10]:
            if time.time() - start_time > time_budget:
                break
            A = _generate_gp(ratio, start_val)
            if not _is_valid(A):
                continue
            A_sorted, obj = _local_search(A, time_budget - (time.time() - start_time), rng)
            if obj < best_obj:
                best_obj = obj
                best_A = A_sorted
        if time.time() - start_time > time_budget:
            break
    
    # Try rational ratio geometric progressions
    rational_ratios = [(3, 2), (4, 3), (5, 4), (5, 3), (7, 5), (8, 5), (9, 7), (11, 8), (13, 10), (16, 13)]
    for p, q in rational_ratios:
        for start_val in [1, 2, 3, 5]:
            if time.time() - start_time > time_budget:
                break
            A = _generate_rational_gp(p, q, start_val)
            if not _is_valid(A):
                continue
            A_sorted, obj = _local_search(A, time_budget - (time.time() - start_time), rng)
            if obj < best_obj:
                best_obj = obj
                best_A = A_sorted
        if time.time() - start_time > time_budget:
            break
    
    # Try random starts
    remaining = time_budget - (time.time() - start_time)
    if remaining > 2:
        for _ in range(20):
            if time.time() - start_time > time_budget:
                break
            A = random.sample(range(1, 10001), 20)
            if not _is_valid(A):
                continue
            A_sorted, obj = _local_search(A, time_budget - (time.time() - start_time), rng)
            if obj < best_obj:
                best_obj = obj
                best_A = A_sorted
    
    if best_A is None:
        best_A = list(range(1, 21))
        best_obj = _objective(best_A)
    
    return best_A, best_obj

def _grid_search_rational_gp(time_budget):
    """Deterministic grid search over rational ratios p/q and starting values.
    
    For each (p, q, start), generate the sequence:
      a_k = start * floor(p^k / q^k) adjusted to stay within [1, 10000]
    Actually, generate as: a_0 = start, a_{k+1} = round(a_k * p / q)
    Then take first 20 distinct values.
    """
    start_time = time.time()
    best_A = None
    best_obj = float('inf')
    
    # Small integer ratios p/q where p > q
    # Focus on ratios that create multiplicative structure
    p_values = range(2, 15)
    q_values = range(1, 15)
    start_values = range(1, 50)
    
    for p in p_values:
        for q in q_values:
            if p <= q:
                continue
            # Skip if ratio is too large (would exceed 10000 quickly)
            ratio = p / q
            if ratio > 3.0:
                continue
            
            for start_val in start_values:
                if time.time() - start_time > time_budget:
                    return best_A, best_obj
                
                # Generate sequence
                A = []
                current = start_val
                for k in range(20):
                    if current < 1 or current > 10000:
                        break
                    if current not in A:
                        A.append(current)
                    if len(A) >= 20:
                        break
                    current = int(round(current * p / q))
                    if current <= A[-1]:
                        break
                
                if len(A) < 20:
                    # Fill with small integers
                    for v in range(1, 10001):
                        if len(A) >= 20:
                            break
                        if v not in A:
                            A.append(v)
                
                if not _is_valid(A):
                    continue
                
                obj = _objective_np(A)
                if obj < best_obj:
                    best_obj = obj
                    best_A = list(A)
    
    return best_A, best_obj

def construct():
    start_time = time.time()
    total_budget = 80  # leave 10s margin before 90s kill
    
    best_A = None
    best_obj = float('inf')
    
    # Phase 1: Deterministic grid search over rational GP parameters
    phase1_budget = 20
    try:
        gp_A, gp_obj = _grid_search_rational_gp(phase1_budget)
        if gp_A is not None:
            best_A = gp_A
            best_obj = gp_obj
    except Exception:
        pass
    
    # Phase 2: Multi-start local search
    elapsed = time.time() - start_time
    phase2_budget = max(5, total_budget - elapsed - 5)
    try:
        ms_A, ms_obj = _multi_start_local_search(phase2_budget)
        if ms_obj < best_obj:
            best_A = ms_A
            best_obj = ms_obj
    except Exception:
        pass
    
    # Phase 3: Final refinement
    elapsed = time.time() - start_time
    remaining = max(0, total_budget - elapsed)
    if remaining > 1 and best_A is not None:
        try:
            refined_A, refined_obj = _local_search(best_A, remaining, random.Random())
            if refined_obj < best_obj:
                best_A = refined_A
                best_obj = refined_obj
        except Exception:
            pass
    
    # Ensure validity
    if best_A is None or not _is_valid(best_A):
        best_A = list(range(1, 21))
        best_obj = _objective(best_A)
    
    return sorted(best_A)

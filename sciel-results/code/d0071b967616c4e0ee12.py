import time
import random
import numpy as np

MAX_N = 20
MAX_VAL = 10000
TIME_LIMIT = 82  # safety margin before 90s kill

def _objective(A):
    """Compute max(|A+A|, |A*A|) for a set A of distinct positive integers."""
    A_arr = np.array(A, dtype=np.int64)
    # A+A with equal summands allowed (i <= j)
    # We can just compute all pairs and unique them, or use broadcasting
    # For n=20, 190 pairs. Broadcasting is fine.
    sums = A_arr[:, None] + A_arr[None, :]
    # We only need unique values. 
    # Note: A+A is symmetric, but set(sums) handles duplicates.
    # Using set on a flattened numpy array is fast.
    num_sums = len(set(sums.flatten()))
    
    # A*A with equal factors allowed
    products = A_arr[:, None] * A_arr[None, :]
    num_products = len(set(products.flatten()))
    
    return max(num_sums, num_products)

def _is_valid(A):
    """Check if A has 20 distinct positive integers <= 10000."""
    if len(A) != 20:
        return False
    if len(set(A)) != 20:
        return False
    for x in A:
        if not isinstance(x, int) or x < 1 or x > MAX_VAL:
            return False
    return True

def _rational_geometric_search(deadline):
    """
    Search for sets of the form { round(a * (p/q)^i) } for i=0..19.
    We iterate over small p, q and optimize a.
    """
    best_obj = float('inf')
    best_A = None
    rng = random.Random(42)
    
    # Small integers for ratio p/q
    # p/q should be > 1 for increasing sequence (usually optimal for bounds)
    # or < 1 for decreasing. Let's stick to p > q for p/q > 1.
    # If p/q is close to 1, terms are close, sumset is small.
    # If p/q is large, terms spread out, product set might have collisions?
    # Actually, geometric progression has |A*A| = 2n-1 if exact.
    # Integer rounding breaks exactness but preserves structure.
    
    max_pq = 20 # Search up to 20/1, 19/1, etc
    
    for p in range(2, max_pq + 1):
        for q in range(1, p):
            if time.time() > deadline:
                return best_A, best_obj
            
            ratio = p / q
            
            # The sequence is a, a*ratio, a*ratio^2, ...
            # a * ratio^19 <= MAX_VAL
            # a <= MAX_VAL / (ratio^19)
            max_a = int(MAX_VAL / (ratio ** 19))
            if max_a < 1:
                continue
            
            # We want to test a range of 'a' values.
            # Good candidates for 'a' are:
            # 1. Max possible a (packs to the top)
            # 2. Small a (1, 2, 3...)
            # 3. Values that make a * (p/q)^k integers?
            #    If a is multiple of q^19, then all terms are integers.
            #    a = k * q^19.
            #    Let's test a = q^19 * k for small k.
            
            candidates_a = set()
            
            # 1. Boundary
            candidates_a.add(max_a)
            
            # 2. Small integers
            for k in range(1, 10):
                if k <= max_a:
                    candidates_a.add(k)
            
            # 3. Multiples of q^19 to ensure integrality of exact geometric progression
            #    term_i = a * (p/q)^i = (k*q^19) * p^i / q^i = k * q^(19-i) * p^i
            #    This is always an integer.
            #    This creates an EXACT geometric progression in integers.
            #    For an exact GP, |A*A| is exactly 2n-1 = 39 (since a_i a_j = a^2 r^{i+j}).
            #    |A+A| is also structured.
            #    So we should prioritize these!
            
            base = q ** 19
            # a must be <= max_a.
            # max_k = max_a // base
            max_k = max_a // base
            if max_k > 0:
                for k in range(1, min(max_k, 20) + 1): # Test small k
                    candidates_a.add(k * base)
            
            # 4. Random samples in [1, max_a]
            for _ in range(5):
                candidates_a.add(rng.randint(1, max_a))
            
            # Evaluate candidates
            for a in candidates_a:
                if a < 1:
                    continue
                
                A = []
                valid = True
                current = a
                for i in range(20):
                    if current > MAX_VAL:
                        valid = False
                        break
                    if current in A:
                        valid = False
                        break
                    A.append(current)
                    # Update current for next term: a * (p/q)^(i+1)
                    # current = a * (p/q)^i
                    # next = current * p / q
                    # To avoid floating point errors, use integer arithmetic if possible
                    # But p/q might not divide current.
                    # Let's use exact rational arithmetic for the next step?
                    # Or just float and round?
                    # If we use a = k * q^19, then current is always integer.
                    # next = current * p // q. Since current is divisible by q (for i<19), this is exact.
                    # For i=19, we don't need next.
                    
                    # Check if current is divisible by q to ensure exactness for next step
                    # Actually, if a is multiple of q^19, a*q^(19-i) is divisible by q for i<19.
                    # So next = (current * p) // q is exact.
                    
                    if i < 19:
                        if current % q != 0:
                            # If not exact, we might have rounding issues if we used float.
                            # But here we are using integer logic for the "exact" candidates.
                            # For other candidates (random/small), we should use float rounding?
                            # The task allows any integers.
                            # If we use float rounding, we might get duplicates or non-integers.
                            # Let's stick to the strategy:
                            # If a is multiple of q^19, use exact integer arithmetic.
                            # Else, use float rounding and check validity.
                            pass 
                        
                        next_val = (current * p) // q
                        # If current * p is not divisible by q, this truncates.
                        # Is that okay? It generates a sequence of integers.
                        # It might not be strictly geometric in reals, but close.
                        current = next_val
                
                if not valid or len(A) != 20:
                    continue
                
                # Check distinctness again just in case (integer arithmetic might collapse)
                if len(set(A)) != 20:
                    continue
                
                obj = _objective(A)
                if obj < best_obj:
                    best_obj = obj
                    best_A = list(A)
                
                if time.time() > deadline:
                    return best_A, best_obj

    return best_A, best_obj

def _local_search(A, time_limit, rng):
    """Local search with simulated annealing to minimize the objective."""
    A = list(A)
    current_obj = _objective(A)
    best_A = list(A)
    best_obj = current_obj
    
    start_time = time.time()
    temperature = 100.0
    cooling_rate = 0.995
    min_temperature = 0.01
    
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
            
            if new_val < 1 or new_val > MAX_VAL:
                continue
            if new_val in A:
                continue
            
            # Test the change
            A[idx] = new_val
            new_obj = _objective(A)
            A[idx] = old_val
            
            delta = new_obj - current_obj
            if delta <= 0 or rng.random() < min(1.0, max(0.0, -delta / (temperature + 1e-10))):
                A[idx] = new_val
                current_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_A = list(A)
                found_improvement = True
                break
        
        temperature *= cooling_rate
    
    return best_A, best_obj

def _model_and_optimize(time_budget):
    """
    Model-and-optimize approach:
    1. Use structural insight: geometric sequences have small sum/product sets
    2. Parameterize by (start, ratio) and locally optimize
    3. Use multi-start with different initial parameterizations
    """
    rng = random.Random(42)
    best_A = None
    best_obj = float('inf')
    
    start_time = time.time()
    
    # Phase 1: Explore geometric parameter space
    # For a geometric sequence with ratio r, |A+A| and |A*A| are both O(n^2)
    # but with structure that can reduce collisions
    
    # Try various geometric ratios and starting values
    ratios = [1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0, 2.5, 3.0]
    starts = [1, 2, 3, 5, 8, 10, 15, 20]
    
    for ratio in ratios:
        if time.time() - start_time > time_budget:
            break
        for start in starts:
            if time.time() - start_time > time_budget:
                break
            A = _geometric_start(ratio, start)
            if not _is_valid(A):
                continue
            A_sorted, obj = _local_search(A, time_budget - (time.time() - start_time), rng)
            if obj < best_obj:
                best_obj = obj
                best_A = A_sorted
    
    # Phase 2: Refine with random perturbations around best
    if best_A is not None and time.time() - start_time < time_budget:
        remaining = time_budget - (time.time() - start_time)
        for _ in range(5):
            if remaining < 2:
                break
            # Perturb the best solution
            A = list(best_A)
            idx = rng.randint(0, 19)
            delta = rng.randint(-100, 100)
            new_val = A[idx] + delta
            if 1 <= new_val <= MAX_VAL and new_val not in A:
                A[idx] = new_val
            A_sorted, obj = _local_search(A, remaining - 2, rng)
            if obj < best_obj:
                best_obj = obj
                best_A = A_sorted
            remaining = time_budget - (time.time() - start_time)
    
    if best_A is None:
        best_A = list(range(1, 21))
        best_obj = _objective(best_A)
    
    return best_A, best_obj

def _geometric_start(ratio, start_val):
    """Generate a geometric-like starting set."""
    A = set()
    current = start_val
    while len(A) < 20 and current <= MAX_VAL:
        A.add(current)
        current = int(current * ratio)
        if current <= start_val:
            break
    if len(A) < 20:
        while len(A) < 20:
            v = random.randint(1, MAX_VAL)
            A.add(v)
    return list(A)

def construct():
    start_time = time.time()
    time_budget = TIME_LIMIT - 5  # leave margin
    
    # Phase 1: Deterministic rational geometric search
    # This exploits multiplicative collisions to minimize product set size
    deadline1 = start_time + (time_budget * 0.6)
    best_A, best_obj = _rational_geometric_search(deadline1)
    
    # Phase 2: Local search refinement
    if best_A is None:
        best_A = list(range(1, 21))
        best_obj = _objective(best_A)
    
    remaining = TIME_LIMIT - 5 - (time.time() - start_time)
    if remaining > 1:
        try:
            refined_A, refined_obj = _local_search(best_A, remaining, random.Random())
            if refined_obj < best_obj:
                best_A = refined_A
                best_obj = refined_obj
        except Exception:
            pass
    
    # Ensure validity
    if not _is_valid(best_A):
        best_A = list(range(1, 21))
    
    return sorted(best_A)

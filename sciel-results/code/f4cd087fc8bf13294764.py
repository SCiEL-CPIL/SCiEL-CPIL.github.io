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
    # Using np.unique is faster than set() for numpy arrays.
    num_sums = len(np.unique(sums))
    
    # A*A with equal factors allowed
    products = A_arr[:, None] * A_arr[None, :]
    num_products = len(np.unique(products))
    
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
            
            # Random delta - use smaller steps for better convergence
            if rng.random() < 0.4:
                new_val = old_val + rng.randint(-30, 30)
            elif rng.random() < 0.7:
                new_val = old_val + rng.randint(-100, 100)
            else:
                new_val = old_val + rng.randint(-300, 300)
            
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

def _polish_pass(A, time_limit, rng):
    """Final polish with very small steps to find local optima."""
    A = list(A)
    current_obj = _objective(A)
    best_A = list(A)
    best_obj = current_obj
    
    start_time = time.time()
    
    # Use much smaller steps for polishing
    while time.time() - start_time < time_limit:
        # Try all elements with small perturbations
        for idx in range(20):
            if time.time() - start_time > time_limit:
                break
            
            old_val = A[idx]
            
            # Try small deltas
            for delta in [-5, -3, -2, -1, 1, 2, 3, 5]:
                new_val = old_val + delta
                if new_val < 1 or new_val > MAX_VAL:
                    continue
                if new_val in A:
                    continue
                
                A[idx] = new_val
                new_obj = _objective(A)
                A[idx] = old_val
                
                if new_obj < current_obj:
                    A[idx] = new_val
                    current_obj = new_obj
                    if new_obj < best_obj:
                        best_obj = new_obj
                        best_A = list(A)
                    break
    
    return best_A, best_obj

def construct():
    start_time = time.time()
    time_budget = TIME_LIMIT - 5  # leave margin
    deadline = start_time + time_budget
    
    best_A = None
    best_obj = float('inf')
    
    # Phase 1: Deterministic Rational Geometric Search
    # This is the structural change requested.
    try:
        A, obj = _rational_geometric_search(deadline)
        if A:
            best_A = A
            best_obj = obj
    except Exception:
        pass
    
    # If we haven't found anything, fallback to simple geometric
    if best_A is None:
        A = [2**i for i in range(20)]
        # 2^19 = 524288 > 10000. So this is invalid.
        # Use smaller ratio.
        A = [1] * 20 # Invalid distinct
        # Just use a valid fallback
        best_A = list(range(1, 21))
        best_obj = _objective(best_A)
        
    # Phase 2: Local Search Refinement
    # Use remaining time to polish the best candidate found.
    if time.time() < deadline:
        remaining = deadline - time.time()
        if remaining > 1.0:
            try:
                # First do broader local search
                refined_A, refined_obj = _local_search(best_A, remaining * 0.7, random.Random(123))
                if refined_obj < best_obj:
                    best_A = refined_A
                    best_obj = refined_obj
                
                # Then do a polish pass with smaller steps
                remaining2 = deadline - time.time()
                if remaining2 > 0.5:
                    polished_A, polished_obj = _polish_pass(best_A, remaining2, random.Random(456))
                    if polished_obj < best_obj:
                        best_A = polished_A
                        best_obj = polished_obj
            except Exception:
                pass
    
    # Ensure validity
    if not _is_valid(best_A):
        best_A = list(range(1, 21))
    
    return sorted(best_A)

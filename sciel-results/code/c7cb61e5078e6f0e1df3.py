import time
import random
import numpy as np
from itertools import combinations

MAX_N = 20
MAX_VAL = 10000
TIME_LIMIT = 85  # Safety margin

def _objective_fast(A, cached_indices=None):
    """Compute max(|A+A|, |A*A|) for a set A of distinct positive integers using NumPy."""
    A_arr = np.array(A, dtype=np.int64)
    n = len(A_arr)
    
    if cached_indices is None:
        indices = np.triu_indices(n)
    else:
        indices = cached_indices
    
    # Vectorized sum and product calculation
    sums = A_arr[indices[0]] + A_arr[indices[1]]
    products = A_arr[indices[0]] * A_arr[indices[1]]
    
    # Count unique values
    unique_sums = len(np.unique(sums))
    unique_products = len(np.unique(products))
    
    return max(unique_sums, unique_products)

def _objective(A):
    """Compute max(|A+A|, |A*A|) for a set A of distinct positive integers using NumPy."""
    return _objective_fast(A)

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

def _generate_rational_gp(start, num, den, n):
    """
    Generate n distinct integers from a rational geometric progression:
    a_k = start * (num/den)^k
    This helps create multiplicative collisions.
    """
    A = []
    current = float(start)
    ratio = float(num) / float(den)
    
    for _ in range(n):
        val = int(round(current))
        if val < 1:
            val = 1
        if val > MAX_VAL:
            break
        if val not in A:
            A.append(val)
        current *= ratio
        
    # If we didn't get enough, fill with small increments
    if len(A) < n:
        last = A[-1] if A else start
        while len(A) < n:
            last += 1
            if last > MAX_VAL: break
            if last not in A:
                A.append(last)
            
    return A[:n]

def _generate_multiplicative_structure(base, exponents, n):
    """
    Generate a set with strong multiplicative structure by using
    products of base^e_i where e_i are chosen exponents.
    This creates many multiplicative collisions.
    """
    A = set()
    for e in exponents:
        val = base ** e
        if 1 <= val <= MAX_VAL:
            A.add(val)
    
    A = sorted(A)
    if len(A) < n:
        # Fill with additional powers or multiples
        last = A[-1] if A else base
        while len(A) < n:
            last += 1
            if last > MAX_VAL:
                break
            if last not in A:
                A.append(last)
    
    return A[:n]

def _generate_composite_structure(n):
    """
    Generate a set with elements that are products of small primes,
    creating multiplicative structure.
    """
    # Use products of small primes to create multiplicative collisions
    small_primes = [2, 3, 5, 7]
    A = set()
    
    # Generate products of up to 3 small primes
    for p1 in small_primes:
        for p2 in small_primes:
            for p3 in small_primes:
                val = p1 * p2 * p3
                if 1 <= val <= MAX_VAL:
                    A.add(val)
    
    A = sorted(A)
    if len(A) < n:
        # Add more structured values
        for val in range(1, MAX_VAL + 1):
            if val not in A:
                A.add(val)
            if len(A) >= n:
                break
    
    return sorted(A)[:n]

def _generate_two_cluster_gp(n, base1, base2, ratio1, ratio2, rng):
    """
    Generate a set as union of two geometric progressions with different bases.
    Cluster 1: base1 * ratio1^k for k=0..n1-1
    Cluster 2: base2 * ratio2^k for k=0..n2-1
    This creates multiplicative collisions within each cluster.
    """
    n1 = n // 2
    n2 = n - n1
    
    cluster1 = []
    current = float(base1)
    for _ in range(n1):
        val = int(round(current))
        if val < 1:
            val = 1
        if val > MAX_VAL:
            break
        if val not in cluster1:
            cluster1.append(val)
        current *= ratio1
    
    cluster2 = []
    current = float(base2)
    for _ in range(n2):
        val = int(round(current))
        if val < 1:
            val = 1
        if val > MAX_VAL:
            break
        if val not in cluster2:
            cluster2.append(val)
        current *= ratio2
    
    A = sorted(set(cluster1 + cluster2))
    
    # Fill if needed
    if len(A) < n:
        last = max(A) if A else 1
        while len(A) < n:
            last += 1
            if last > MAX_VAL:
                break
            if last not in A:
                A.append(last)
    
    return sorted(A)[:n]

def _local_search(A, time_limit, rng):
    """Local search with simulated annealing to minimize the objective."""
    A = sorted(A)
    current_obj = _objective(A)
    best_A = list(A)
    best_obj = current_obj
    
    start_time = time.time()
    temperature = 50.0
    cooling_rate = 0.998
    cached_indices = np.triu_indices(20)
    
    while temperature > 0.05:
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random element to modify
        idx = rng.randint(0, 19)
        old_val = A[idx]
        
        # Try a few random changes with structured mutations
        accepted = False
        for attempt in range(15):
            if time.time() - start_time > time_limit:
                break
            
            # Strategy: Mix of small perturbations, multiplicative moves, and structured jumps
            r = rng.random()
            if r < 0.3:
                # Small perturbation
                scale = max(1, old_val // 20)
                new_val = old_val + rng.randint(-scale, scale)
            elif r < 0.5:
                # Multiplicative move: multiply or divide by small factor
                factor = rng.choice([2, 3, 4, 5])
                if rng.random() < 0.5:
                    new_val = old_val * factor
                else:
                    new_val = old_val // factor if old_val % factor == 0 else old_val // factor + 1
            elif r < 0.7:
                # Structured jump to a value with multiplicative relationship
                # Try to find a value that shares factors with other elements
                candidates = []
                for other in A:
                    if other == old_val:
                        continue
                    # Try multiples and divisors
                    for mult in [2, 3]:
                        cand = other * mult
                        if 1 <= cand <= MAX_VAL and cand != old_val:
                            candidates.append(cand)
                    if other > 1:
                        cand = other // 2
                        if 1 <= cand <= MAX_VAL and cand != old_val:
                            candidates.append(cand)
                
                if candidates:
                    new_val = rng.choice(candidates[:20])
                else:
                    new_val = rng.randint(1, MAX_VAL)
            else:
                # Larger random jump
                new_val = rng.randint(1, MAX_VAL)
            
            if new_val < 1 or new_val > MAX_VAL:
                continue
            if new_val in A:
                continue
            
            # Test the change
            A[idx] = new_val
            new_obj = _objective_fast(A, cached_indices)
            A[idx] = old_val
            
            delta = new_obj - current_obj
            if delta <= 0 or rng.random() < min(1.0, max(0.0, -delta / (temperature + 1e-10))):
                A[idx] = new_val
                current_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_A = list(A)
                accepted = True
                break
        
        temperature *= cooling_rate
    
    return best_A, best_obj

def _polish_pass(A, time_limit, rng):
    """
    High-precision polish: try all single-element replacements and swaps
    to find local optima with exact evaluation.
    """
    A = sorted(A)
    current_obj = _objective(A)
    best_A = list(A)
    best_obj = current_obj
    cached_indices = np.triu_indices(20)
    start_time = time.time()
    
    # Phase 1: Try all single replacements
    for i in range(20):
        if time.time() - start_time > time_limit:
            break
        old_val = A[i]
        # Try nearby values and structured values
        candidates = set()
        # Nearby values
        for delta in range(-50, 51):
            cand = old_val + delta
            if 1 <= cand <= MAX_VAL and cand != old_val:
                candidates.add(cand)
        # Multiplicative relationships
        for other in A:
            if other == old_val:
                continue
            for mult in [2, 3, 4, 5]:
                cand = other * mult
                if 1 <= cand <= MAX_VAL and cand != old_val:
                    candidates.add(cand)
            if other > 1:
                cand = other // 2
                if 1 <= cand <= MAX_VAL and cand != old_val:
                    candidates.add(cand)
        
        for new_val in candidates:
            if time.time() - start_time > time_limit:
                break
            if new_val in A:
                continue
            A[i] = new_val
            new_obj = _objective_fast(A, cached_indices)
            if new_obj < best_obj:
                best_obj = new_obj
                best_A = list(A)
            A[i] = old_val
    
    # Phase 2: Try all pairwise swaps
    for i in range(20):
        if time.time() - start_time > time_limit:
            break
        for j in range(i + 1, 20):
            if time.time() - start_time > time_limit:
                break
            # Try swapping A[i] and A[j] with nearby values
            for delta_i in range(-20, 21, 5):
                for delta_j in range(-20, 21, 5):
                    if time.time() - start_time > time_limit:
                        break
                    new_i = A[i] + delta_i
                    new_j = A[j] + delta_j
                    if new_i < 1 or new_i > MAX_VAL or new_j < 1 or new_j > MAX_VAL:
                        continue
                    if new_i == new_j:
                        continue
                    if new_i in A or new_j in A:
                        continue
                    A[i] = new_i
                    A[j] = new_j
                    new_obj = _objective_fast(A, cached_indices)
                    if new_obj < best_obj:
                        best_obj = new_obj
                        best_A = list(A)
                    A[i] = A[i]  # restore
                    A[j] = A[j]
    
    return best_A, best_obj

def construct():
    start_time = time.time()
    time_budget = TIME_LIMIT - 5  # Leave margin for safety
    
    rng = random.Random(42)
    best_A = None
    best_obj = float('inf')
    
    starts = []
    
    # Start 1: Powers of 2 (strong multiplicative structure)
    starts.append([2**i for i in range(20)])
    
    # Start 2: Mixed powers - 2^i for i=0..9, 3^i for i=0..9
    mixed_powers = sorted(set([2**i for i in range(10)] + [3**i for i in range(10)]))
    if len(mixed_powers) >= 20:
        starts.append(mixed_powers[:20])
    else:
        # Fill with more powers
        for i in range(10, 20):
            mixed_powers.append(2**i)
        starts.append(sorted(mixed_powers)[:20])
    
    # Start 3: Rational GP (1, 3/2)
    starts.append(_generate_rational_gp(1, 3, 2, 20))
    
    # Start 4: Rational GP (1, 4/3)
    starts.append(_generate_rational_gp(1, 4, 3, 20))
    
    # Start 5: Multiplicative structure with small primes
    starts.append(_generate_composite_structure(20))
    
    # Start 6: Powers of 6 (2*3) for more multiplicative collisions
    starts.append([6**i for i in range(10)] + [6**i * 2 for i in range(10)])
    starts[5] = sorted(set(starts[5]))[:20]
    while len(starts[5]) < 20:
        starts[5].append(starts[5][-1] + 1)
    starts[5] = sorted(starts[5])[:20]
    
    # Start 7: Random
    starts.append(sorted(rng.sample(range(1, MAX_VAL + 1), 20)))
    
    # Start 8: Two-cluster GP with bases 2 and 3
    starts.append(_generate_two_cluster_gp(20, 1, 2, 2.0, 3.0, rng))
    
    # Start 9: Two-cluster GP with bases 1 and 4
    starts.append(_generate_two_cluster_gp(20, 1, 4, 2.0, 4.0, rng))
    
    # Start 10: Two-cluster GP with bases 2 and 5
    starts.append(_generate_two_cluster_gp(20, 2, 5, 2.0, 5.0, rng))

    # Run local search on each start with adaptive time allocation
    for i, A in enumerate(starts):
        if not _is_valid(A):
            # Fix if invalid
            A = sorted(set(A))
            if len(A) < 20:
                available = [x for x in range(1, MAX_VAL+1) if x not in A]
                A += rng.sample(available, 20 - len(A))
            elif len(A) > 20:
                A = A[:20]
            A = sorted(A)
        
        remaining_time = time_budget - (time.time() - start_time)
        if remaining_time < 0.5:
            break
        
        # Adaptive time allocation: give more time to promising starts
        # First starts get more time, later starts get less
        if i == 0:
            alloc = remaining_time * 0.25
        elif i == 1:
            alloc = remaining_time * 0.2
        elif i == 2:
            alloc = remaining_time * 0.15
        else:
            alloc = max(0.5, remaining_time / (len(starts) - i + 1))
        
        A_opt, obj_opt = _local_search(A, alloc, rng)
        
        if obj_opt < best_obj:
            best_obj = obj_opt
            best_A = A_opt
    
    # Final refinement on the best found
    remaining = time_budget - (time.time() - start_time)
    if remaining > 0.3 and best_A:
        # First do a polish pass with exact evaluation
        A_polished, obj_polished = _polish_pass(best_A, remaining * 0.6, rng)
        if obj_polished < best_obj:
            best_A = A_polished
            best_obj = obj_polished
        
        # Then do a final local search
        remaining2 = time_budget - (time.time() - start_time)
        if remaining2 > 0.3:
            A_ref, obj_ref = _local_search(best_A, remaining2, rng)
            if obj_ref < best_obj:
                best_A = A_ref
                best_obj = obj_ref
    
    if not best_A:
        best_A = list(range(1, 21))
        
    return sorted(best_A)

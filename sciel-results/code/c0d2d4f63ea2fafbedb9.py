import time
import random
from math import gcd

# EVOLVE-BLOCK-START

def _prime_factor_vector(n):
    """Return the exponent vector of n in the prime basis (primes up to 40)."""
    # Primes up to 40: 2,3,5,7,11,13,17,19,23,29,31,37
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    vec = [0] * 12
    for i, p in enumerate(primes):
        while n % p == 0:
            vec[i] += 1
            n //= p
    return tuple(vec)

def _is_linearly_independent(vecs):
    """Check if a list of integer vectors is linearly independent over Q.
    Uses Gaussian elimination with fractions to avoid floating point issues."""
    if not vecs:
        return True
    n = len(vecs)
    dim = len(vecs[0])
    # Build matrix
    matrix = [list(v) for v in vecs]
    rank = 0
    for col in range(dim):
        # Find pivot
        pivot = -1
        for row in range(rank, n):
            if matrix[row][col] != 0:
                pivot = row
                break
        if pivot == -1:
            continue
        # Swap
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        # Eliminate
        pivot_val = matrix[rank][col]
        for row in range(rank + 1, n):
            factor = matrix[row][col] / pivot_val
            for c in range(col, dim):
                matrix[row][c] -= factor * matrix[rank][c]
        rank += 1
        if rank == n:
            return True
    return rank == n

def _can_add_vector(current_vecs, new_vec):
    """Check if adding new_vec keeps the set linearly independent."""
    return _is_linearly_independent(current_vecs + [new_vec])

def _greedy_build(candidates, vectors, deadline, rng):
    """
    Greedy build using vector-based independence check.
    Returns (result_list, vecs_list).
    """
    result = []
    current_vecs = []
    for c, v in candidates:
        if time.time() > deadline:
            break
        if _can_add_vector(current_vecs, v):
            result.append(c)
            current_vecs.append(v)
    return result, current_vecs

def _local_search(result, current_vecs, all_candidates, all_vectors, deadline, rng, max_swaps=300):
    """
    Try to improve the set by swapping elements.
    Returns (best_result, best_vecs, best_size).
    """
    best_size = len(result)
    best_result = list(result)
    best_vecs = list(current_vecs)
    
    result_set = set(result)
    
    for _ in range(max_swaps):
        if time.time() > deadline:
            break
        
        if not result:
            break
        
        out_idx = rng.randrange(len(result))
        out_elem = result[out_idx]
        
        # Find candidates not in result
        available = [(c, v) for c, v in all_candidates if c not in result_set]
        if not available:
            break
        
        in_elem, in_vec = rng.choice(available)
        
        # Build new set
        new_result = result[:out_idx] + result[out_idx+1:] + [in_elem]
        new_vecs = current_vecs[:out_idx] + current_vecs[out_idx+1:] + [in_vec]
        
        if _is_linearly_independent(new_vecs) and len(new_result) > best_size:
            best_size = len(new_result)
            best_result = list(new_result)
            best_vecs = list(new_vecs)
            result = list(new_result)
            current_vecs = list(new_vecs)
            result_set = set(result)
    
    return best_result, best_vecs, best_size

def _randomized_greedy(all_candidates, all_vectors, deadline, rng, n_restarts=100):
    """
    Multiple randomized greedy restarts.
    """
    best_result = []
    best_vecs = []
    best_size = 0
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        shuffled = list(zip(all_candidates, all_vectors))
        rng.shuffle(shuffled)
        
        result, vecs = _greedy_build(shuffled, all_vectors, deadline, rng)
        
        if len(result) > best_size:
            best_size = len(result)
            best_result = list(result)
            best_vecs = list(vecs)
    
    # Local search on the best result
    if best_size > 0:
        best_result, best_vecs, best_size = _local_search(
            best_result, best_vecs, all_candidates, all_vectors, deadline, rng
        )
    
    return best_result, best_vecs

def _branch_and_bound(candidates, vectors, current_set, current_vecs, deadline, best_size, rng):
    """
    Branch and bound to find maximum independent set in the compatibility graph.
    candidates: list of (value, vector) tuples to consider
    current_set: list of values in current clique
    current_vecs: list of vectors in current clique
    Returns (best_set, best_vecs, best_size)
    """
    if time.time() > deadline:
        return current_set, current_vecs, len(current_set)
    
    if len(candidates) == 0:
        return current_set, current_vecs, len(current_set)
    
    # Upper bound: current size + remaining candidates
    upper_bound = len(current_set) + len(candidates)
    if upper_bound <= best_size:
        return current_set, current_vecs, best_size
    
    # Try adding the first candidate
    val, vec = candidates[0]
    remaining = candidates[1:]
    
    # Option 1: Include val
    if _can_add_vector(current_vecs, vec):
        new_set = current_set + [val]
        new_vecs = current_vecs + [vec]
        # Filter remaining to only those compatible with new_set
        filtered = []
        for cv, vv in remaining:
            if _can_add_vector(new_vecs, vv):
                filtered.append((cv, vv))
        result_set, result_vecs, result_size = _branch_and_bound(
            filtered, vectors, new_set, new_vecs, deadline, best_size, rng
        )
        if result_size > best_size:
            best_size = result_size
            current_set = result_set
            current_vecs = result_vecs
    
    # Option 2: Exclude val
    result_set, result_vecs, result_size = _branch_and_bound(
        remaining, vectors, current_set, current_vecs, deadline, best_size, rng
    )
    if result_size > best_size:
        best_size = result_size
        current_set = result_set
        current_vecs = result_vecs
    
    return current_set, current_vecs, best_size

def construct():
    """
    Returns a list of distinct integers in the range 2..40 such that
    all 2^k subset products are distinct. Maximizes k.
    """
    t0 = time.time()
    deadline = t0 + 55.0  # 55 second budget (5s margin before 60s kill)
    
    rng = random.Random(42)
    
    all_values = list(range(2, 41))
    all_vectors = [_prime_factor_vector(n) for n in all_values]
    all_candidates = list(zip(all_values, all_vectors))
    
    # Phase 1: Fast deterministic greedy with sorted order
    def count_prime_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            if n % d == 0:
                count += 1
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            count += 1
        return count
    
    sorted_indices = sorted(range(40), key=lambda i: count_prime_factors(all_values[i]))
    sorted_candidates = [(all_values[i], all_vectors[i]) for i in sorted_indices]
    base_result, base_vecs = _greedy_build(sorted_candidates, all_vectors, deadline, rng)
    
    # Phase 2: Randomized greedy with multiple restarts
    rand_result, rand_vecs = _randomized_greedy(all_values, all_vectors, deadline, rng, n_restarts=150)
    
    # Pick the better one
    if len(rand_result) > len(base_result):
        result = rand_result
        current_vecs = rand_vecs
    else:
        result = base_result
        current_vecs = base_vecs
    
    # Phase 3: More local search iterations on the best result
    if time.time() < deadline:
        result, current_vecs, _ = _local_search(
            result, current_vecs, all_values, all_vectors, deadline, rng, max_swaps=500
        )
    
    # Phase 4: Branch and bound on remaining candidates
    if time.time() < deadline - 5:
        current_set = set(result)
        remaining = [(v, vec) for v, vec in all_candidates if v not in current_set]
        # Sort remaining by some heuristic (fewer prime factors first)
        remaining.sort(key=lambda x: sum(1 for e in x[1] if e > 0))
        
        result, current_vecs, _ = _branch_and_bound(
            remaining, all_vectors, list(result), list(current_vecs), deadline, len(result), rng
        )
    
    # Ensure all elements are Python int
    return [int(x) for x in result]

# EVOLVE-BLOCK-END

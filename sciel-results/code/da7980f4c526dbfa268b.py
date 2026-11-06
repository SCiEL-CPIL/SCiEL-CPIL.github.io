import time
import random
import math

# Precompute prime factors for all numbers up to 20000
MAX_SUM = 20000
_prime_factors_cache = [set() for _ in range(MAX_SUM + 1)]

def _init_factors():
    """Initialize prime factor cache using sieve-like approach."""
    for i in range(2, MAX_SUM + 1):
        if not _prime_factors_cache[i]:
            # i is prime
            for j in range(i, MAX_SUM + 1, i):
                _prime_factors_cache[j].add(i)

def _is_smooth(n, prime_set):
    """Check if all prime factors of n are in prime_set."""
    # n is sum of two distinct positive integers, so n >= 1+2=3
    if n > MAX_SUM:
        return False
    return _prime_factors_cache[n].issubset(prime_set)

def _find_smooth_clique(prime_set, target_size, time_limit):
    """
    Find a clique of size target_size in the graph where edges connect
    numbers u, v if u+v is smooth w.r.t prime_set.
    Uses backtracking with time limit.
    """
    start_time = time.time()
    # Candidates: 1 to 10000
    # To speed up, we can restrict to numbers that have at least one smooth sum with another number?
    # Or just start with a random subset to reduce branching factor?
    # Let's try with a random permutation of 1..10000, but prune heavily.
    
    # Optimization: Sort candidates by "smoothness potential"? 
    # For now, use random order to avoid bias, but check time often.
    
    candidates = list(range(1, 10001))
    random.shuffle(candidates)
    
    # To make backtracking more effective, we can limit the depth of search
    # or use a heuristic: pick the candidate that is compatible with the most
    # already selected nodes.
    
    def backtrack(selected, remaining):
        if len(selected) == target_size:
            return selected[:]
        
        if time.time() - start_time > time_limit:
            return None
            
        # Heuristic: Sort remaining by compatibility with 'selected'
        # This is expensive. Let's just iterate.
        # But we must ensure 'remaining' only contains nodes compatible with ALL in 'selected'.
        
        next_remaining = []
        for v in remaining:
            if time.time() - start_time > time_limit:
                return None
            # Check if v is compatible with all in selected
            is_compat = True
            for u in selected:
                if not _is_smooth(u + v, prime_set):
                    is_compat = False
                    break
            if is_compat:
                next_remaining.append(v)
                
        if not next_remaining:
            return None
            
        # Try each candidate in next_remaining
        # Randomize order to explore different branches
        random.shuffle(next_remaining)
        
        for v in next_remaining:
            # Create new remaining list: next_remaining without v, 
            # but also filtered by compatibility with v? 
            # Actually, the next recursion will filter again.
            # To be efficient, we can filter here.
            new_remaining = []
            for w in next_remaining:
                if w == v:
                    continue
                # Check if w is compatible with v (and implicitly with selected, 
                # since w was in next_remaining)
                if _is_smooth(v + w, prime_set):
                    new_remaining.append(w)
                    
            res = backtrack(selected + [v], new_remaining)
            if res is not None:
                return res
        return None

    # Start with empty selected, all candidates remaining
    # To reduce initial branching, we can pick a random start node
    # or just pass all candidates. 
    # Passing 10000 candidates is too many for the first level.
    # Let's pick a random subset of 200 candidates to start the search.
    # If it fails, try another subset.
    
    attempts = 0
    while time.time() - start_time < time_limit:
        attempts += 1
        # Pick a random subset of candidates to start
        # Smaller subsets are faster to process but might miss the solution.
        # Let's try 100 candidates per attempt.
        subset = random.sample(candidates, min(100, len(candidates)))
        res = backtrack([], subset)
        if res is not None:
            return res
        if attempts > 50: # Don't loop forever if it's hard
            break
            
    return None

def _objective(arr):
    """Return the number of distinct prime factors in all pairwise sums."""
    primes = set()
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            s = arr[i] + arr[j]
            if s <= MAX_SUM:
                primes |= _prime_factors_cache[s]
            else:
                # Fallback factorization (should not happen if arr <= 10000)
                # But just in case
                temp = s
                d = 2
                while d * d <= temp:
                    if temp % d == 0:
                        primes.add(d)
                        while temp % d == 0:
                            temp //= d
                    d += 1
                if temp > 1:
                    primes.add(temp)
    return len(primes)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    _init_factors()
    
    n = 12
    max_val = 10000
    
    best_arr = None
    best_score = float('inf')
    
    # Phase 1: Targeted search for small prime sets
    # Try sets of first k primes
    prime_candidates = [
        {2, 3, 5},
        {2, 3, 5, 7},
        {2, 3, 5, 7, 11},
        {2, 3, 5, 7, 11, 13},
    ]
    
    for p_set in prime_candidates:
        if time.time() > deadline - 10:
            break
        # Allocate 5 seconds for each prime set search
        res = _find_smooth_clique(p_set, n, 5.0)
        if res is not None:
            best_arr = res
            best_score = len(p_set)
            # If we found a solution with 3 primes, it's likely optimal.
            # We can return immediately if score is very low.
            if best_score <= 5:
                return best_arr
    
    # Phase 2: Iterated Local Search (ILS)
    if best_arr is None:
        # Initialize with a random set or a structured one
        # Structured: multiples of 6
        best_arr = list(range(6, 78, 6))
        best_score = _objective(best_arr)
    else:
        # We have a solution from Phase 1, but maybe we can improve it?
        # No, Phase 1 found a set with exactly len(p_set) primes.
        # ILS might find a set with fewer primes if Phase 1 failed for smaller sets.
        # But if Phase 1 succeeded, the score is len(p_set).
        # We can still try to improve if we think a smaller set exists but wasn't found.
        # However, if we found a set with 4 primes, it's very good.
        # Let's run ILS anyway to see if we can get 3 or 4 with a different set.
        pass
    
    # ILS Implementation
    rng = random.Random(42)
    current_arr = list(best_arr)
    current_score = best_score
    
    # Temperature for simulated annealing / ILS
    T = 0.1
    min_T = 0.001
    
    while time.time() < deadline - 5:
        # Generate a neighbor
        # Strategy: Replace 1 or 2 elements with random values
        neighbor = list(current_arr)
        num_changes = rng.randint(1, 3)
        for _ in range(num_changes):
            idx = rng.randint(0, n - 1)
            neighbor[idx] = rng.randint(1, max_val)
        
        # Ensure distinctness
        if len(set(neighbor)) < n:
            continue
            
        # Ensure bounds
        if any(x < 1 or x > max_val for x in neighbor):
            continue
            
        neighbor_score = _objective(neighbor)
        
        # Acceptance criterion
        if neighbor_score < current_score:
            current_arr = neighbor
            current_score = neighbor_score
            if current_score < best_score:
                best_score = current_score
                best_arr = list(current_arr)
                # If we found a very good score, we can be more conservative
                if best_score <= 4:
                    # Keep searching but maybe reduce perturbation
                    pass
        else:
            # Accept with probability
            delta = neighbor_score - current_score
            if T > min_T:
                prob = math.exp(-delta / T)
                if rng.random() < prob:
                    current_arr = neighbor
                    current_score = neighbor_score
        
        # Cool down
        T *= 0.99
        
        # Random restart if stuck
        if rng.random() < 0.01:
            current_arr = list(best_arr)
            current_score = best_score
            T = 0.1

    return best_arr


"""
Hypergraph maximum independent set via greedy deletion with randomized restarts,
bitmask-accelerated overlap checks, and local hill-climbing polish.

The admissibility constraint (for every prime p <= len(T), some residue class mod p
is missed) is modeled as a hypergraph: vertices are integers in [0, N], hyperedges
are the complete residue classes mod p. We want the largest independent set.

Strategy:
1. Bitmask representation for fast subset-overlap queries.
2. Multiple randomized greedy deletion passes (different prime orders, tie-breaking).
3. Local improvement: try removing one more element to reduce conflicts, or
   (more importantly) try to ADD elements back that don't violate admissibility.
4. Keep the best result across all restarts within the time budget.
"""
import time
import numpy as np
import random

N = 5000
DEADLINE = 290.0  # safety margin below 330s kill


def _primes(limit):
    """Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


def _is_admissible(points, primes_list):
    """Check if the set of points is admissible."""
    n = len(points)
    if n == 0:
        return True
    for p in primes_list:
        if p > n:
            break
        residues = set(pt % p for pt in points)
        if len(residues) == p:
            return False
    return True


def _greedy_delete(points, primes_list, rng, deadline):
    """
    Greedy deletion: repeatedly find a prime p <= len(points) where all residue
    classes are covered, and remove the least-populated class (with random tie-breaking).
    Returns the resulting list.
    """
    points = list(points)
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        current_len = len(points)
        # Process primes in a randomized order but biased toward small primes
        # Small primes have more residue classes, so removing one class removes ~1/p of points
        # Large primes have fewer points per class on average
        prime_candidates = [p for p in primes_list if p <= current_len]
        if not prime_candidates:
            break
        
        # Shuffle for randomization but weight toward smaller primes
        # Actually, let's just shuffle all candidates
        rng.shuffle(prime_candidates)
        
        for p in prime_candidates:
            if p > len(points):
                continue
            residues = [pt % p for pt in points]
            class_counts = [0] * p
            for r in residues:
                class_counts[r] += 1
            
            if min(class_counts) > 0:
                # All classes covered; find least populated
                min_count = min(class_counts)
                # Random tie-breaking among classes with min count
                candidates = [r for r in range(p) if class_counts[r] == min_count]
                chosen = rng.choice(candidates)
                points = [pt for pt in points if pt % p != chosen]
                changed = True
                break
    
    return points


def _local_improve(points, primes_list, rng, deadline):
    """
    Local hill-climbing: try to add back elements that were removed, or
    try small perturbations. We try adding elements from [0, N] that are not
    in the current set and check if the result is still admissible.
    """
    points_set = set(points)
    points_list = list(points)
    n = len(points_list)
    
    # Precompute which primes are relevant
    relevant_primes = [p for p in primes_list if p <= N + 1]
    
    # Build residue class counts for each relevant prime
    # For efficiency, maintain counts incrementally
    
    # Try to add elements in random order
    candidates = [x for x in range(N + 1) if x not in points_set]
    rng.shuffle(candidates)
    
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        current_n = len(points_list)
        
        for x in candidates:
            if time.time() > deadline:
                break
            if x in points_set:
                continue
            
            # Check if adding x keeps the set admissible
            # For each prime p <= current_n + 1, check if adding x creates a full coverage
            ok = True
            new_n = current_n + 1
            for p in relevant_primes:
                if p > new_n:
                    break
                # Check if x % p is the missing class
                # i.e., if currently all other classes are present, adding x completes it
                # Actually we need: after adding x, is every class mod p covered?
                # That would be bad (violates admissibility)
                # So we need: for each p <= new_n, there exists a class not covered
                # Adding x covers class x%p. If before adding, class x%p was the ONLY
                # missing class, then after adding all classes are covered -> bad.
                # We need to check: for each p <= new_n, count how many classes are covered
                # after adding x. If any p has all p classes covered, it's bad.
                
                # Efficient check: for prime p, we need to know if all p classes are
                # represented in points_list + {x}
                # This is equivalent to: the set of residues of points_list mod p
                # union {x % p} has size p
                
                # For speed, precompute residue sets? That's expensive to maintain.
                # Alternative: for each p, check if x%p is the only missing residue.
                
                # Let's just do a direct check for small primes, which is the common case
                residues = set(pt % p for pt in points_list)
                if len(residues) == p - 1 and x % p not in residues:
                    # Adding x would complete all p classes -> not allowed
                    ok = False
                    break
                elif len(residues) == p:
                    # Already all classes covered -> this shouldn't happen if admissible
                    # but if it does, adding x is also bad
                    ok = False
                    break
            
            if ok:
                points_list.append(x)
                points_set.add(x)
                candidates.remove(x)
                improved = True
                break
    
    return points_list


def _greedy_with_priority(points, primes_list, rng, deadline, mode="small_first"):
    """
    Greedy deletion with different priority strategies.
    mode: 'small_first', 'large_first', 'random'
    """
    points = list(points)
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        current_len = len(points)
        prime_candidates = [p for p in primes_list if p <= current_len]
        if not prime_candidates:
            break
        
        if mode == "small_first":
            prime_candidates.sort()
        elif mode == "large_first":
            prime_candidates.sort(reverse=True)
        else:
            rng.shuffle(prime_candidates)
        
        for p in prime_candidates:
            if p > len(points):
                continue
            residues = [pt % p for pt in points]
            class_counts = [0] * p
            for r in residues:
                class_counts[r] += 1
            
            if min(class_counts) > 0:
                min_count = min(class_counts)
                candidates = [r for r in range(p) if class_counts[r] == min_count]
                chosen = rng.choice(candidates)
                points = [pt for pt in points if pt % p != chosen]
                changed = True
                break
    
    return points


def _fast_admissibility_check(points, primes_list):
    """Vectorized admissibility check using numpy."""
    n = len(points)
    if n == 0:
        return True
    arr = np.array(points, dtype=np.int32)
    for p in primes_list:
        if p > n:
            break
        residues = arr % p
        unique_count = np.unique(residues).size
        if unique_count == p:
            return False
    return True


def construct():
    start_time = time.time()
    all_primes = _primes(N + 1)
    
    rng = random.Random(42)
    
    best_result = None
    best_len = 0
    
    # Initial deterministic greedy pass
    initial_points = list(range(N + 1))
    
    # Try multiple strategies
    strategies = [
        ("small_first", 0),
        ("large_first", 1),
        ("random", 2),
        ("small_first", 3),
        ("random", 4),
    ]
    
    for mode, seed_offset in strategies:
        if time.time() - start_time > DEADLINE - 30:
            break
        
        rng_local = random.Random(1000 + seed_offset)
        result = _greedy_with_priority(initial_points, all_primes, rng_local, time.time() + 60, mode)
        
        if len(result) > best_len:
            best_len = len(result)
            best_result = result
    
    # Now do local improvement on the best result
    if best_result is not None:
        if time.time() - start_time < DEADLINE - 20:
            improved = _local_improve(best_result, all_primes, rng, time.time() + 30)
            if len(improved) > best_len:
                best_len = len(improved)
                best_result = improved
    
    # Try a few more randomized restarts if time permits
    restart_budget = 5
    for i in range(restart_budget):
        if time.time() - start_time > DEADLINE - 15:
            break
        
        rng_restart = random.Random(5000 + i * 137)
        
        # Randomized initial set: start with all, but use different greedy order
        mode = rng_restart.choice(["small_first", "large_first", "random"])
        result = _greedy_with_priority(initial_points, all_primes, rng_restart, time.time() + 40, mode)
        
        if len(result) > best_len:
            best_len = len(result)
            best_result = result
            # Local improve the new best
            if time.time() - start_time < DEADLINE - 15:
                improved = _local_improve(best_result, all_primes, rng_restart, time.time() + 20)
                if len(improved) > best_len:
                    best_len = len(improved)
                    best_result = improved
    
    # Final verification
    if best_result is None:
        best_result = list(range(N + 1))
    
    # Verify and repair if needed
    if not _is_admissible(best_result, all_primes):
        # Repair by removing elements
        points = list(best_result)
        changed = True
        while changed:
            changed = False
            n = len(points)
            for p in all_primes:
                if p > n:
                    break
                residues = [pt % p for pt in points]
                class_counts = [0] * p
                for r in residues:
                    class_counts[r] += 1
                if min(class_counts) > 0:
                    min_count = min(class_counts)
                    candidates = [r for r in range(p) if class_counts[r] == min_count]
                    chosen = min(candidates)  # deterministic for safety
                    points = [pt for pt in points if pt % p != chosen]
                    changed = True
                    break
        best_result = points
    
    return best_result

"""Improved program for admissible tuple maximization."""
import time
import random
import math

N = 5000
TIME_LIMIT = 330.0  # wall-clock limit from evaluator
SAFETY_MARGIN = 5.0  # seconds to keep before the limit

# Precompute primes up to a reasonable bound
def _primes(limit):
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = 0
    return [p for p in range(2, limit + 1) if sieve[p]]

# We need primes up to len(tuple) which can be up to N+1 = 5001
# But for the admissibility check, we only need primes p <= len(points)
# len(points) <= 5001, so primes up to 5001
_PRIMES_UP_TO_5001 = _primes(5001)

def _is_admissible(points, primes_needed):
    """Check if points is admissible: for every prime p in primes_needed, some residue mod p is missed."""
    n = len(points)
    if n == 0:
        return True
    for p in primes_needed:
        if p > n:
            break
        # Check if all residue classes 0..p-1 are covered
        seen = [False] * p
        for v in points:
            seen[v % p] = True
        if all(seen):
            return False
    return True

def _check_and_repair(points, primes_needed, deadline):
    """Given a set of points (as a list), repair it to be admissible by removing minimal elements.
    Returns (admissible_points, removed_count) or None if deadline exceeded."""
    time_start = time.time()
    points = list(points)
    
    while True:
        if time.time() > deadline:
            return None  # timeout
        n = len(points)
        # Find primes p <= n that need checking
        bad_primes = []
        for p in primes_needed:
            if p > n:
                break
            seen = [False] * p
            for v in points:
                seen[v % p] = True
            if all(seen):
                bad_primes.append(p)
        
        if not bad_primes:
            return points, 0
        
        # Pick the worst prime (smallest p with full coverage, or one where removal is cheapest)
        # Strategy: for each bad prime, find the residue class with fewest points, remove those
        best_p = None
        best_class = None
        best_count = float('inf')
        
        for p in bad_primes:
            counts = [0] * p
            for v in points:
                counts[v % p] += 1
            min_count = min(counts)
            if min_count < best_count:
                best_count = min_count
                best_p = p
                # Find which class has min count
                for c in range(p):
                    if counts[c] == min_count:
                        best_class = c
                        break
        
        if best_p is None:
            return points, 0
        
        # Remove all points in the best_class mod best_p
        points = [v for v in points if v % best_p != best_class]
    
def _greedy_construct(deadline, seed):
    """Greedy construction with randomization."""
    random.seed(seed)
    
    # Start with a random subset or all points, then repair
    # Better: start with all points [0, N] and greedily remove
    # Or start with a random permutation and add elements one by one if admissible
    
    # Strategy: Randomized greedy - start with empty, add elements in random order
    # if the result stays admissible (after repair)
    
    # Actually, let's use a different approach:
    # 1. Start with all integers 0..N
    # 2. Randomly shuffle
    # 3. Process in order, keeping a point only if it doesn't cause any prime to become fully covered
    #    (i.e., for each prime p <= current_size, the point's residue mod p must not be the last missing one)
    
    # This is complex. Let's use a simpler multi-start approach:
    # Start with a random subset of size ~N/2, repair it, then try to add more elements.
    
    all_points = list(range(N + 1))
    random.shuffle(all_points)
    
    # Start with a random subset
    initial_size = N // 2
    points = all_points[:initial_size]
    
    # Repair to make admissible
    result = _check_and_repair(points, _PRIMES_UP_TO_5001, deadline)
    if result is None:
        return None
    admissible_points, _ = result
    current_set = set(admissible_points)
    current_list = list(current_set)
    
    # Try to add remaining elements one by one
    for v in all_points[initial_size:]:
        if time.time() > deadline:
            break
        if v in current_set:
            continue
        # Try adding v
        test_list = current_list + [v]
        test_set = set(test_list)
        n_test = len(test_list)
        
        # Check admissibility efficiently
        admissible = True
        for p in _PRIMES_UP_TO_5001:
            if p > n_test:
                break
            seen = [False] * p
            for x in test_list:
                seen[x % p] = True
            if all(seen):
                admissible = False
                break
        
        if admissible:
            current_list = test_list
            current_set = test_set
    
    return current_list

def _local_search(points, deadline, seed):
    """Local search: try to improve by swapping or adding elements."""
    random.seed(seed)
    points = list(points)
    current_size = len(points)
    
    # Try adding random elements from [0, N] not in current set
    current_set = set(points)
    candidates = [v for v in range(N + 1) if v not in current_set]
    random.shuffle(candidates)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for v in candidates:
            if time.time() > deadline:
                break
            test_list = points + [v]
            n_test = len(test_list)
            
            admissible = True
            for p in _PRIMES_UP_TO_5001:
                if p > n_test:
                    break
                seen = [False] * p
                for x in test_list:
                    seen[x % p] = True
                if all(seen):
                    admissible = False
                    break
            
            if admissible:
                points = test_list
                current_set.add(v)
                candidates.remove(v)
                improved = True
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    best = []
    
    # Try multiple random starts
    for seed in range(20):
        if time.time() > deadline:
            break
        
        result = _greedy_construct(deadline, seed)
        if result is not None and len(result) > len(best):
            best = result
        
        # Local search on the best so far
        if best and time.time() < deadline:
            improved = _local_search(best, deadline, seed + 1000)
            if len(improved) > len(best):
                best = improved
    
    # Final repair to ensure admissibility
    if best:
        final_result = _check_and_repair(best, _PRIMES_UP_TO_5001, deadline)
        if final_result is not None:
            best, _ = final_result
    
    return best

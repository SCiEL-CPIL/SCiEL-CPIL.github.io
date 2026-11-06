"""Combined program: vectorized greedy set-reduction with randomized tie-breaking,
multiple restarts, and local improvement (add-back) for larger admissible tuples."""
import time
import random
import numpy as np

N = 5000
DEADLINE = 300.0  # seconds, safety margin below 330s kill


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


def _greedy_repair(points, all_primes, rng, time_limit):
    """Greedy repair: repeatedly find a prime p <= len(points) where all residue
    classes mod p are covered, and remove the least-populated (randomly among ties)
    residue class. Returns the repaired points array."""
    start = time.time()
    changed = True
    while changed:
        if time.time() - start > time_limit:
            break
        changed = False
        current_len = len(points)
        
        # Process primes in ascending order
        for p in all_primes:
            if p > current_len:
                break
            
            residues = points % p
            class_counts = np.bincount(residues, minlength=p)
            
            if np.all(class_counts > 0):
                # Find minimum count
                min_count = class_counts.min()
                # Get all classes with minimum count
                min_classes = np.where(class_counts == min_count)[0]
                # Randomly pick one among the ties
                min_class = min_classes[rng.randint(0, len(min_classes) - 1)]
                # Remove all points with that residue
                points = points[points % p != min_class]
                changed = True
                break
    
    return points


def _is_admissible(points, all_primes):
    """Check if points form an admissible tuple."""
    n = len(points)
    for p in all_primes:
        if p > n:
            break
        residues = set(v % p for v in points)
        if len(residues) == p:
            return False
    return True


def _local_improve(points, all_primes, rng, time_limit):
    """Try to add back removed points one at a time if the result stays admissible.
    Uses a set of candidate points (the complement in [0, N])."""
    start = time.time()
    current = set(points.tolist())
    candidates = [x for x in range(N + 1) if x not in current]
    rng.shuffle(candidates)
    
    improved = False
    for cand in candidates:
        if time.time() - start > time_limit:
            break
        # Try adding cand
        trial = list(current) + [cand]
        if _is_admissible(np.array(trial, dtype=np.int32), all_primes):
            current.add(cand)
            improved = True
    
    if improved:
        return np.array(sorted(current), dtype=np.int32)
    return points


def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    # Precompute primes up to N+1
    all_primes = _primes(N + 1)
    
    # Start with all integers in [0, N]
    points = np.arange(N + 1, dtype=np.int32)
    
    # Phase 1: Initial greedy repair with a time budget
    time_budget = (DEADLINE - (time.time() - start_time)) * 0.4
    points = _greedy_repair(points, all_primes, rng, time_budget)
    
    best_size = len(points)
    best_points = points.copy()
    
    # Phase 2: Multiple restarts with randomized tie-breaking
    elapsed = time.time() - start_time
    remaining = DEADLINE - 10 - elapsed  # safety margin
    
    if remaining > 5:
        # Do several restarts, each starting from full set
        num_restarts = min(20, int(remaining / 5))  # at least 5s per restart
        for restart in range(num_restarts):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            # Fresh start
            pts = np.arange(N + 1, dtype=np.int32)
            # Use a different random seed for each restart
            rng_r = random.Random(42 + restart * 1000)
            
            # Greedy repair with randomized tie-breaking
            pts = _greedy_repair(pts, all_primes, rng_r, 2.0)
            
            if len(pts) > best_size:
                best_size = len(pts)
                best_points = pts.copy()
    
    # Phase 3: Local improvement on the best result
    elapsed = time.time() - start_time
    remaining = DEADLINE - 10 - elapsed
    if remaining > 2:
        best_points = _local_improve(best_points, all_primes, rng, remaining - 1)
        best_size = len(best_points)
    
    # Final verification and fix
    result = best_points.tolist()
    n_final = len(result)
    for p in all_primes:
        if p > n_final:
            break
        residues = set(v % p for v in result)
        if len(residues) == p:
            counts = [0] * p
            for v in result:
                counts[v % p] += 1
            min_c = min(counts)
            min_r = counts.index(min_c)
            result = [v for v in result if v % p != min_r]
            n_final = len(result)
    
    return result

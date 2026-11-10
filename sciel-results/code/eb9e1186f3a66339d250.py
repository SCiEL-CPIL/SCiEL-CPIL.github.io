"""Improved program for the narrow admissible tuple task.

Strategy: multi-strategy greedy construction with randomized restarts and iterative local search (add-back).
The core idea is to start with a large candidate set (all integers in [0, N]) and iteratively remove
residue classes that are fully covered, while trying to add back removed elements when possible.
Randomized restarts with different removal orders help escape local optima.
"""

import time
import random
import math

N = 5000
DEADLINE = 320  # seconds; leave safety margin before 330s kill

def _primes(limit):
    """Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p*p::p] = bytearray(len(sieve[p*p::p]))
    return [p for p in range(2, limit + 1) if sieve[p]]

# Precompute primes up to N (max possible tuple length is N+1, but realistically much smaller)
_PRIMES_UP_TO_N = _primes(N)

def _is_admissible(points, primes_needed):
    """Check if points form an admissible tuple with respect to primes_needed."""
    for p in primes_needed:
        residues = set(v % p for v in points)
        if len(residues) == p:
            return False
    return True

def _count_covered_residues(points, p):
    """Return the set of residues mod p that are covered by points."""
    return set(v % p for v in points)

def _greedy_remove(points, primes_to_check, rng):
    """
    Greedy removal: while there exists a prime p in primes_to_check such that all p residues
    are covered, remove the least-populated residue class mod p.
    Returns the final list of points.
    """
    points = list(points)
    n = len(points)
    # We only need to check primes up to len(points)
    relevant_primes = [p for p in primes_to_check if p <= len(points)]
    
    while relevant_primes:
        # Find a prime where all residues are covered
        violated = None
        for p in relevant_primes:
            residues = set(v % p for v in points)
            if len(residues) == p:
                violated = p
                break
        
        if violated is None:
            break
        
        p = violated
        # Count residues
        classes = {}
        for v in points:
            r = v % p
            if r not in classes:
                classes[r] = []
            classes[r].append(v)
        
        # Remove the smallest class (with random tie-breaking)
        min_size = min(len(c) for c in classes.values())
        smallest_classes = [c for c in classes.values() if len(c) == min_size]
        chosen = rng.choice(smallest_classes)
        chosen_set = set(chosen)
        points = [v for v in points if v not in chosen_set]
        
        # Update relevant_primes: if len(points) changed, some primes may no longer be relevant
        relevant_primes = [p for p in relevant_primes if p <= len(points)]
    
    return points

def _add_back_search(points, all_candidates, primes_to_check, rng, max_rounds=100):
    """
    Try to add back removed elements one at a time if it doesn't violate admissibility.
    all_candidates: set of all integers in [0, N]
    points: current admissible tuple
    """
    points_set = set(points)
    removed = [v for v in all_candidates if v not in points_set]
    rng.shuffle(removed)
    
    # We need to check primes up to len(points) + 1 (since adding one increases size by 1)
    improved = True
    round_num = 0
    while improved and round_num < max_rounds:
        improved = False
        round_num += 1
        # Re-check in random order
        order = list(removed)
        rng.shuffle(order)
        for v in order:
            if v in points_set:
                continue
            new_points = points_set | {v}
            # Check admissibility: primes up to len(new_points)
            new_len = len(new_points)
            ok = True
            for p in primes_to_check:
                if p > new_len:
                    break
                residues = set(x % p for x in new_points)
                if len(residues) == p:
                    ok = False
                    break
            if ok:
                points_set.add(v)
                improved = True
                # Continue trying to add more
        # If we added anything, continue; the removed list is implicitly updated via points_set
    
    return list(points_set)

def _construct_one_strategy(rng, strategy):
    """
    Construct an admissible tuple using a specific strategy.
    Returns the list of points.
    """
    all_candidates = set(range(N + 1))
    
    if strategy == 0:
        # Start with all candidates, greedy remove
        points = list(all_candidates)
        points = _greedy_remove(points, _PRIMES_UP_TO_N, rng)
    elif strategy == 1:
        # Start with a random subset of ~70% density, then greedy remove
        density = 0.7
        points = [v for v in range(N + 1) if rng.random() < density]
        points = _greedy_remove(points, _PRIMES_UP_TO_N, rng)
    elif strategy == 2:
        # Start with all candidates, but use a different removal heuristic:
        # remove the residue class that has the most "conflict" (i.e., removing it helps most primes)
        points = list(all_candidates)
        n = len(points)
        relevant_primes = [p for p in _PRIMES_UP_TO_N if p <= n]
        
        while relevant_primes:
            violated = None
            for p in relevant_primes:
                residues = set(v % p for v in points)
                if len(residues) == p:
                    violated = p
                    break
            if violated is None:
                break
            
            p = violated
            classes = {}
            for v in points:
                r = v % p
                if r not in classes:
                    classes[r] = []
                classes[r].append(v)
            
            # Score each class: how many other primes would be helped by removing this class?
            # Approximation: count how many of these elements are in "tight" classes for other primes
            # For simplicity, just pick smallest with random tie-break
            min_size = min(len(c) for c in classes.values())
            smallest_classes = [c for c in classes.values() if len(c) == min_size]
            chosen = rng.choice(smallest_classes)
            chosen_set = set(chosen)
            points = [v for v in points if v not in chosen_set]
            relevant_primes = [p for p in relevant_primes if p <= len(points)]
        
        points = _greedy_remove(points, _PRIMES_UP_TO_N, rng)
    elif strategy == 3:
        # Start with primes > sqrt(N) plus 0, then add back
        points = [0] + [p for p in _PRIMES_UP_TO_N if p * p > N]
        points = _add_back_search(points, all_candidates, _PRIMES_UP_TO_N, rng, max_rounds=200)
    else:
        # Strategy 4: Random initial subset of varying density
        density = rng.uniform(0.3, 0.9)
        points = [v for v in range(N + 1) if rng.random() < density]
        points = _greedy_remove(points, _PRIMES_UP_TO_N, rng)
        points = _add_back_search(points, all_candidates, _PRIMES_UP_TO_N, rng, max_rounds=50)
    
    return points

def construct():
    """
    Main entry point. Uses multi-strategy greedy construction with randomized restarts
    and iterative local search (add-back) to find a large admissible tuple in [0, N].
    """
    start_time = time.time()
    best_points = []
    best_size = 0
    
    # First, do a quick baseline to have something valid
    rng0 = random.Random(42)
    baseline = _construct_one_strategy(rng0, 0)
    if _is_admissible(baseline, [p for p in _PRIMES_UP_TO_N if p <= len(baseline)]):
        best_points = baseline
        best_size = len(baseline)
    
    # Multi-restart search
    seed = 12345
    rng = random.Random(seed)
    num_restarts = 0
    
    while time.time() - start_time < DEADLINE:
        strategy = rng.randint(0, 4)
        points = _construct_one_strategy(rng, strategy)
        
        # Verify admissibility
        if not _is_admissible(points, [p for p in _PRIMES_UP_TO_N if p <= len(points)]):
            # Repair if needed
            points = _greedy_remove(points, _PRIMES_UP_TO_N, rng)
        
        if len(points) > best_size:
            best_size = len(points)
            best_points = points
            # Try add-back on the new best
            if time.time() - start_time < DEADLINE - 10:
                improved = _add_back_search(best_points, set(range(N + 1)), _PRIMES_UP_TO_N, rng, max_rounds=100)
                if _is_admissible(improved, [p for p in _PRIMES_UP_TO_N if p <= len(improved)]):
                    if len(improved) > best_size:
                        best_size = len(improved)
                        best_points = improved
        
        num_restarts += 1
        # If we've been running for a while without improvement, try more aggressive add-back
        if num_restarts % 5 == 0 and time.time() - start_time < DEADLINE - 30:
            # Final polish: try extensive add-back on the current best
            improved = _add_back_search(best_points, set(range(N + 1)), _PRIMES_UP_TO_N, rng, max_rounds=500)
            if _is_admissible(improved, [p for p in _PRIMES_UP_TO_N if p <= len(improved)]):
                if len(improved) > best_size:
                    best_size = len(improved)
                    best_points = improved
    
    # Final safety check
    if not _is_admissible(best_points, [p for p in _PRIMES_UP_TO_N if p <= len(best_points)]):
        # Emergency repair
        best_points = _greedy_remove(best_points, _PRIMES_UP_TO_N, random.Random(0))
    
    return best_points

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from math import isqrt

N = 5000
DEADLINE = 280.0  # Safety margin before 300s kill

# Precompute primes up to N
def _primes(limit):
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, isqrt(limit) + 1):
        if sieve[p]:
            sieve[p*p::p] = bytearray(len(sieve[p*p::p]))
    return [p for p in range(limit + 1) if sieve[p]]

ALL_PRIMES = _primes(N)
PRIMES_BY_INDEX = {p: i for i, p in enumerate(ALL_PRIMES)}

def _primes_up_to(n):
    """Return list of primes <= n."""
    return [p for p in ALL_PRIMES if p <= n]

def _is_admissible_fast(points, max_p):
    """
    Check admissibility using bitmask for each prime.
    For each prime p <= max_p, check if all p residues are covered.
    Returns True if admissible.
    """
    for p in ALL_PRIMES:
        if p > max_p:
            break
        mask = 0
        for v in points:
            mask |= (1 << (v % p))
        if mask == (1 << p) - 1:
            return False
    return True

def _greedy_repair(points, primes_list, deadline):
    """
    Remove elements to make the set admissible.
    Greedily remove elements from the smallest residue class for violating primes.
    """
    points = list(points)
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        for p in primes_list:
            if p > len(points):
                break
            # Count residues
            classes = {}
            for v in points:
                r = v % p
                classes[r] = classes.get(r, 0) + 1
            if len(classes) == p:
                # Find smallest class
                min_count = min(classes.values())
                min_residues = [r for r, c in classes.items() if c == min_count]
                # Remove all elements in one of the smallest classes
                remove_set = set()
                for v in points:
                    if v % p in min_residues:
                        remove_set.add(v)
                points = [v for v in points if v not in remove_set]
                changed = True
                break
    return points

def _add_back_search(points, primes_list, deadline, max_attempts=500):
    """
    Try to add back removed elements or new elements to maximize size.
    """
    points = list(points)
    point_set = set(points)
    
    # Candidate pool: all integers in [0, N] not in points
    # We'll try random candidates and structured ones
    
    np.random.seed(42)
    attempts = 0
    improved = True
    while improved and time.time() < deadline and attempts < max_attempts:
        improved = False
        attempts += 1
        
        # Generate candidates
        if attempts % 10 == 0:
            # Structured candidates: small numbers, primes
            candidates = []
            for i in range(0, min(100, N + 1)):
                if i not in point_set:
                    candidates.append(i)
            for p in ALL_PRIMES:
                if p <= N and p not in point_set:
                    candidates.append(p)
        else:
            # Random candidates
            candidates = [np.random.randint(0, N + 1) for _ in range(50)]
        
        # Try each candidate
        for cand in candidates:
            if cand in point_set:
                continue
            new_points = points + [cand]
            new_size = len(new_points)
            if _is_admissible_fast(new_points, new_size):
                points = new_points
                point_set.add(cand)
                improved = True
                break
    
    return points

def _local_optimization(points, deadline):
    """
    Local search: try swapping elements to improve admissibility margins.
    """
    points = list(points)
    point_set = set(points)
    
    # Compute residue counts for each prime
    def compute_margins(pts):
        margins = {}
        for p in ALL_PRIMES:
            if p > len(pts):
                break
            counts = [0] * p
            for v in pts:
                counts[v % p] += 1
            # Margin: how many residues are not fully covered
            # We want to maximize the minimum "slack"
            margins[p] = min(counts)
        return margins
    
    # Try to improve by removing elements that cause tight constraints
    # and adding elements that relax them
    for _ in range(100):
        if time.time() > deadline:
            break
        
        margins = compute_margins(points)
        if not margins:
            break
        
        # Find the tightest prime
        tightest_p = min(margins, key=margins.get)
        tightest_margin = margins[tightest_p]
        
        if tightest_margin > 0:
            # All residues have at least 1 element, try to add more
            # Find which residue has the minimum count
            counts = [0] * tightest_p
            for v in points:
                counts[v % tightest_p] += 1
            min_count = min(counts)
            min_residues = [r for r, c in enumerate(counts) if c == min_count]
            
            # Try adding elements in the least populated residues
            for r in min_residues:
                # Find candidates with this residue
                found = False
                for cand in range(r, N + 1, tightest_p):
                    if cand not in point_set:
                        new_points = points + [cand]
                        if _is_admissible_fast(new_points, len(new_points)):
                            points = new_points
                            point_set.add(cand)
                            found = True
                            break
                if found:
                    break
        else:
            # Some residue is empty, we need to add an element in that residue
            counts = [0] * tightest_p
            for v in points:
                counts[v % tightest_p] += 1
            empty_residues = [r for r, c in enumerate(counts) if c == 0]
            
            for r in empty_residues:
                found = False
                for cand in range(r, N + 1, tightest_p):
                    if cand not in point_set:
                        new_points = points + [cand]
                        if _is_admissible_fast(new_points, len(new_points)):
                            points = new_points
                            point_set.add(cand)
                            found = True
                            break
                if found:
                    break
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_points = []
    best_size = 0
    
    # Multiple strategies with different initial configurations
    strategies = [
        # Strategy 1: Start with 0
        lambda: [0],
        # Strategy 2: Start with small primes
        lambda: [p for p in ALL_PRIMES if p <= 100],
        # Strategy 3: Start with 0,1,2,3
        lambda: [0, 1, 2, 3],
        # Strategy 4: Start with even numbers up to 50
        lambda: list(range(0, 50, 2)),
        # Strategy 5: Start with 0 and large primes
        lambda: [0] + [p for p in ALL_PRIMES if p > 50 and p <= N],
    ]
    
    for strat_idx, strategy_fn in enumerate(strategies):
        if time.time() > deadline - 30:
            break
        
        points = strategy_fn()
        
        # Repair to make admissible
        primes_needed = _primes_up_to(len(points))
        points = _greedy_repair(points, primes_needed, deadline - 20)
        
        # Validate
        if not _is_admissible_fast(points, len(points)):
            continue
        
        # Add back search
        primes_needed = _primes_up_to(len(points) + 100)
        points = _add_back_search(points, primes_needed, deadline - 10, max_attempts=300)
        
        # Local optimization
        points = _local_optimization(points, deadline - 5)
        
        # Final validation
        if _is_admissible_fast(points, len(points)):
            if len(points) > best_size:
                best_size = len(points)
                best_points = points
        else:
            # Fallback: repair again
            points = _greedy_repair(points, _primes_up_to(len(points)), deadline - 5)
            if _is_admissible_fast(points, len(points)):
                if len(points) > best_size:
                    best_size = len(points)
                    best_points = points
    
    # If we still have time, do a final greedy expansion
    if time.time() < deadline - 10 and best_points:
        best_points = _add_back_search(best_points, _primes_up_to(len(best_points) + 200), 
                                       deadline - 5, max_attempts=200)
        if _is_admissible_fast(best_points, len(best_points)):
            best_points = _local_optimization(best_points, deadline - 2)
    
    # Ensure we have a valid result
    if not best_points or not _is_admissible_fast(best_points, len(best_points)):
        # Fallback to a known valid set
        best_points = [0, 1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        while not _is_admissible_fast(best_points, len(best_points)) and len(best_points) > 1:
            best_points.pop()
    
    return best_points
# EVOLVE-BLOCK-END

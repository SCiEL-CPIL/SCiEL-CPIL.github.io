"""Improved admissible tuple construction for the fixed-diameter maximization task."""
import time
import math
import random
import numpy as np
from collections import defaultdict

N = 5000
DEADLINE_SECONDS = 300.0  # safety margin before 330s kill

def _primes(limit):
    """Return list of primes up to limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p*p::p] = bytearray(len(sieve[p*p::p]))
    return [i for i in range(limit + 1) if sieve[i]]

def _is_admissible(points, primes_to_check):
    """Check if points form an admissible tuple."""
    for p in primes_to_check:
        residues = set(v % p for v in points)
        if len(residues) == p:
            return False
    return True

def _greedy_removal_optimal(points, primes_to_check, deadline):
    """
    Optimized greedy removal: for each prime p that is fully covered,
    remove the smallest residue class. Process primes in order of smallest
    residue class size to maximize overlap of removals.
    """
    points = set(points)
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        
        # Find all fully-covered primes and their smallest residue class
        candidates = []
        for p in primes_to_check:
            if p > len(points):
                continue
            classes = defaultdict(int)
            for v in points:
                classes[v % p] += 1
            if len(classes) == p:
                # All residues covered, find smallest class
                min_class = min(classes.items(), key=lambda x: x[1])
                candidates.append((min_class[1], p, min_class[0]))
        
        if not candidates:
            break
        
        # Sort by smallest class size (remove smallest first for max overlap)
        candidates.sort()
        for size, p, r in candidates:
            if time.time() > deadline:
                break
            # Remove all points with v % p == r
            to_remove = {v for v in points if v % p == r}
            if to_remove:
                points -= to_remove
                changed = True
                break  # Restart after one removal
    
    return list(points)

def _add_back_greedy(points, primes_to_check, deadline):
    """Try to add back elements greedily, checking admissibility efficiently."""
    points = set(points)
    candidates = set(range(N + 1)) - points
    candidates = list(candidates)
    random.shuffle(candidates)
    
    added = True
    while added:
        if time.time() > deadline:
            break
        added = False
        for v in candidates[:]:
            if time.time() > deadline:
                break
            test_points = points | {v}
            new_size = len(test_points)
            admissible = True
            for p in primes_to_check:
                if p > new_size:
                    break
                residues = set(x % p for x in test_points)
                if len(residues) == p:
                    admissible = False
                    break
            if admissible:
                points.add(v)
                candidates.remove(v)
                added = True
    
    return list(points)

def _local_search_swap(points, primes_to_check, deadline):
    """Try swapping: remove one point, add two different points if admissible."""
    points = set(points)
    candidates = set(range(N + 1)) - points
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for v_remove in list(points):
            if time.time() > deadline:
                break
            temp_points = points - {v_remove}
            temp_size = len(temp_points)
            
            # Try adding pairs of candidates
            for v1 in list(candidates):
                if time.time() > deadline:
                    break
                for v2 in list(candidates):
                    if v1 == v2:
                        continue
                    if time.time() > deadline:
                        break
                    test_points = temp_points | {v1, v2}
                    new_size = len(test_points)
                    admissible = True
                    for p in primes_to_check:
                        if p > new_size:
                            break
                        residues = set(x % p for x in test_points)
                        if len(residues) == p:
                            admissible = False
                            break
                    if admissible:
                        points = test_points
                        candidates = set(range(N + 1)) - points
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    return list(points)

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # Primes up to N+1 since tuple size can be up to N+1
    all_primes = _primes(N + 1)
    
    best_points = []
    best_size = 0
    
    # Strategy 1: Start with all integers, greedy removal
    points = list(range(N + 1))
    points = _greedy_removal_optimal(points, all_primes, deadline)
    points = _add_back_greedy(points, all_primes, deadline)
    best_points = points
    best_size = len(points)
    
    if time.time() > deadline - 30:
        return best_points
    
    # Strategy 2: Multiple random restarts with different seeds
    random.seed(123)
    for attempt in range(10):
        if time.time() > deadline - 20:
            break
        
        # Create a random subset as starting point
        subset_size = random.randint(3000, 4500)
        init_points = random.sample(range(N + 1), subset_size)
        init_points = _greedy_removal_optimal(init_points, all_primes, deadline)
        init_points = _add_back_greedy(init_points, all_primes, deadline)
        
        if len(init_points) > best_size:
            best_points = init_points
            best_size = len(init_points)
    
    # Strategy 3: Local search on best solution
    if time.time() < deadline - 10:
        best_points = _local_search_swap(best_points, all_primes, deadline)
        best_size = len(best_points)
    
    # Final verification and fallback
    primes_to_check = [p for p in all_primes if p <= len(best_points)]
    if not _is_admissible(best_points, primes_to_check):
        points = list(range(N + 1))
        points = _greedy_removal_optimal(points, all_primes, deadline)
        best_points = points
    
    return best_points

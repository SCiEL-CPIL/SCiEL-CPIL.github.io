"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random

N = 5000
DEADLINE = 320  # seconds, safety margin before 330s kill

# EVOLVE-BLOCK-START
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


def _greedy_remove(points, primes_list):
    """Greedy removal: iteratively remove least-populated residue class for violating primes."""
    points = list(points)
    changed = True
    while changed:
        changed = False
        n = len(points)
        for p in primes_list:
            if p > n:
                break
            classes = {}
            for v in points:
                r = v % p
                if r not in classes:
                    classes[r] = []
                classes[r].append(v)
            if len(classes) == p:
                smallest_class = min(classes.keys(), key=lambda r: len(classes[r]))
                to_remove = set(classes[smallest_class])
                points = [v for v in points if v not in to_remove]
                changed = True
                break
    return points


def _try_insertion_incremental(points, removed_elements, primes_list, rng):
    """Try to insert removed elements using incremental admissibility tracking."""
    points = list(points)
    n = len(points)
    
    # Track residue class occupancy for each relevant prime
    # For prime p, we track: how many elements are in each residue class mod p
    # A prime p is "saturated" if all p residue classes have at least one element
    # Adding an element x is allowed iff for no prime p <= n+1 does x complete all p classes
    
    # Build initial counts
    relevant_primes = [p for p in primes_list if p <= n]
    class_counts = {}
    for p in relevant_primes:
        counts = [0] * p
        for v in points:
            counts[v % p] += 1
        class_counts[p] = counts
    
    removed = list(removed_elements)
    rng.shuffle(removed)
    
    for elem in removed:
        n = len(points)
        new_n = n + 1
        # Check all primes p <= new_n
        can_add = True
        for p in primes_list:
            if p > new_n:
                break
            if p not in class_counts:
                # Build counts for this prime
                counts = [0] * p
                for v in points:
                    counts[v % p] += 1
                class_counts[p] = counts
            
            r = elem % p
            if class_counts[p][r] == 0:
                # This residue class is empty. Check if all others are non-empty
                # If yes, adding would saturate prime p -> violation
                all_nonzero = True
                for r2 in range(p):
                    if r2 != r and class_counts[p][r2] == 0:
                        all_nonzero = False
                        break
                if all_nonzero:
                    can_add = False
                    break
        
        if can_add:
            points.append(elem)
            # Update counts for all primes p <= new_n
            for p in primes_list:
                if p > new_n:
                    break
                class_counts[p][elem % p] += 1
    
    return points


def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    max_len = N + 1
    primes_list = _primes(max_len)
    
    all_points = list(range(N + 1))
    
    # Phase 1: Greedy removal from full set
    points = _greedy_remove(all_points, primes_list)
    points_set = set(points)
    removed = [v for v in all_points if v not in points_set]
    
    best = list(points)
    best_size = len(best)
    
    # Phase 2: Multiple rounds of randomized incremental insertion
    while time.time() - start_time < DEADLINE - 5:
        candidate = _try_insertion_incremental(best, removed, primes_list, rng)
        if len(candidate) > best_size:
            best = candidate
            best_size = len(best)
            best_set = set(best)
            removed = [v for v in all_points if v not in best_set]
    
    return best
# EVOLVE-BLOCK-END

"""Construct the largest admissible tuple in [0, 5000]."""
import time
import math
from collections import defaultdict

N = 5000

def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

_PRIMES_CACHE = {}

def _get_primes(limit):
    if limit not in _PRIMES_CACHE:
        _PRIMES_CACHE[limit] = _primes(limit)
    return _PRIMES_CACHE[limit]

def _is_admissible(points, primes_list):
    """Check if points form an admissible tuple for the given primes."""
    n = len(points)
    for p in primes_list:
        if p > n:
            break
        covered = set()
        for v in points:
            covered.add(v % p)
            if len(covered) == p:
                return False
    return True

def _repair_greedy(points, primes_list):
    """Repair points to make them admissible by removing elements greedily."""
    points = list(points)
    changed = True
    while changed:
        changed = False
        n = len(points)
        for p in primes_list:
            if p > n:
                break
            # Count elements in each residue class
            classes = defaultdict(list)
            for v in points:
                classes[v % p].append(v)
            if len(classes) == p:
                # All classes covered; remove the smallest class
                smallest = min(classes.values(), key=len)
                remove_set = set(smallest)
                points = [v for v in points if v not in remove_set]
                changed = True
                break
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 330s kill
    
    best_result = []
    best_size = 0
    
    # Precompute all primes up to N+1
    all_primes = _get_primes(N + 1)
    
    # Strategy 1: Start with all numbers and repair
    all_points = list(range(N + 1))
    repaired = _repair_greedy(all_points, all_primes)
    if len(repaired) > best_size:
        best_size = len(repaired)
        best_result = repaired
    
    if time.time() > deadline:
        return best_result
    
    # Strategy 2: Start with odd numbers (misses class 0 mod 2)
    odd_points = [x for x in range(1, N + 1, 2)]
    repaired2 = _repair_greedy(odd_points, all_primes)
    if len(repaired2) > best_size:
        best_size = len(repaired2)
        best_result = repaired2
    
    if time.time() > deadline:
        return best_result
    
    # Strategy 3: Numbers coprime to 2*3*5 = 30
    residues_30 = [x for x in range(1, 30) if math.gcd(x, 30) == 1]
    points3 = [x for x in range(1, N + 1) if x % 30 in residues_30]
    repaired3 = _repair_greedy(points3, all_primes)
    if len(repaired3) > best_size:
        best_size = len(repaired3)
        best_result = repaired3
    
    if time.time() > deadline:
        return best_result
    
    # Strategy 4: Numbers coprime to 2*3*5*7 = 210
    residues_210 = [x for x in range(1, 210) if math.gcd(x, 210) == 1]
    points4 = [x for x in range(1, N + 1) if x % 210 in residues_210]
    repaired4 = _repair_greedy(points4, all_primes)
    if len(repaired4) > best_size:
        best_size = len(repaired4)
        best_result = repaired4
    
    if time.time() > deadline:
        return best_result
    
    # Strategy 5: Numbers coprime to 2*3*5*7*11 = 2310
    residues_2310 = [x for x in range(1, 2310) if math.gcd(x, 2310) == 1]
    points5 = [x for x in range(1, N + 1) if x % 2310 in residues_2310]
    repaired5 = _repair_greedy(points5, all_primes)
    if len(repaired5) > best_size:
        best_size = len(repaired5)
        best_result = repaired5
    
    if time.time() > deadline:
        return best_result
    
    # Strategy 6: Numbers coprime to 2*3*5*7*11*13 = 30030
    prod = 2 * 3 * 5 * 7 * 11 * 13  # 30030
    points6 = [x for x in range(1, N + 1) if math.gcd(x, prod) == 1]
    repaired6 = _repair_greedy(points6, all_primes)
    if len(repaired6) > best_size:
        best_size = len(repaired6)
        best_result = repaired6
    
    if time.time() > deadline:
        return best_result
    
    # Strategy 7: Try to add back elements to the best result
    if time.time() < deadline - 10:
        best_set = set(best_result)
        current_list = list(best_result)
        
        # Try adding each missing element
        for candidate in range(N + 1):
            if time.time() > deadline - 10:
                break
            if candidate in best_set:
                continue
            test_list = current_list + [candidate]
            test_primes = _get_primes(len(test_list))
            if _is_admissible(test_list, test_primes):
                current_list = test_list
                best_set.add(candidate)
    
    return current_list if 'current_list' in dir() and len(current_list) >= best_size else best_result

import time
import math
import random

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def _greedy_pass(primes, seed_val):
    random.seed(seed_val)
    points = list(range(N + 1))
    
    for p in primes:
        current_size = len(points)
        if p > current_size:
            break
            
        counts = [0] * p
        for v in points:
            counts[v % p] += 1
            
        if 0 in counts:
            continue
            
        min_count = min(counts)
        candidates = [r for r in range(p) if counts[r] == min_count]
        
        target_residue = candidates[random.randint(0, len(candidates) - 1)]
        
        points = [v for v in points if v % p != target_residue]
        
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 290  # 10s safety margin
    
    primes = _primes(N)
    
    best_points = None
    best_size = -1
    
    seed_counter = 0
    while time.time() < deadline:
        current_points = _greedy_pass(primes, seed_counter)
        current_size = len(current_points)
        
        if current_size > best_size:
            best_size = current_size
            best_points = current_points
            
        seed_counter += 1
        
        if best_size == N + 1:
            break

    if best_points is None:
        best_points = _greedy_pass(primes, 0)
        
    return best_points

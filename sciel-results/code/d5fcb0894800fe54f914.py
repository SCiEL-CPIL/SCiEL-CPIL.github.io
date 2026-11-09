import random
import time

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def _is_admissible(points, primes_list):
    n = len(points)
    if n == 0:
        return True
    # Only need to check primes p <= n
    for p in primes_list:
        if p > n:
            break
        # Check if all residues 0..p-1 are covered
        # Use a set for speed if p is small, or boolean array
        residues = set()
        for v in points:
            residues.add(v % p)
            if len(residues) == p:
                return False # All covered, not admissible
    return True

def _repair(points, primes_list, rng):
    # Work with a list
    points = list(points)
    changed = True
    while changed:
        changed = False
        n = len(points)
        if n == 0:
            break
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
                # Find min size
                min_size = min(len(v) for v in classes.values())
                # Find all classes with min size
                candidates = [v for v in classes.values() if len(v) == min_size]
                # Pick one randomly
                to_remove = rng.choice(candidates)
                # Remove
                points = [v for v in points if v not in to_remove]
                changed = True
                break
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    # Initial set: 1 and primes > sqrt(N)
    primes = _primes(N)
    initial_points = [1] + [p for p in primes if p * p > N]
    
    best_result = []
    
    # Run multiple times with different random seeds
    # We can run many iterations since it's fast
    num_runs = 100
    
    for i in range(num_runs):
        if time.time() > deadline:
            break
        rng = random.Random(i)
        result = _repair(initial_points, primes, rng)
        if len(result) > len(best_result):
            best_result = result
            
    return best_result

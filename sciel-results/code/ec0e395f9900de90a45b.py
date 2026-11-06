"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from math import isqrt

# EVOLVE-BLOCK-START
def _prime_factors(n):
    """Return set of prime factors of n."""
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def _objective(arr):
    """Count distinct prime factors across all pairwise sums."""
    all_primes = set()
    n = len(arr)
    for i in range(n):
        for j in range(i+1, n):
            s = arr[i] + arr[j]
            all_primes |= _prime_factors(s)
    return len(all_primes)

def _objective_fast(arr):
    """Faster objective using precomputed factorization cache."""
    all_primes = set()
    n = len(arr)
    for i in range(n):
        for j in range(i+1, n):
            s = arr[i] + arr[j]
            # Inline factorization
            d = 2
            while d * d <= s:
                while s % d == 0:
                    all_primes.add(d)
                    s //= d
                d += 1 if d == 2 else 2
            if s > 1:
                all_primes.add(s)
    return len(all_primes)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    n = 12
    max_val = 10000
    
    # Initial population: try several strategies
    best_arr = None
    best_score = float('inf')
    
    # Strategy 1: Small consecutive integers
    arr1 = list(range(1, 13))
    score1 = _objective(arr1)
    best_arr = arr1[:]
    best_score = score1
    
    # Strategy 2: All even numbers (small)
    arr2 = list(range(2, 25, 2))
    score2 = _objective(arr2)
    if score2 < best_score:
        best_score = score2
        best_arr = arr2[:]
    
    # Strategy 3: Powers of 2
    arr3 = [2**i for i in range(12)]
    if max(arr3) <= max_val:
        score3 = _objective(arr3)
        if score3 < best_score:
            best_score = score3
            best_arr = arr3[:]
    
    # Strategy 4: All multiples of 6 (to get primes 2,3 in sums)
    arr4 = [6*(i+1) for i in range(12)]
    score4 = _objective(arr4)
    if score4 < best_score:
        best_score = score4
        best_arr = arr4[:]
    
    # Strategy 5: All multiples of 30
    arr5 = [30*(i+1) for i in range(12)]
    score5 = _objective(arr5)
    if score5 < best_score:
        best_score = score5
        best_arr = arr5[:]
    
    # Local search / simulated annealing
    random.seed(42)
    current_arr = best_arr[:]
    current_score = best_score
    
    for iteration in range(500000):
        if time.time() > deadline:
            break
        
        # Random perturbation
        new_arr = current_arr[:]
        
        # Choose type of move
        move_type = random.random()
        if move_type < 0.5:
            # Change one element
            idx = random.randint(0, n-1)
            new_val = random.randint(1, max_val)
            if new_val not in new_arr:
                new_arr[idx] = new_val
            else:
                continue
        elif move_type < 0.8:
            # Swap two elements with random values
            i, j = random.sample(range(n), 2)
            v1, v2 = random.sample(range(1, max_val+1), 2)
            if v1 not in new_arr and v2 not in new_arr:
                new_arr[i] = v1
                new_arr[j] = v2
            else:
                continue
        else:
            # Add/subtract small amount to one element
            idx = random.randint(0, n-1)
            delta = random.choice([-2, -1, 1, 2, 3, -3, 5, -5, 10, -10])
            new_val = new_arr[idx] + delta
            if 1 <= new_val <= max_val and new_val not in new_arr:
                new_arr[idx] = new_val
            else:
                continue
        
        new_score = _objective_fast(new_arr)
        
        if new_score < current_score:
            current_arr = new_arr
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_arr = new_arr[:]
        elif new_score <= 6:  # Accept worse solutions if still good
            # Simulated annealing: accept with some probability
            if random.random() < 0.1:
                current_arr = new_arr
                current_score = new_score
    
    # Final local refinement
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            for j in range(i+1, n):
                if time.time() > deadline:
                    break
                # Try swapping values
                trial = best_arr[:]
                trial[i], trial[j] = trial[j], trial[i]
                s = _objective_fast(trial)
                if s < best_score:
                    best_score = s
                    best_arr = trial
                    improved = True
                if improved:
                    break
            if improved:
                break
    
    return best_arr
# EVOLVE-BLOCK-END

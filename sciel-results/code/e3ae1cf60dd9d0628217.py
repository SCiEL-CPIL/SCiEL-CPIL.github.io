import time
import random
import math
import sys

N = 5000
MAX_TIME = 290.0  # Safety margin before 330s kill

# Precompute primes up to N (and beyond for checking)
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

# We need primes up to N+1 because len(S) can be N+1
PRIMES = _primes(N + 1)

def _local_search(seed=0):
    """
    Local search to maximize the size of an admissible tuple.
    """
    rng = random.Random(seed)
    
    # State representation:
    in_set = [False] * (N + 1)
    # counts[p] is a list of size p
    counts = {p: [0] * p for p in PRIMES}
    # zero_counts[p] is the number of residues r where counts[p][r] == 0
    zero_counts = {p: p for p in PRIMES}
    
    S_active = set()
    current_size = 0
    
    def commit_add(x):
        nonlocal current_size
        in_set[x] = True
        S_active.add(x)
        current_size += 1
        for p in PRIMES:
            r = x % p
            if counts[p][r] == 0:
                zero_counts[p] -= 1
            counts[p][r] += 1

    def commit_remove(x):
        nonlocal current_size
        in_set[x] = False
        S_active.remove(x)
        current_size -= 1
        for p in PRIMES:
            r = x % p
            counts[p][r] -= 1
            if counts[p][r] == 0:
                zero_counts[p] += 1

    def can_add(x):
        new_size = current_size + 1
        for p in PRIMES:
            if p > new_size:
                break
            r = x % p
            if counts[p][r] == 0:
                if zero_counts[p] == 1:
                    return False
        return True

    # Greedy Initialization
    available = list(range(N + 1))
    rng.shuffle(available)
    for x in available:
        if can_add(x):
            commit_add(x)
            
    best_size = current_size
    best_set = set(S_active)
    
    start_time = time.time()
    
    # Local Search Loop
    while time.time() - start_time < MAX_TIME - 5:
        # Try to add any element
        added = False
        # To avoid O(N) list creation every time, we can just iterate range
        # But we need to skip in_set elements.
        # Since N is small, list comprehension is fine.
        non_set = [x for x in range(N + 1) if not in_set[x]]
        if not non_set:
            break # Full set? Unlikely for admissible.
        
        rng.shuffle(non_set)
        
        for x in non_set:
            if can_add(x):
                commit_add(x)
                added = True
                if current_size > best_size:
                    best_size = current_size
                    best_set = set(S_active)
                break # Restart checking after an add
        
        if not added:
            # Try to swap
            if time.time() - start_time > MAX_TIME - 10:
                break
                
            if not S_active or not non_set:
                break
                
            # Pick random y from S, x from non-set
            y = rng.choice(list(S_active))
            x = rng.choice(non_set)
            
            if x == y:
                continue
            
            # Check swap validity
            # S' = S \ {y} U {x}
            valid_swap = True
            for p in PRIMES:
                if p > current_size:
                    break
                ry = y % p
                rx = x % p
                c_ry = counts[p][ry]
                c_rx = counts[p][rx]
                z = zero_counts[p]
                
                if ry == rx:
                    new_z = z
                else:
                    new_z = z
                    if c_ry == 1:
                        new_z += 1
                    if c_rx == 0:
                        new_z -= 1
                if new_z == 0:
                    valid_swap = False
                    break
            
            if valid_swap:
                commit_remove(y)
                commit_add(x)
                # Size unchanged, so best_size not updated.
                # But we might enable further adds.
                
    return list(best_set)

def construct():
    """
    Construct the largest admissible tuple in [0, 5000].
    Returns a list of distinct integers.
    """
    start_time = time.time()
    
    best_result = []
    best_len = 0
    
    # Run local search with multiple seeds
    seed = 0
    while time.time() - start_time < MAX_TIME - 5:
        res = _local_search(seed)
        if len(res) > best_len:
            best_len = len(res)
            best_result = res
        seed += 1
        
    return best_result

# EVOLVE-BLOCK-END

import time
import random

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
NUM_PRIMES = len(PRIMES)

def get_vector(n):
    vec = [0] * NUM_PRIMES
    for i, p in enumerate(PRIMES):
        while n % p == 0:
            vec[i] += 1
            n //= p
    return tuple(vec)

CANDIDATES = {n: get_vector(n) for n in range(2, 41)}
ZERO_VEC = tuple([0] * NUM_PRIMES)
ALL_NUMBERS = list(range(2, 41))

# Precompute sum of exponents for heuristic
EXP_SUMS = {n: sum(CANDIDATES[n]) for n in ALL_NUMBERS}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_size = 0
    best_set = []
    
    # Baseline: Primes
    baseline = [p for p in PRIMES if p <= 40]
    best_size = len(baseline)
    best_set = baseline.copy()
    
    # Different ordering strategies
    orderings = []
    for _ in range(5):
        rand_list = ALL_NUMBERS[:]
        random.shuffle(rand_list)
        orderings.append(rand_list)
    
    orderings.append(sorted(ALL_NUMBERS, key=lambda x: EXP_SUMS[x]))
    orderings.append(sorted(ALL_NUMBERS, key=lambda x: -EXP_SUMS[x]))
    
    # Cycle through orderings
    idx = 0
    while time.time() < deadline:
        order = orderings[idx % len(orderings)]
        idx += 1
        
        S_sums = {ZERO_VEC}
        D = {ZERO_VEC}
        current_set = []
        
        for n in order:
            v = CANDIDATES[n]
            if v in D:
                continue
            
            # Update S_sums
            new_sums = set()
            for s in S_sums:
                new_s = tuple(a + b for a, b in zip(s, v))
                new_sums.add(new_s)
            
            # Update D
            new_D_adds = set()
            for d in D:
                new_D_adds.add(tuple(a + b for a, b in zip(d, v)))
                new_D_adds.add(tuple(a - b for a, b in zip(d, v)))
            
            S_sums = S_sums.union(new_sums)
            D = D.union(new_D_adds)
            
            current_set.append(n)
            
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()
            
            if best_size >= 20:
                break

    return best_set

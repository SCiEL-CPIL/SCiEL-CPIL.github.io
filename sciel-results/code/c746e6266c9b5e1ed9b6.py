import time
import random
from itertools import combinations

def is_valid_set(S):
    if not S:
        return True
    
    sums = set()
    for i in range(1, len(S) + 1):
        for comb in combinations(S, i):
            sums.add(sum(comb))
            
    sum_list = sorted(sums)
    n = len(sum_list)
    for i in range(n):
        a = sum_list[i]
        for j in range(i + 1, n):
            b = sum_list[j]
            if b >= 2 * a:
                if b % a == 0:
                    return False
    return True

def greedy_construct(candidates, random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)
    
    S = []
    shuffled = candidates[:]
    random.shuffle(shuffled)
    
    for x in shuffled:
        new_S = S + [x]
        if is_valid_set(new_S):
            S = new_S
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    # Generate candidates
    # Focus on larger numbers to reduce divisibility likelihood
    candidates = []
    for i in range(100, 257):
        # Check if prime
        is_prime = True
        for j in range(2, int(i**0.5) + 1):
            if i % j == 0:
                is_prime = False
                break
        if is_prime:
            candidates.append(i)
            
    all_candidates = candidates + [x for x in range(100, 257) if x not in candidates]
    
    best_set = []
    best_size = 0
    
    attempt = 0
    while time.time() < deadline:
        # Alternate between prime-only and all-candidates
        if attempt % 3 == 0:
            curr_candidates = candidates
        else:
            curr_candidates = all_candidates
            
        S = greedy_construct(curr_candidates)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
        attempt += 1
        
    # Ensure return type is list of ints
    return [int(x) for x in best_set]

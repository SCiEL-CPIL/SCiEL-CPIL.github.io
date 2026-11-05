import random
import time
from itertools import combinations

def is_sidon(subset):
    """Check if a given subset has all unique pair sums including diagonals."""
    sums = [0] * 257
    n = len(subset)
    for i in range(n):
        si = subset[i]
        for j in range(i, n):
            s = si + subset[j]
            if sums[s]:
                return False
            sums[s] = 1
    return True

def max_sidon_size(candidate_set):
    """Find the size of the largest Sidon subset of the given candidate set (exact)."""
    n = len(candidate_set)
    cs = sorted(candidate_set)
    for k in range(n, 0, -1):
        for subset in combinations(cs, k):
            if is_sidon(subset):
                return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    best_set = list(range(12))
    best_size = max_sidon_size(best_set)
    
    # Structured candidates designed to force maximum pair-sum collisions
    candidates = [
        list(range(12)),
        [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15],
        [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13],
        [0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23],
        [0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31, 32],
        [0, 1, 2, 3, 10, 11, 12, 20, 21, 22, 30, 31],
        [0, 1, 2, 5, 6, 7, 10, 11, 12, 15, 16, 17],
        [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14],
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33],
        [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12],
        [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12],
        [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12],
        [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12],
        [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12],
        [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12],
        [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13],
        [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13],
        [0, 1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 14],
        [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13],
        [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13],
        [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13],
        [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 13],
        [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13],
    ]
    
    for c in candidates:
        if len(set(c)) != 12:
            continue
        if any(x < 0 or x > 128 for x in c):
            continue
        sz = max_sidon_size(c)
        if sz < best_size:
            best_size = sz
            best_set = list(c)
            if best_size <= 3:
                break
    
    if best_size <= 3:
        return best_set
    
    # Local search with targeted mutations
    random.seed(42)
    current_set = list(best_set)
    current_size = best_size
    
    def try_mutate(s, sz, budget=500):
        """Try random single-element mutations."""
        best_s = list(s)
        best_sz = sz
        s_set = set(s)
        for _ in range(budget):
            if time.time() >= deadline:
                break
            idx = random.randint(0, 11)
            new_val = random.randint(0, 128)
            if new_val in s_set:
                continue
            new_s = list(s)
            new_s[idx] = new_val
            new_sz = max_sidon_size(new_s)
            if new_sz < best_sz:
                best_sz = new_sz
                best_s = new_s
                s_set = set(new_s)
        return best_s, best_sz
    
    # Simulated annealing style local search
    while time.time() < deadline - 5:
        mutated, mut_sz = try_mutate(current_set, current_size)
        if mut_sz < current_size:
            current_set = mutated
            current_size = mut_sz
            if current_size < best_size:
                best_size = current_size
                best_set = list(current_set)
                if best_size <= 3:
                    break
        elif mut_sz == current_size and random.random() < 0.1:
            # Occasionally accept equal to escape local optima
            current_set = mutated
    
    # Final random restarts if time remains
    if time.time() < deadline - 3:
        while time.time() < deadline - 1:
            s = random.sample(range(129), 12)
            sz = max_sidon_size(s)
            if sz < best_size:
                best_size = sz
                best_set = s
                if best_size <= 3:
                    break
    
    return best_set

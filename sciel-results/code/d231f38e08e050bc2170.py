"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from math import log

# EVOLVE-BLOCK-START

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    sums = set()
    diffs = set()
    for i in range(n):
        for j in range(i, n):
            sums.add(U[i] + U[j])
        for j in range(i + 1, n):
            d = U[j] - U[i]
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    ratio = len(diffs) / len(sums)
    if ratio <= 1:
        return -1
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
    return 1 + log(ratio) / log(D)

def try_random_search(deadline):
    best_score = -1
    best_set = [0, 1, 4, 9, 11]
    
    # Known good sets
    candidates = [
        [0, 1, 3],
        [0, 1, 4, 9, 11],
        [0, 1, 3, 7, 12],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 31],
        [0, 1, 4, 10, 12, 17, 25],
        [0, 1, 2, 4, 8],
        [0, 1, 2, 5, 11],
        [0, 1, 3, 7, 14],
        [0, 1, 3, 8, 13],
        [0, 2, 3, 7, 12],
        [0, 1, 4, 11, 13],
        [0, 1, 4, 9, 13],
    ]
    
    for c in candidates:
        s = score_set(c)
        if s > best_score:
            best_score = s
            best_set = c
    
    # Random local search
    random.seed(42)
    start_time = time.time()
    
    while time.time() < deadline - 5:
        # Start from a random valid set
        n = random.randint(4, 8)
        max_val = random.randint(15, 60)
        
        # Generate a random set with 0 and n-1 other elements
        other_elements = random.sample(range(1, max_val), n - 1)
        current = sorted([0] + other_elements)
        
        current_score = score_set(current)
        if current_score < 0:
            continue
        
        # Local search: try moving one element
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(current)):
                if i == 0:
                    continue  # Keep 0
                old_val = current[i]
                # Try changing to a nearby value
                for delta in [-2, -1, 1, 2]:
                    new_val = old_val + delta
                    if new_val <= 0 or new_val >= max_val:
                        continue
                    new_set = sorted(set([0] + [current[j] if j != i else new_val for j in range(len(current))]))
                    if len(new_set) != len(current):
                        continue
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_set = current[:]
                        break
        
        # Also try adding/removing elements
        if time.time() < deadline - 5:
            # Try adding an element
            if len(current) < 8:
                for new_val in range(1, max_val + 5):
                    if new_val not in current:
                        new_set = sorted(current + [new_val])
                        new_score = score_set(new_set)
                        if new_score > current_score:
                            current = new_set
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best_set = current[:]
                            break
            
            # Try removing an element (not 0)
            if len(current) > 4:
                for i in range(1, len(current)):
                    new_set = [current[j] for j in range(len(current)) if j != i]
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_set = current[:]
                        break
    
    return best_set

def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    best_set = try_random_search(deadline)
    return best_set

# EVOLVE-BLOCK-END

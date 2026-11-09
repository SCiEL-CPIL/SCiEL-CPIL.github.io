"""Optimized program: Exhaustive search for small sets and local search for larger ones."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations
from math import log

def _compute_score(U):
    """Compute the score for a given set U."""
    U = np.array(sorted(set(U)), dtype=np.int64)
    n = len(U)
    if n == 0 or 0 not in U:
        return 0.0
    max_u = U[-1]
    if max_u == 0:
        return 1.0
    
    # Compute U+U and U-U
    sumset = np.unique(U[:, None] + U[None, :])
    diffset = np.unique(U[:, None] - U[None, :])
    len_sum = len(sumset)
    len_diff = len(diffset)
    
    if len_sum == 0:
        return 0.0
    ratio = len_diff / len_sum
    if ratio <= 0:
        return 0.0
        
    denominator = log(2 * max_u + 1)
    if denominator == 0:
        return 1.0
        
    return 1 + log(ratio) / denominator

def _local_search(U, max_iterations=5000, timeout=10.0):
    """Local search: add/remove/swap elements to improve score."""
    start_time = time.time()
    current = sorted(set(U))
    if 0 not in current:
        current.insert(0, 0)
    current_score = _compute_score(current)
    best = list(current)
    best_score = current_score
    
    for i in range(max_iterations):
        if time.time() - start_time > timeout:
            break
        
        U_set = set(current)
        max_u = max(U_set) if U_set else 0
        n = len(U_set)
        
        # Generate candidates
        candidates = set()
        # Perturbations
        for x in list(U_set):
            for off in [-2, -1, 1, 2, 3, 5]:
                c = x + off
                if 0 <= c <= 100: # Limit search space for small sets
                    candidates.add(c)
        # Sums/Diffs
        elems = list(U_set)
        for a in elems:
            for b in elems:
                s = a + b
                if s <= 100:
                    candidates.add(s)
                d = abs(a - b)
                if 0 < d <= 100:
                    candidates.add(d)
        # Random
        for _ in range(10):
            candidates.add(random.randint(0, 100))
        # Boundary
        if max_u > 0:
            candidates.add(max_u + 1)
            candidates.add(max_u * 2)
            
        candidate_list = list(candidates)
        random.shuffle(candidate_list)
        
        improved = False
        for cand in candidate_list:
            if time.time() - start_time > timeout:
                break
            if cand in U_set:
                continue
            new_set = U_set | {cand}
            new_score = _compute_score(new_set)
            if new_score > current_score + 1e-10:
                current = sorted(new_set)
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best = list(current)
                improved = True
                break
        
        if not improved:
            # Try removals
            removable = [x for x in current if x != 0]
            random.shuffle(removable)
            for cand in removable[:10]:
                if time.time() - start_time > timeout:
                    break
                new_set = U_set - {cand}
                if len(new_set) < 2:
                    continue
                new_score = _compute_score(new_set)
                if new_score > current_score + 1e-10:
                    current = sorted(new_set)
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best = list(current)
                    improved = True
                    break
        
        if not improved and i > 10:
            # Swap
            if random.random() < 0.5:
                removable = [x for x in current if x != 0]
                if removable:
                    rem = random.choice(removable)
                    new_elem = random.randint(0, 100)
                    new_set = (U_set - {rem}) | {new_elem}
                    if len(new_set) >= 2:
                        new_score = _compute_score(new_set)
                        if new_score > current_score + 1e-10:
                            current = sorted(new_set)
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best = list(current)
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    best_result = [0, 1, 2]
    best_score = _compute_score(best_result)
    
    # 1. Exhaustive search for small sets (n=3 to 6) within a bounded range
    # This is fast and likely to find the global optimum for small n
    max_val = 30
    for n in range(3, 7):
        if time.time() > deadline - 10:
            break
        # Use combinations for exact search
        # To speed up, we can skip if n is large and max_val is large, but here max_val is small
        for combo in combinations(range(1, max_val + 1), n - 1):
            U = [0] + list(combo)
            score = _compute_score(U)
            if score > best_score:
                best_score = score
                best_result = U
    
    # 2. Local search from the best found
    if time.time() < deadline - 5:
        improved, score = _local_search(best_result, max_iterations=10000, timeout=min(15.0, deadline - time.time() - 2))
        if score > best_score:
            best_score = score
            best_result = improved
            
    # 3. Additional random restarts for local search to ensure we don't miss a better local optimum
    if time.time() < deadline - 5:
        for _ in range(20):
            if time.time() > deadline - 2:
                break
            # Generate a random small set
            n = random.randint(3, 6)
            rand_set = {0}
            while len(rand_set) < n:
                rand_set.add(random.randint(1, 50))
            rand_list = sorted(rand_set)
            improved, score = _local_search(rand_list, max_iterations=2000, timeout=min(5.0, deadline - time.time() - 1))
            if score > best_score:
                best_score = score
                best_result = improved

    # Final check
    final_score = _compute_score(best_result)
    if final_score < 1.0:
        best_result = [0, 1, 2, 4]
        
    return best_result
# EVOLVE-BLOCK-END

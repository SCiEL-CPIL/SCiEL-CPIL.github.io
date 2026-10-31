"""Dual-view construction: define U via Sidon-like difference structure, minimize additive energy."""
import numpy as np
import math
import random

def construct():
    """
    Construct U using dual-view approach:
    - Start with Bose-Chowla Sidon sets (optimal max U for size N)
    - Use local search to minimize |U+U| while preserving |U-U|
    - Score = 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)
    """
    
    def compute_score(U):
        if not U or 0 not in U:
            return 0.0
        U_set = set(U)
        max_u = max(U_set)
        if max_u == 0:
            return 0.0
        sum_set = set()
        diff_set = set()
        for a in U_set:
            for b in U_set:
                sum_set.add(a + b)
                diff_set.add(a - b)
        n_diff = len(diff_set)
        n_sum = len(sum_set)
        if n_sum == 0:
            return 0.0
        log_denom = math.log(2 * max_u + 1)
        if log_denom <= 0:
            return 0.0
        return 1.0 + math.log(n_diff / n_sum) / log_denom

    def local_search(U, max_iters=2000, max_val=10000):
        """Enhanced local search: try swapping, adding, removing, and 2-swap moves."""
        U = sorted(set(U))
        if 0 not in U:
            U = sorted([0] + U)
        
        best_U = list(U)
        best_score = compute_score(U)
        
        # Precompute sets for fast collision checking if possible, 
        # but N is small (<50), so O(N^2) per eval is ~2500 ops. 
        # 2000 iters * 2500 = 5M ops. Fast enough.
        
        for iteration in range(max_iters):
            improved = False
            U_set = set(best_U)
            n = len(best_U)
            
            # 1. Try single element moves (shift)
            for i, elem in enumerate(best_U):
                if elem == 0:
                    continue
                # Try a range of shifts. 
                # We want to try values that might reduce sum collisions.
                # Just try local neighborhood first.
                search_range = 5 + (iteration % 10) # Expand range over time
                for delta in range(-search_range, search_range + 1):
                    if delta == 0: continue
                    new_val = elem + delta
                    if new_val < 0 or new_val in U_set:
                        continue
                    if new_val > max_val:
                        continue
                    new_U = list(best_U)
                    new_U[i] = new_val
                    new_U = sorted(set(new_U))
                    if 0 not in new_U or len(new_U) < 2:
                        continue
                    new_score = compute_score(new_U)
                    if new_score > best_score + 1e-10:
                        best_score = new_score
                        best_U = new_U
                        improved = True
                        break
                if improved:
                    break
            
            # 2. Try swapping two elements
            if not improved and n >= 3:
                # Randomly pick pairs to avoid O(N^2 * Range^2) explosion
                for _ in range(10):
                    i, j = random.sample(range(1, n), 2)
                    e1, e2 = best_U[i], best_U[j]
                    for d1 in range(-3, 4):
                        for d2 in range(-3, 4):
                            new_v1 = e1 + d1
                            new_v2 = e2 + d2
                            if new_v1 < 0 or new_v2 < 0:
                                continue
                            if new_v1 > max_val or new_v2 > max_val:
                                continue
                            U_set_check = set(best_U)
                            if new_v1 in U_set_check or new_v2 in U_set_check:
                                continue
                            if new_v1 == new_v2:
                                continue
                            new_U = list(best_U)
                            new_U[i] = new_v1
                            new_U[j] = new_v2
                            new_U = sorted(set(new_U))
                            if 0 not in new_U or len(new_U) < 2:
                                continue
                            new_score = compute_score(new_U)
                            if new_score > best_score + 1e-10:
                                best_score = new_score
                                best_U = new_U
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            # 3. Try adding a new element
            if not improved:
                # Sample candidates from a wider range
                max_candidate = min(max_val, max(best_U) * 2 + 100)
                for _ in range(20):
                    candidate = random.randint(1, max_candidate)
                    if candidate in U_set:
                        continue
                    new_U = sorted(best_U + [candidate])
                    new_score = compute_score(new_U)
                    if new_score > best_score + 1e-10:
                        best_score = new_score
                        best_U = new_U
                        improved = True
                        break
            
            # 4. Try removing an element
            if not improved and len(best_U) > 5:
                # Remove element with highest "cost" (hardest to fit)
                # Just try removing random non-zero elements
                for _ in range(5):
                    i = random.randint(1, len(best_U) - 1)
                    new_U = [x for j, x in enumerate(best_U) if j != i]
                    new_score = compute_score(new_U)
                    if new_score > best_score + 1e-10:
                        best_score = new_score
                        best_U = new_U
                        improved = True
                        break
            
            if not improved:
                # If stuck, try a larger random jump
                if iteration > 100:
                    i = random.randint(1, len(best_U) - 1)
                    jump = random.randint(10, 50) * random.choice([-1, 1])
                    new_val = best_U[i] + jump
                    if 0 < new_val < max_val and new_val not in set(best_U):
                        new_U = list(best_U)
                        new_U[i] = new_val
                        new_U = sorted(set(new_U))
                        new_score = compute_score(new_U)
                        if new_score > best_score + 1e-10:
                            best_score = new_score
                            best_U = new_U
                            improved = True
                if not improved:
                    break
        
        return best_U, best_score

    def bose_chowla(p):
        """
        Construct a Bose-Chowla Sidon set of size p in [0, p^2).
        U = { x + x^2 * p | x in 0..p-1 }
        Note: x=0 gives 0.
        """
        U = []
        for x in range(p):
            val = x + (x * x) * p
            U.append(val)
        return sorted(U)

    candidates = []
    
    # 1. Bose-Chowla Sidon sets for various primes
    primes = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
    for p in primes:
        U = bose_chowla(p)
        candidates.append(U)
        
    # 2. Mian-Chowla style greedy (known good for small N)
    def greedy_sidon(n_target, max_val=10000):
        U = [0]
        diffs = set([0])
        for _ in range(n_target - 1):
            found = False
            for candidate in range(1, max_val):
                # Check if candidate creates duplicate differences
                ok = True
                for u in U:
                    d = candidate - u
                    if d in diffs:
                        ok = False
                        break
                if ok:
                    # Also check d = u - candidate (which is -d)
                    # Since diffs contains all a-b, if d is in diffs, -d is also in diffs?
                    # If we store all a-b, yes.
                    U.append(candidate)
                    for u in U[:-1]:
                        diffs.add(candidate - u)
                        diffs.add(u - candidate)
                    found = True
                    break
            if not found:
                break
        return U

    for n in [10, 12, 15, 18, 20]:
        U = greedy_sidon(n, max_val=5000)
        if len(U) >= n - 2: # Allow slight failure
            candidates.append(U)

    # 3. Random perturbations of Bose-Chowla
    random.seed(123)
    for p in [11, 13, 17]:
        base = bose_chowla(p)
        for _ in range(5):
            U = list(base)
            # Perturb non-zero elements
            for i in range(1, len(U)):
                if random.random() < 0.3:
                    U[i] += random.randint(-5, 5)
            U = sorted(set([x for x in U if x >= 0]))
            if 0 not in U:
                U.insert(0, 0)
            if len(U) >= 5:
                candidates.append(U)

    # Run local search on each candidate
    best_U = [0, 1]
    best_score = 0.0
    
    for U in candidates:
        U = sorted(set(U))
        if 0 not in U:
            U = [0] + U
        if len(U) < 2:
            continue
        searched_U, searched_score = local_search(U, max_iters=1000, max_val=10000)
        if searched_score > best_score:
            best_score = searched_score
            best_U = searched_U
            
    return best_U

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END

"""Dual-view construction: define U via Sidon-like difference structure, minimize additive energy."""
import numpy as np
import math
import random

def construct():
    """
    Construct U using dual-view approach:
    - Start with Sidon-like structures (maximize distinct differences)
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
    
    def local_search(U, max_iters=500, max_val=600):
        """Enhanced local search: try swapping, adding, removing, and 2-swap moves."""
        U = sorted(set(U))
        if 0 not in U:
            U = sorted([0] + U)
        
        best_U = list(U)
        best_score = compute_score(U)
        
        for iteration in range(max_iters):
            improved = False
            U_set = set(best_U)
            n = len(best_U)
            
            # Try swapping each element (except 0) with a wider range
            for i, elem in enumerate(best_U):
                if elem == 0:
                    continue
                # Wider search range
                for delta in range(-20, 21):
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
            
            # Try 2-swap: swap two elements simultaneously
            if not improved and n >= 3:
                for i in range(1, n):
                    for j in range(i + 1, n):
                        e1, e2 = best_U[i], best_U[j]
                        for d1 in range(-5, 6):
                            for d2 in range(-5, 6):
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
                    if improved:
                        break
            
            # Try adding a new element (sample candidates for speed)
            if not improved:
                U_set = set(best_U)
                # Try a subset of candidates for speed
                candidates_to_try = list(range(1, min(max_val + 1, 200)))
                random.shuffle(candidates_to_try)
                for candidate in candidates_to_try[:50]:
                    if candidate in U_set:
                        continue
                    new_U = sorted(best_U + [candidate])
                    new_score = compute_score(new_U)
                    if new_score > best_score + 1e-10:
                        best_score = new_score
                        best_U = new_U
                        improved = True
                        break
            
            # Try removing an element (except 0)
            if not improved:
                for i, elem in enumerate(best_U):
                    if elem == 0:
                        continue
                    new_U = [x for j, x in enumerate(best_U) if j != i]
                    if len(new_U) < 2:
                        continue
                    new_score = compute_score(new_U)
                    if new_score > best_score + 1e-10:
                        best_score = new_score
                        best_U = new_U
                        improved = True
                        break
            
            if not improved:
                break
        
        return best_U, best_score
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Known Sidon-like sequences (Mian-Chowla style)
    candidates.append([0, 1, 4, 10, 18, 23, 31, 40, 50, 61, 73, 86, 100, 115, 131, 148])
    candidates.append([0, 1, 4, 10, 18, 23, 31, 40, 50, 61, 73, 86, 100, 115])
    candidates.append([0, 1, 4, 10, 18, 23, 31, 40, 50, 61, 73, 86])
    candidates.append([0, 1, 4, 10, 18, 23, 31, 40, 50, 61])
    candidates.append([0, 1, 3, 7, 12, 20, 30, 44, 60, 79, 101])
    
    # 2. Perturbed arithmetic progressions with varied structures
    for base in [6, 8, 10, 12, 14, 16, 18]:
        for n in range(4, 13):
            U = [i * base for i in range(n)]
            # Multiple perturbation strategies
            strategy = random.randint(0, 2)
            if strategy == 0:
                for i in range(1, len(U)):
                    U[i] += (i % 2) * 1 + (i % 3)
            elif strategy == 1:
                for i in range(1, len(U)):
                    U[i] += (i % 3) * 2
            else:
                for i in range(1, len(U)):
                    U[i] += random.randint(0, 3)
            U = sorted(set(U))
            if 0 not in U:
                U = [0] + U
            if len(U) >= 2:
                candidates.append(U)
    
    # 3. Power-of-2 based and other structured sets
    candidates.append([0] + [2**i for i in range(1, 10)])
    candidates.append([0, 1, 3, 7, 15, 31, 63, 127, 255])
    candidates.append([0, 1, 2, 5, 10, 20, 40, 80, 160])
    candidates.append([0, 1, 4, 16, 64, 256])
    candidates.append([0, 2, 4, 8, 16, 32, 64, 128])
    
    # 4. Random constructions with Sidon bias (improved)
    random.seed(42)
    for trial in range(25):
        n = random.randint(5, 16)
        max_val = random.randint(100, 500)
        U = [0]
        U_set = {0}
        diff_counts = {}
        for _ in range(n - 1):
            best_candidate = None
            best_collision = float('inf')
            best_score_candidate = float('-inf')
            for _ in range(100):
                candidate = random.randint(1, max_val)
                if candidate in U_set:
                    continue
                # Count how many new differences collide with existing ones
                collisions = 0
                sum_collisions = 0
                for a in U_set:
                    d1 = candidate - a
                    d2 = a - candidate
                    if d1 in diff_counts:
                        collisions += diff_counts[d1]
                    if d2 in diff_counts:
                        collisions += diff_counts[d2]
                    # Also count sum collisions
                    s = candidate + a
                    # We'll estimate sum collisions by checking if s is already a sum
                    # For speed, just use difference collisions
                # Prefer candidates with fewer collisions and larger values (to spread out)
                score = -collisions * 10 + candidate * 0.1
                if score > best_score_candidate:
                    best_score_candidate = score
                    best_collision = collisions
                    best_candidate = candidate
            if best_candidate is not None:
                U.append(best_candidate)
                U_set.add(best_candidate)
                # Update diff counts
                for a in U_set:
                    if a == best_candidate:
                        continue
                    d1 = best_candidate - a
                    d2 = a - best_candidate
                    diff_counts[d1] = diff_counts.get(d1, 0) + 1
                    diff_counts[d2] = diff_counts.get(d2, 0) + 1
        U = sorted(U)
        candidates.append(U)
    
    # 5. Greedy construction: build U to maximize |U-U|/|U+U| ratio
    random.seed(77)
    for trial in range(10):
        max_val = random.randint(200, 400)
        U = [0]
        U_set = {0}
        for _ in range(12):
            best_candidate = None
            best_ratio = float('-inf')
            for candidate in range(1, max_val + 1, 3):  # Step by 3 for speed
                if candidate in U_set:
                    continue
                new_U = sorted(U + [candidate])
                # Quick score estimate
                new_score = compute_score(new_U)
                if new_score > best_ratio:
                    best_ratio = new_score
                    best_candidate = candidate
            if best_candidate is not None:
                U.append(best_candidate)
                U_set.add(best_candidate)
        U = sorted(U)
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
        searched_U, searched_score = local_search(U, max_iters=400, max_val=500)
        if searched_score > best_score:
            best_score = searched_score
            best_U = searched_U
    
    return best_U

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END

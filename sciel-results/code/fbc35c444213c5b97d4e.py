"""Dual-view construction: define U via Sidon-like difference structure, minimize additive energy."""
import numpy as np
import math
import random

def construct():
    """
    Construct U using dual-view approach:
    - Start with structured sets (APs) that minimize |U+U|
    - Use difference-set-guided perturbations to increase |U-U|
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

    def fast_local_search(U, max_iters=100, max_val=200):
        """Optimized local search."""
        U = sorted(set(U))
        if 0 not in U:
            U = [0] + U
        best_U = list(U)
        best_score = compute_score(U)
        
        for _ in range(max_iters):
            improved = False
            U_set = set(best_U)
            n = len(best_U)
            
            # Try moving each element
            for i in range(1, n):
                elem = best_U[i]
                best_new_val = None
                best_new_score = best_score
                
                for new_val in range(max(1, elem - 10), min(max_val + 1, elem + 11)):
                    if new_val == elem or new_val in U_set:
                        continue
                    new_U = best_U.copy()
                    new_U[i] = new_val
                    new_U = sorted(set(new_U))
                    if 0 not in new_U:
                        continue
                    new_score = compute_score(new_U)
                    if new_score > best_new_score:
                        best_new_score = new_score
                        best_new_val = new_val
                
                if best_new_val is not None:
                    best_U[i] = best_new_val
                    best_U = sorted(set(best_U))
                    best_score = best_new_score
                    improved = True
                    break
            
            # Try adding
            if not improved:
                best_candidate = None
                best_candidate_score = best_score
                # Sample from range
                for candidate in random.sample(range(1, max_val + 1), min(50, max_val)):
                    if candidate in U_set:
                        continue
                    new_U = sorted(best_U + [candidate])
                    new_score = compute_score(new_U)
                    if new_score > best_candidate_score:
                        best_candidate_score = new_score
                        best_candidate = candidate
                
                if best_candidate is not None:
                    best_U = sorted(best_U + [best_candidate])
                    best_score = best_candidate_score
                    improved = True
            
            # Try removing
            if not improved and n > 2:
                for i in range(1, n):
                    new_U = [x for j, x in enumerate(best_U) if j != i]
                    new_score = compute_score(new_U)
                    if new_score > best_score:
                        best_U = new_U
                        best_score = new_score
                        improved = True
                        break
            
            if not improved:
                break
        
        return best_U, best_score

    candidates = []
    
    # Generate specific high-ratio structures
    # 1. Small sets with high ratio (e.g., Mian-Chowla like)
    mc_seqs = [
        [0, 1, 4, 10, 18, 23, 31, 40, 50, 61, 73, 86, 100],
        [0, 1, 3, 7, 12, 20, 30, 44, 60, 79, 99, 122, 148],
        [0, 1, 4, 10, 18, 23, 31, 40, 50, 61, 73, 86],
        [0, 1, 3, 7, 12, 20, 30, 44, 60],
        [0, 1, 4, 10, 18, 23, 31, 40],
        [0, 1, 3, 7, 12, 20, 30],
        [0, 1, 4, 10, 18, 23],
        [0, 1, 3, 7, 12, 20],
        [0, 1, 4, 10, 18],
        [0, 1, 3, 7],
        [0, 1, 4],
        [0, 1, 3],
        [0, 1, 2],
        [0, 1]
    ]
    candidates.extend(mc_seqs)
    
    # 2. Perturbed APs
    random.seed(42)
    for base in [2, 3, 4, 5, 6]:
        for n in range(3, 15):
            U = [i * base for i in range(n)]
            for i in range(1, len(U)):
                U[i] += random.randint(-2, 2)
            U = sorted(set(U))
            if 0 not in U: U = [0] + U
            if len(U) >= 2: candidates.append(U)
            
    # 3. Two-cluster sets
    for n1 in [2, 3, 4, 5]:
        for n2 in [2, 3, 4, 5]:
            for gap in [5, 10, 20, 50, 100]:
                U1 = list(range(n1))
                U2 = [gap + i for i in range(n2)]
                U = sorted(U1 + U2)
                if 0 not in U: U = [0] + U
                candidates.append(U)
                
    # Run local search on a subset of candidates to save time
    best_U = [0, 1]
    best_score = 0.0
    
    # Prioritize smaller sets and known good structures
    random.shuffle(candidates)
    # Put known good ones at the front
    good_indices = [i for i, c in enumerate(candidates) if len(c) < 10]
    for idx in good_indices:
        candidates.insert(0, candidates.pop(idx))
        
    for U in candidates[:50]:
        U = sorted(set(U))
        if 0 not in U:
            U = [0] + U
        if len(U) < 2:
            continue
        searched_U, searched_score = fast_local_search(U, max_iters=50, max_val=150)
        if searched_score > best_score:
            best_score = searched_score
            best_U = searched_U
            
    return best_U

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END

"""Dual-view construction: define U via Sidon-like difference structure, minimize additive energy."""
import numpy as np
import math
import random
import time

def construct():
    """
    Construct U using dual-view approach:
    - Define U through its difference set structure (Sidon-like)
    - Minimize |U+U| while preserving |U-U|
    - Score = 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)
    """
    
    def compute_score(U):
        """Compute the score efficiently."""
        if not U or 0 not in U:
            return 0.0
        U_set = set(U)
        max_u = max(U_set)
        if max_u == 0:
            return 0.0
        
        # Compute sum and difference sets
        sum_set = set()
        diff_set = set()
        n = len(U_set)
        U_list = sorted(U_set)
        
        for i in range(n):
            for j in range(n):
                sum_set.add(U_list[i] + U_list[j])
                diff_set.add(U_list[i] - U_list[j])
        
        n_diff = len(diff_set)
        n_sum = len(sum_set)
        if n_sum == 0:
            return 0.0
        
        log_denom = math.log(2 * max_u + 1)
        if log_denom <= 0:
            return 0.0
        
        return 1.0 + math.log(n_diff / n_sum) / log_denom
    
    def compute_score_fast(U, sum_set_cache=None, diff_set_cache=None):
        """Compute score with optional caching for speed."""
        if not U or 0 not in U:
            return 0.0
        U_set = set(U)
        max_u = max(U_set)
        if max_u == 0:
            return 0.0
        
        n = len(U_set)
        U_list = sorted(U_set)
        
        sum_set = set()
        diff_set = set()
        
        for i in range(n):
            ui = U_list[i]
            for j in range(n):
                uj = U_list[j]
                sum_set.add(ui + uj)
                diff_set.add(ui - uj)
        
        n_diff = len(diff_set)
        n_sum = len(sum_set)
        if n_sum == 0:
            return 0.0
        
        log_denom = math.log(2 * max_u + 1)
        if log_denom <= 0:
            return 0.0
        
        return 1.0 + math.log(n_diff / n_sum) / log_denom
    
    def local_search(U, max_iters=300, max_val=400):
        """
        Enhanced local search with multiple move types:
        1. Single element move (shift)
        2. Two-element swap
        3. Add/remove element
        4. Structural moves based on difference set analysis
        """
        U = sorted(set(U))
        if 0 not in U:
            U = sorted([0] + U)
        
        best_U = list(U)
        best_score = compute_score(U)
        no_improve_count = 0
        
        for iteration in range(max_iters):
            improved = False
            U_set = set(best_U)
            n = len(best_U)
            
            # Move 1: Try shifting each non-zero element
            for i in range(1, n):
                elem = best_U[i]
                # Try a range of shifts
                for delta in range(-15, 16):
                    if delta == 0:
                        continue
                    new_val = elem + delta
                    if new_val < 0 or new_val > max_val:
                        continue
                    if new_val in U_set and new_val != elem:
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
            
            # Move 2: Try two-element swap
            if not improved and n >= 3:
                for i in range(1, n):
                    for j in range(i + 1, n):
                        e1, e2 = best_U[i], best_U[j]
                        # Try limited shifts for each
                        for d1 in range(-8, 9, 2):
                            for d2 in range(-8, 9, 2):
                                if d1 == 0 and d2 == 0:
                                    continue
                                new_v1 = e1 + d1
                                new_v2 = e2 + d2
                                if new_v1 < 0 or new_v2 < 0:
                                    continue
                                if new_v1 > max_val or new_v2 > max_val:
                                    continue
                                
                                U_set_check = set(best_U)
                                # Remove original values for check
                                U_set_check.discard(e1)
                                U_set_check.discard(e2)
                                
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
            
            # Move 3: Try adding an element
            if not improved:
                U_set = set(best_U)
                max_current = max(best_U)
                # Try candidates in a smart range
                candidates = []
                # Try values that create good difference structure
                for candidate in range(1, max_val + 1):
                    if candidate in U_set:
                        continue
                    # Quick heuristic: prefer values that create new differences
                    new_diffs = 0
                    for a in U_set:
                        d = candidate - a
                        if d not in {x - y for x in U_set for y in U_set}:
                            new_diffs += 1
                    candidates.append((new_diffs, candidate))
                
                # Sort by new differences (descending) and try top candidates
                candidates.sort(reverse=True)
                for new_diffs, candidate in candidates[:30]:
                    new_U = sorted(best_U + [candidate])
                    new_score = compute_score(new_U)
                    if new_score > best_score + 1e-10:
                        best_score = new_score
                        best_U = new_U
                        improved = True
                        break
            
            # Move 4: Try removing an element
            if not improved:
                for i in range(1, n):
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
                no_improve_count += 1
                if no_improve_count >= 10:
                    break
            else:
                no_improve_count = 0
        
        return best_U, best_score
    
    # Generate diverse initial candidates using dual-view approach
    
    candidates = []
    
    # 1. Mian-Chowla sequence (Sidon sequence) - maximizes distinct differences
    mian_chowla = [0, 1, 4, 10, 18, 23, 31, 40, 50, 61, 73, 86, 100, 115, 131, 148, 166, 187, 209, 232]
    for length in range(8, 21):
        candidates.append(mian_chowla[:length])
    
    # 2. Other known Sidon sequences
    candidates.append([0, 1, 3, 7, 12, 20, 30, 44, 60, 79, 101, 126, 154, 185])
    candidates.append([0, 1, 2, 5, 10, 20, 40, 80, 160, 320])
    candidates.append([0, 1, 4, 16, 64, 256, 1024])
    candidates.append([0, 1, 3, 7, 15, 31, 63, 127, 255])
    candidates.append([0, 2, 4, 8, 16, 32, 64, 128, 256])
    
    # 3. Construct Sidon-like sequences greedily
    random.seed(123)
    for trial in range(20):
        n_target = random.randint(8, 18)
        max_val = random.randint(150, 350)
        
        U = [0]
        U_set = {0}
        existing_diffs = set()
        
        for _ in range(n_target - 1):
            best_candidate = None
            best_score_candidate = float('-inf')
            
            # Try many candidates
            for _ in range(200):
                candidate = random.randint(1, max_val)
                if candidate in U_set:
                    continue
                
                # Count new differences
                new_diffs = 0
                new_sums = 0
                for a in U_set:
                    d1 = candidate - a
                    d2 = a - candidate
                    if d1 not in existing_diffs:
                        new_diffs += 1
                    if d2 not in existing_diffs:
                        new_diffs += 1
                    # Estimate sum collisions
                    s = candidate + a
                    # For speed, just count potential new sums
                    new_sums += 1
                
                # Score: prefer more new differences, penalize large values slightly
                score = new_diffs * 10 - candidate * 0.05
                if score > best_score_candidate:
                    best_score_candidate = score
                    best_candidate = candidate
            
            if best_candidate is not None:
                U.append(best_candidate)
                U_set.add(best_candidate)
                # Update existing differences
                for a in U_set:
                    existing_diffs.add(best_candidate - a)
                    existing_diffs.add(a - best_candidate)
        
        U = sorted(U)
        candidates.append(U)
    
    # 4. Arithmetic progression based with perturbations
    for base in [5, 7, 9, 11, 13]:
        for n in range(6, 15):
            U = [i * base for i in range(n)]
            # Add small perturbations
            for i in range(1, len(U)):
                U[i] += random.randint(-1, 2)
            U = sorted(set(U))
            if 0 not in U:
                U = [0] + U
            if len(U) >= 4:
                candidates.append(U)
    
    # 5. Power-law sequences
    for base in [2, 3]:
        for n in range(6, 12):
            U = [0] + [int(base ** (i * 0.7)) for i in range(1, n)]
            U = sorted(set(U))
            if len(U) >= 4:
                candidates.append(U)
    
    # Run local search on each candidate
    best_U = [0, 1]
    best_score = 0.0
    
    # Sort candidates by initial score to prioritize good starting points
    scored_candidates = []
    for U in candidates:
        U = sorted(set(U))
        if 0 not in U:
            U = [0] + U
        if len(U) < 3:
            continue
        score = compute_score(U)
        scored_candidates.append((score, U))
    
    scored_candidates.sort(reverse=True)
    
    # Run local search on top candidates
    for score, U in scored_candidates[:15]:
        searched_U, searched_score = local_search(U, max_iters=250, max_val=350)
        if searched_score > best_score:
            best_score = searched_score
            best_U = searched_U
    
    return best_U

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END

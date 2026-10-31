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
    
    def compute_diff_sum_stats(U):
        """Compute |U-U| and |U+U| efficiently."""
        U_set = set(U)
        sum_set = set()
        diff_set = set()
        for a in U_set:
            for b in U_set:
                sum_set.add(a + b)
                diff_set.add(a - b)
        return len(diff_set), len(sum_set), max(U_set) if U_set else 0
    
    def local_search_diff_guided(U, max_iters=300, max_val=400):
        """Local search guided by difference set structure.
        
        Key idea: when moving an element, check if it creates new differences
        without creating too many new sums. Prioritize moves that add differences
        over moves that add sums.
        """
        U = sorted(set(U))
        if 0 not in U:
            U = sorted([0] + U)
        
        best_U = list(U)
        best_score = compute_score(U)
        
        for iteration in range(max_iters):
            improved = False
            U_set = set(best_U)
            n = len(best_U)
            
            # Precompute current diff and sum sets for quick evaluation
            cur_sum_set = set()
            cur_diff_set = set()
            for a in U_set:
                for b in U_set:
                    cur_sum_set.add(a + b)
                    cur_diff_set.add(a - b)
            
            # Try moving each element (except 0) with diff-guided scoring
            for i, elem in enumerate(best_U):
                if elem == 0:
                    continue
                
                best_new_val = None
                best_new_score = best_score
                best_new_diff = len(cur_diff_set)
                best_new_sum = len(cur_sum_set)
                
                # Search a range of new values
                for new_val in range(max(1, elem - 15), min(max_val + 1, elem + 16)):
                    if new_val == elem or new_val in U_set:
                        continue
                    
                    # Estimate new diff and sum counts
                    new_diff_count = 0
                    new_sum_count = 0
                    
                    # New differences involving new_val
                    for a in U_set:
                        if a == elem:
                            continue
                        d1 = new_val - a
                        d2 = a - new_val
                        if d1 not in cur_diff_set:
                            new_diff_count += 1
                        if d2 not in cur_diff_set:
                            new_diff_count += 1
                        s = new_val + a
                        if s not in cur_sum_set:
                            new_sum_count += 1
                    
                    # Also need to account for differences/sums that only involved elem
                    # This is complex, so use a heuristic score
                    # Prefer moves that add more diffs than sums
                    net_gain = new_diff_count - new_sum_count * 0.8
                    
                    if net_gain > 0:
                        # Do full score calculation for promising moves
                        new_U = list(best_U)
                        new_U[i] = new_val
                        new_U = sorted(set(new_U))
                        if 0 not in new_U or len(new_U) < 2:
                            continue
                        new_score = compute_score(new_U)
                        if new_score > best_new_score + 1e-10:
                            best_new_score = new_score
                            best_new_val = new_val
                
                if best_new_val is not None:
                    best_U[i] = best_new_val
                    best_U = sorted(set(best_U))
                    best_score = best_new_score
                    improved = True
                    break
            
            # Try adding a new element with diff-guided selection
            if not improved:
                U_set = set(best_U)
                cur_sum_set = set()
                cur_diff_set = set()
                for a in U_set:
                    for b in U_set:
                        cur_sum_set.add(a + b)
                        cur_diff_set.add(a - b)
                
                best_candidate = None
                best_candidate_score = best_score
                
                # Sample candidates
                candidates = list(range(1, min(max_val + 1, 200)))
                random.shuffle(candidates)
                
                for candidate in candidates[:80]:
                    if candidate in U_set:
                        continue
                    
                    # Estimate new diffs and sums
                    new_diff_count = 0
                    new_sum_count = 0
                    for a in U_set:
                        d1 = candidate - a
                        d2 = a - candidate
                        if d1 not in cur_diff_set:
                            new_diff_count += 1
                        if d2 not in cur_diff_set:
                            new_diff_count += 1
                        s = candidate + a
                        if s not in cur_sum_set:
                            new_sum_count += 1
                    
                    net_gain = new_diff_count - new_sum_count * 0.7
                    if net_gain > 0:
                        new_U = sorted(best_U + [candidate])
                        new_score = compute_score(new_U)
                        if new_score > best_candidate_score + 1e-10:
                            best_candidate_score = new_score
                            best_candidate = candidate
                
                if best_candidate is not None:
                    best_U = sorted(best_U + [best_candidate])
                    best_score = best_candidate_score
                    improved = True
            
            # Try removing an element
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
            
            # Try 2-element swap
            if not improved and n >= 3:
                for i in range(1, n):
                    for j in range(i + 1, n):
                        e1, e2 = best_U[i], best_U[j]
                        for d1 in range(-8, 9):
                            for d2 in range(-8, 9):
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
            
            if not improved:
                break
        
        return best_U, best_score
    
    # Generate diverse initial candidates focused on structured sets
    candidates = []
    
    # 1. Arithmetic progressions with perturbations (start structured, then diversify)
    for base in [4, 5, 6, 7, 8, 9, 10, 12, 14]:
        for n in range(5, 18):
            U = [i * base for i in range(n)]
            # Light perturbation to break perfect AP structure
            perturbation_strength = random.randint(1, 3)
            for i in range(1, len(U)):
                U[i] += random.randint(-perturbation_strength, perturbation_strength)
            U = sorted(set(U))
            if 0 not in U:
                U = [0] + U
            if len(U) >= 2:
                candidates.append(U)
    
    # 2. Geometric-like sequences (good for large max, controlled sums)
    candidates.append([0, 1, 3, 9, 27, 81, 243])
    candidates.append([0, 1, 3, 7, 15, 31, 63, 127])
    candidates.append([0, 1, 4, 16, 64, 256])
    candidates.append([0, 2, 5, 12, 29, 66, 149])
    
    # 3. Mian-Chowla / Sidon sequences
    candidates.append([0, 1, 4, 10, 18, 23, 31, 40, 50, 61, 73, 86, 100])
    candidates.append([0, 1, 3, 7, 12, 20, 30, 44, 60, 79])
    candidates.append([0, 1, 4, 10, 18, 23, 31, 40, 50, 61])
    
    # 4. Hybrid: start with AP, then add sparse large elements
    for base in [5, 7, 9]:
        for n in range(4, 10):
            U = [i * base for i in range(n)]
            # Add a few sparse large elements
            for _ in range(random.randint(1, 3)):
                U.append(random.randint(n * base + 10, n * base + 100))
            U = sorted(set(U))
            if 0 not in U:
                U = [0] + U
            if len(U) >= 2:
                candidates.append(U)
    
    # 5. Random constructions with structure bias
    random.seed(123)
    for trial in range(20):
        n = random.randint(6, 15)
        max_val = random.randint(150, 350)
        U = [0]
        U_set = {0}
        
        # Start with a small structured core
        core_size = random.randint(3, 5)
        base = random.randint(3, 8)
        for i in range(core_size):
            val = i * base + random.randint(0, 2)
            if val not in U_set and val > 0:
                U.append(val)
                U_set.add(val)
        
        # Add more elements with bias toward creating new differences
        for _ in range(n - len(U)):
            best_candidate = None
            best_net_gain = -float('inf')
            
            for _ in range(50):
                candidate = random.randint(1, max_val)
                if candidate in U_set:
                    continue
                
                # Estimate new diffs vs sums
                new_diffs = 0
                new_sums = 0
                for a in U_set:
                    d1 = candidate - a
                    d2 = a - candidate
                    # Check if these diffs already exist
                    diff_exists = False
                    sum_exists = False
                    for b in U_set:
                        if candidate - a == b - a or a - candidate == b - a:
                            diff_exists = True
                            break
                    for b in U_set:
                        if candidate + a == b + a:
                            sum_exists = True
                            break
                    if not diff_exists:
                        new_diffs += 1
                    if not sum_exists:
                        new_sums += 1
                
                net_gain = new_diffs - new_sums * 0.8
                if net_gain > best_net_gain:
                    best_net_gain = net_gain
                    best_candidate = candidate
            
            if best_candidate is not None and best_net_gain > 0:
                U.append(best_candidate)
                U_set.add(best_candidate)
        
        U = sorted(U)
        candidates.append(U)
    
    # Run diff-guided local search on each candidate
    best_U = [0, 1]
    best_score = 0.0
    
    for U in candidates:
        U = sorted(set(U))
        if 0 not in U:
            U = [0] + U
        if len(U) < 2:
            continue
        searched_U, searched_score = local_search_diff_guided(U, max_iters=250, max_val=350)
        if searched_score > best_score:
            best_score = searched_score
            best_U = searched_U
    
    return best_U

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END

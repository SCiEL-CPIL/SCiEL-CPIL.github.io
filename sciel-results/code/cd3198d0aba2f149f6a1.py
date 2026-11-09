"""Rewritten program: greedy constructive algorithm for tight Sidon sets."""
import time
import random
import numpy as np
from math import log

def _compute_score(U):
    """Compute the score for a given set U."""
    U_arr = np.array(sorted(set(U)), dtype=np.int64)
    if len(U_arr) == 0 or 0 not in U_arr:
        return 0.0
    n = len(U_arr)
    if n == 0:
        return 0.0
    max_u = U_arr[-1]
    if max_u == 0:
        return 1.0
    
    sumset = np.unique(U_arr[:, None] + U_arr[None, :])
    diffset = np.unique(U_arr[:, None] - U_arr[None, :])
    
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
    
    score = 1 + log(ratio) / denominator
    return score

def _is_sidon(arr):
    """Check if arr is a Sidon set (all pairwise sums distinct)."""
    n = len(arr)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = arr[i] + arr[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _greedy_sidon(start_set, max_n=12, max_val=2000):
    """Greedy: add smallest candidate that keeps Sidon property."""
    current = sorted(list(start_set))
    if 0 not in current:
        current.insert(0, 0)
    current = sorted(set(current))
    
    for target_n in range(len(current) + 1, max_n + 1):
        # Try to add the smallest possible element
        added = False
        for candidate in range(current[-1] + 1, max_val + 1):
            new_set = current + [candidate]
            if _is_sidon(new_set):
                current = new_set
                added = True
                break
        if not added:
            break
    
    return current

def _local_search(U, max_iterations=10000, timeout=20.0):
    """Local search to improve the set."""
    start_time = time.time()
    current = sorted(set(U))
    if 0 not in current:
        current.insert(0, 0)
    current = sorted(set(current))
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
        
        # Small perturbations
        for x in list(U_set):
            for offset in [-2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15]:
                c = x + offset
                if c >= 0:
                    candidates.add(c)
        
        # Extend max
        candidates.add(max_u + 1)
        candidates.add(max_u + 2)
        candidates.add(max_u + 3)
        candidates.add(max_u + 4)
        candidates.add(max_u + 5)
        
        # Random
        for _ in range(20):
            candidates.add(random.randint(0, max_u + 20))
        
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
            if new_score > current_score + 1e-9:
                current = sorted(new_set)
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best = list(current)
                improved = True
                break
        
        if not improved:
            # Try removing an element
            removable = [x for x in current if x != 0]
            random.shuffle(removable)
            for cand in removable[:15]:
                if time.time() - start_time > timeout:
                    break
                new_set = U_set - {cand}
                if len(new_set) < 2:
                    continue
                new_score = _compute_score(new_set)
                if new_score > current_score + 1e-9:
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
                    new_elem = random.randint(0, max_u + 30)
                    new_set = (U_set - {rem}) | {new_elem}
                    if len(new_set) >= 2:
                        new_score = _compute_score(new_set)
                        if new_score > current_score + 1e-9:
                            current = sorted(new_set)
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best = list(current)
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    best_set = [0, 1]
    best_score = _compute_score(best_set)
    
    # Strategy 1: Greedy Sidon construction for various sizes
    for target_n in range(3, 13):
        if time.time() > deadline - 30:
            break
        # Try greedy from different starting points
        for start in [[0], [0, 1], [0, 1, 3]]:
            if time.time() > deadline - 25:
                break
            # Greedy build up to target_n
            current = list(start)
            for _ in range(target_n - len(current)):
                # Find smallest candidate that maintains Sidon
                added = False
                for cand in range(current[-1] + 1, 5000):
                    new_set = current + [cand]
                    if _is_sidon(new_set):
                        current = new_set
                        added = True
                        break
                if not added:
                    break
            
            if len(current) >= target_n - 1:
                s = _compute_score(current)
                if s > best_score:
                    best_score = s
                    best_set = list(current)
    
    # Strategy 2: Local search from the best greedy result
    if time.time() < deadline - 15:
        remaining = deadline - time.time() - 5
        if remaining > 10:
            improved, score = _local_search(best_set, max_iterations=50000, timeout=remaining)
            if score > best_score:
                best_set = improved
                best_score = score
    
    # Strategy 3: Try specific known good structures
    known_structures = [
        [0, 1, 4, 6],
        [0, 1, 4, 10, 12],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 25],
        [0, 1, 4, 10, 12, 17, 25, 28],
        [0, 1, 4, 9, 10, 12, 17, 25, 28],
    ]
    for struct in known_structures:
        if time.time() > deadline - 10:
            break
        s = _compute_score(struct)
        if s > best_score:
            best_score = s
            best_set = list(struct)
    
    # Final polish
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 2
        if remaining > 5:
            final_result, final_score = _local_search(best_set, max_iterations=20000, timeout=remaining)
            if final_score > best_score:
                best_set = final_result
                best_score = final_score
    
    # Final validation
    final_score = _compute_score(best_set)
    if final_score < 1.0:
        best_set = [0, 1, 4, 6]
    
    return best_set

"""Optimized search for 4-AP-free subset of [1,256] maximizing 3-AP count."""
import time
import random
import numpy as np

N = 256
DEADLINE = 85.0  # safety margin before 90s kill

def _precompute():
    """Precompute 4-AP constraints and 3-AP contribution data."""
    # For each element i (1-indexed), find all 4-APs containing i
    # A 4-AP is (a, a+d, a+2d, a+3d)
    # i can be at position 0,1,2,3 in the AP
    four_ap_sets = [set() for _ in range(N + 1)]  # 1-indexed
    
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e > N:
                break
            four_ap_sets[a].add((b, c, e))
            four_ap_sets[b].add((a, c, e))
            four_ap_sets[c].add((a, b, e))
            four_ap_sets[e].add((a, b, c))
    
    # For 3-AP counting: a 3-AP (a, a+d, a+2d) contributes when all three are in A
    # We want to maximize sum over all valid 3-APs of [all three in A]
    # Precompute for each pair (i, j) with i < j, whether they can form a 3-AP
    # and what the third element would be
    # 3-AP: (a, b, c) where b-a = c-b, i.e., a+c = 2b
    # For each b, the pairs (a, c) with a+c=2b, a<b<c
    
    # Alternative: count 3-APs directly. For each d, for each a where a+2d <= N:
    #   (a, a+d, a+2d) is a 3-AP
    # Total 3-APs in A = sum over all (a,d) of [a in A and a+d in A and a+2d in A]
    
    return four_ap_sets

def _count_3aps(selected):
    """Count distinct 3-term APs in selected set (positive difference)."""
    s = set(selected)
    count = 0
    for a in selected:
        for d in range(1, (N - a) // 2 + 1):
            b = a + d
            c = a + 2 * d
            if c > N:
                break
            if b in s and c in s:
                count += 1
    return count

def _has_4ap(selected):
    """Check if selected set contains a nonconstant 4-AP."""
    s = set(selected)
    for a in selected:
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e > N:
                break
            if b in s and c in s and e in s:
                return True
    return False

def _greedy_construction():
    """Greedy: add elements that maximize 3-APs while avoiding 4-APs."""
    selected = set()
    # Start with a random seed or empty
    remaining = list(range(1, N + 1))
    random.shuffle(remaining)
    
    for x in remaining:
        # Check if adding x creates a 4-AP
        test_set = selected | {x}
        if not _has_4ap(test_set):
            selected.add(x)
    
    return selected

def _local_search(initial, time_limit):
    """Simulated annealing / local search to improve 3-AP count."""
    selected = set(initial)
    current_score = _count_3aps(selected)
    best_selected = set(selected)
    best_score = current_score
    
    start_time = time.time()
    
    # Temperature schedule
    T = 50.0
    T_min = 0.1
    cooling = 0.9995
    
    while time.time() - start_time < time_limit:
        T *= cooling
        if T < T_min:
            T = T_min
        
        # Random move: add, remove, or swap an element
        move_type = random.random()
        
        if move_type < 0.35:
            # Try adding an element
            candidates = [x for x in range(1, N + 1) if x not in selected]
            if not candidates:
                continue
            x = random.choice(candidates)
            new_set = selected | {x}
            if _has_4ap(new_set):
                continue
            new_score = _count_3aps(new_set)
            delta = new_score - current_score
            if delta > 0 or random.random() < np.exp(delta / T):
                selected = new_set
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_selected = set(selected)
        
        elif move_type < 0.65:
            # Try removing an element
            if not selected:
                continue
            x = random.choice(list(selected))
            new_set = selected - {x}
            new_score = _count_3aps(new_set)
            delta = new_score - current_score
            if delta > 0 or random.random() < np.exp(delta / T):
                selected = new_set
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_selected = set(selected)
        
        else:
            # Try swapping: remove one, add one
            if not selected:
                continue
            x_remove = random.choice(list(selected))
            candidates = [x for x in range(1, N + 1) if x not in selected]
            if not candidates:
                continue
            x_add = random.choice(candidates)
            new_set = (selected - {x_remove}) | {x_add}
            if _has_4ap(new_set):
                continue
            new_score = _count_3aps(new_set)
            delta = new_score - current_score
            if delta > 0 or random.random() < np.exp(delta / T):
                selected = new_set
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_selected = set(selected)
    
    return best_selected, best_score

def construct():
    start_time = time.time()
    random.seed(42)
    
    # Phase 1: Multiple greedy constructions with different orderings
    best_set = set()
    best_score = -1
    
    num_greedy = 5
    for i in range(num_greedy):
        if time.time() - start_time > DEADLINE - 10:
            break
        gset = _greedy_construction()
        gscore = _count_3aps(gset)
        if gscore > best_score:
            best_score = gscore
            best_set = set(gset)
    
    # Phase 2: Local search from best greedy result
    remaining_time = DEADLINE - (time.time() - start_time) - 2  # 2s safety margin
    if remaining_time > 5:
        best_set, best_score = _local_search(best_set, remaining_time)
    
    # Final verification
    if _has_4ap(best_set):
        # Fallback: remove elements to break 4-APs
        for x in sorted(best_set, reverse=True):
            best_set.discard(x)
            if not _has_4ap(best_set):
                break
    
    return sorted(best_set)

import itertools
import random
import time
import math
from collections import defaultdict

N = 16
NUM_POINTS = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin (evaluator kills at 330s, 300s limit)
    
    all_points = [(x, y) for x in range(N) for y in range(N)]
    point_to_idx = {p: i for i, p in enumerate(all_points)}
    idx_to_point = all_points
    
    # Precompute isosceles conflict: 
    # iso_conflict[i][j] = bitmask of points k such that (i, j, k) is isosceles
    iso_conflict = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]
    
    # Precompute squared distances
    dist2 = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]
    for i in range(NUM_POINTS):
        xi, yi = all_points[i]
        for j in range(NUM_POINTS):
            xj, yj = all_points[j]
            dist2[i][j] = (xi - xj) ** 2 + (yi - yj) ** 2
    
    # Precompute isosceles conflicts
    for i in range(NUM_POINTS):
        for j in range(i + 1, NUM_POINTS):
            mask = 0
            for k in range(NUM_POINTS):
                if k == i or k == j:
                    continue
                d1 = dist2[i][j]
                d2 = dist2[i][k]
                d3 = dist2[j][k]
                if d1 == d2 or d1 == d3 or d2 == d3:
                    mask |= (1 << k)
            iso_conflict[i][j] = mask
            iso_conflict[j][i] = mask
    
    FULL_MASK = (1 << NUM_POINTS) - 1
    
    # Helper: compute blocked mask for a set
    def compute_blocked_mask(chosen_indices):
        blocked = 0
        for i in range(len(chosen_indices)):
            a = chosen_indices[i]
            for j in range(i + 1, len(chosen_indices)):
                b = chosen_indices[j]
                blocked |= iso_conflict[a][b]
        return blocked
    
    # Helper: check if point p_idx can be added to chosen set (fast, uses blocked_mask)
    def can_add_fast(p_idx, chosen_mask, blocked_mask):
        if (chosen_mask >> p_idx) & 1:
            return False
        if (blocked_mask >> p_idx) & 1:
            return False
        return True
    
    # Helper: check if set is maximal using blocked mask
    def is_maximal_fast(chosen_mask, blocked_mask):
        return ((FULL_MASK & ~chosen_mask) & ~blocked_mask) == 0
    
    # Helper: prune set by removing redundant points while maintaining maximality
    def try_prune(chosen_indices):
        if not chosen_indices:
            return chosen_indices
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(chosen_indices)):
                p = chosen_indices[i]
                s_temp = [x for j, x in enumerate(chosen_indices) if j != i]
                if not s_temp:
                    continue
                s_temp_mask = 0
                for idx in s_temp:
                    s_temp_mask |= (1 << idx)
                blocked_temp = compute_blocked_mask(s_temp)
                if is_maximal_fast(s_temp_mask, blocked_temp):
                    chosen_indices = s_temp
                    improved = True
                    break
        return chosen_indices
    
    best_set = None
    best_size = float('inf')
    
    # Approach A: Randomized greedy with coverage heuristic
    num_greedy_trials = 300
    
    for trial in range(num_greedy_trials):
        if time.time() > deadline - 15:
            break
            
        random.seed(42 + trial)
        
        chosen = []
        chosen_mask = 0
        blocked_mask = 0
        
        points_order = list(range(NUM_POINTS))
        random.shuffle(points_order)
        
        while True:
            if time.time() > deadline - 15:
                break
                
            valid_points = []
            for p_idx in points_order:
                if can_add_fast(p_idx, chosen_mask, blocked_mask):
                    valid_points.append(p_idx)
            
            if not valid_points:
                break
            
            if len(chosen) < 2:
                p_idx = random.choice(valid_points)
            else:
                if len(valid_points) > 40:
                    sample = random.sample(valid_points, 40)
                else:
                    sample = valid_points
                
                max_cov = -1
                best_p = sample[0]
                for p_idx in sample:
                    # Coverage: how many valid points would be blocked if we add p_idx
                    new_blocked = 0
                    for q in chosen:
                        new_blocked |= iso_conflict[p_idx][q]
                    cov = 0
                    for q_idx in valid_points:
                        if q_idx == p_idx:
                            continue
                        if (new_blocked >> q_idx) & 1:
                            cov += 1
                    if cov > max_cov:
                        max_cov = cov
                        best_p = p_idx
                p_idx = best_p
            
            # Update blocked_mask incrementally
            for q in chosen:
                blocked_mask |= iso_conflict[p_idx][q]
            
            chosen.append(p_idx)
            chosen_mask |= (1 << p_idx)
        
        if chosen:
            pruned = try_prune(chosen)
            if len(pruned) < best_size:
                best_size = len(pruned)
                best_set = pruned.copy()
    
    # Local search with swap operations using fast maximality check
    if best_set and time.time() < deadline - 30:
        current_set = best_set.copy()
        current_mask = 0
        for idx in current_set:
            current_mask |= (1 << idx)
        current_blocked = compute_blocked_mask(current_set)
        
        for iteration in range(500):
            if time.time() > deadline - 15:
                break
            
            found_improvement = False
            
            # Try removing each point
            for i in range(len(current_set)):
                p = current_set[i]
                s_temp = [x for j, x in enumerate(current_set) if j != i]
                if not s_temp:
                    continue
                s_temp_mask = 0
                for idx in s_temp:
                    s_temp_mask |= (1 << idx)
                blocked_temp = compute_blocked_mask(s_temp)
                if is_maximal_fast(s_temp_mask, blocked_temp):
                    current_set = s_temp
                    current_mask = s_temp_mask
                    current_blocked = blocked_temp
                    found_improvement = True
                    break
            
            if found_improvement:
                if len(current_set) < best_size:
                    best_size = len(current_set)
                    best_set = current_set.copy()
                continue
            
            # Try swap: remove p, add q
            if len(current_set) > 0:
                p = random.choice(current_set)
                s_without_p = [x for x in current_set if x != p]
                s_without_p_mask = 0
                for idx in s_without_p:
                    s_without_p_mask |= (1 << idx)
                
                # Find q that can be added to s_without_p
                if s_without_p:
                    s_without_p_blocked = compute_blocked_mask(s_without_p)
                else:
                    s_without_p_blocked = 0
                
                candidates = []
                for q_idx in range(NUM_POINTS):
                    if can_add_fast(q_idx, s_without_p_mask, s_without_p_blocked):
                        candidates.append(q_idx)
                
                if candidates:
                    q = random.choice(candidates)
                    s_new = s_without_p + [q]
                    s_new_mask = s_without_p_mask | (1 << q)
                    
                    # Try to prune s_new
                    s_pruned = try_prune(s_new)
                    if len(s_pruned) < len(current_set):
                        current_set = s_pruned
                        current_mask = 0
                        for idx in current_set:
                            current_mask |= (1 << idx)
                        current_blocked = compute_blocked_mask(current_set)
                        if len(current_set) < best_size:
                            best_size = len(current_set)
                            best_set = current_set.copy()
    
    if best_set is None:
        # Fallback: simple greedy
        chosen = []
        chosen_mask = 0
        blocked_mask = 0
        for p_idx in range(NUM_POINTS):
            if can_add_fast(p_idx, chosen_mask, blocked_mask):
                for q in chosen:
                    blocked_mask |= iso_conflict[p_idx][q]
                chosen.append(p_idx)
                chosen_mask |= (1 << p_idx)
        return [all_points[idx] for idx in chosen]
    
    return [all_points[idx] for idx in best_set]

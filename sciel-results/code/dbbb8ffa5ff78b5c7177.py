"""Combined approach: minimum-degree greedy + local search refinement.
Builds a maximal no-isosceles set via greedy heuristic, then refines via
point-swap local search to escape local optima and reduce set size."""

import time
import random
import itertools
import numpy as np

N = 16
DEADLINE_SECONDS = 290  # safety margin before 300s kill

# Precompute all pairwise squared distances
_all_pts = list(itertools.product(range(N), repeat=2))
_pt_idx = {p: i for i, p in enumerate(_all_pts)}
_np = len(_all_pts)
_dist_mat = np.zeros((_np, _np), dtype=np.int32)
for i, (x1, y1) in enumerate(_all_pts):
    for j, (x2, y2) in enumerate(_all_pts):
        dx = x1 - x2
        dy = y1 - y2
        _dist_mat[i, j] = dx*dx + dy*dy

def _can_add_idx(p_idx, S_indices):
    """True if point p_idx can be added to set S_indices without creating isosceles triangle."""
    d_p = _dist_mat[p_idx]
    n = len(S_indices)
    for i in range(n):
        si = S_indices[i]
        dpi = d_p[si]
        for j in range(i+1, n):
            sj = S_indices[j]
            dpj = d_p[sj]
            dsij = _dist_mat[si, sj]
            if dpi == dpj or dpi == dsij or dpj == dsij:
                return False
    return True

def _can_add(p, S):
    """True if p can be added to S without creating an isosceles triangle."""
    p_idx = _pt_idx[p]
    S_indices = [_pt_idx[s] for s in S]
    return _can_add_idx(p_idx, S_indices)

def _build_minimal_greedy(order_seed=0):
    """
    Greedy construction using minimum-degree heuristic.
    At each step, among all addable points, pick the one that blocks the fewest others.
    """
    rng = random.Random(order_seed)
    all_points = _all_pts
    S = []
    S_indices = []
    addable = set(range(_np))
    
    while addable:
        best_p_idx = None
        best_score = float('inf')
        
        candidates = list(addable)
        rng.shuffle(candidates)
        
        for p_idx in candidates:
            if not _can_add_idx(p_idx, S_indices):
                continue
            
            new_S_indices = S_indices + [p_idx]
            block_count = 0
            for q_idx in addable:
                if q_idx == p_idx:
                    continue
                if not _can_add_idx(q_idx, new_S_indices):
                    block_count += 1
            
            score = block_count + rng.random() * 0.01
            if score < best_score:
                best_score = score
                best_p_idx = p_idx
        
        if best_p_idx is None:
            break
        
        S.append(_all_pts[best_p_idx])
        S_indices.append(best_p_idx)
        addable.discard(best_p_idx)
        
        new_addable = set()
        for q_idx in addable:
            if _can_add_idx(q_idx, S_indices):
                new_addable.add(q_idx)
        addable = new_addable
    
    return S

def _build_greedy_order(order):
    """Simple greedy in a given order."""
    S = []
    S_indices = []
    for p in order:
        p_idx = _pt_idx[p]
        if _can_add_idx(p_idx, S_indices):
            S.append(p)
            S_indices.append(p_idx)
    return S

def _local_search(S, all_points, max_iterations=200, time_limit=30):
    """
    Local search refinement: try to reduce set size by:
    1. Removing a point and checking if set remains maximal
    2. Random perturbations with greedy rebuild
    """
    S = list(S)
    S_indices = [_pt_idx[s] for s in S]
    S_set = set(S)
    start_time = time.time()
    best_size = len(S)
    best_S = list(S)
    
    def compute_addable_indices(S_idx_list):
        """Return set of indices that can be added to S_idx_list."""
        S_s = set(S_idx_list)
        addable = set()
        for p_idx in range(_np):
            if p_idx not in S_s and _can_add_idx(p_idx, S_idx_list):
                addable.add(p_idx)
        return addable
    
    addable = compute_addable_indices(S_indices)
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Try removing each point in S
        improved = False
        for idx in range(len(S_indices)):
            if time.time() - start_time > time_limit:
                break
            
            S_trial_indices = S_indices[:idx] + S_indices[idx+1:]
            S_trial_set = set(S_trial_indices)
            
            # Check if any point can be added to S_trial
            still_maximal = True
            for p_idx in range(_np):
                if p_idx not in S_trial_set:
                    if _can_add_idx(p_idx, S_trial_indices):
                        still_maximal = False
                        break
            
            if still_maximal:
                # We found a smaller maximal set!
                S_indices = S_trial_indices
                S = [_all_pts[i] for i in S_indices]
                S_set = set(S)
                best_size = len(S)
                best_S = list(S)
                addable = compute_addable_indices(S_indices)
                improved = True
                break  # restart iteration after a successful removal
        
        if not improved:
            # No improvement this round; try random perturbations
            if random.random() < 0.5:
                idx = random.randrange(len(S_indices))
                S_trial_indices = S_indices[:idx] + S_indices[idx+1:]
                # Greedily re-add points to make it maximal again
                S_rebuilt_indices = list(S_trial_indices)
                addable_rebuilt = compute_addable_indices(S_rebuilt_indices)
                # Greedily add points in random order until maximal
                rng_local = random.Random(iteration * 1000 + idx)
                add_list = list(addable_rebuilt)
                rng_local.shuffle(add_list)
                for p_idx in add_list:
                    if _can_add_idx(p_idx, S_rebuilt_indices):
                        S_rebuilt_indices.append(p_idx)
                        addable_rebuilt.discard(p_idx)
                        # Recompute addable periodically for efficiency
                        if len(S_rebuilt_indices) % 5 == 0:
                            S_s = set(S_rebuilt_indices)
                            addable_rebuilt = set()
                            for q_idx in addable_rebuilt:
                                if q_idx not in S_s and _can_add_idx(q_idx, S_rebuilt_indices):
                                    addable_rebuilt.add(q_idx)
                        if not addable_rebuilt:
                            break
                if len(S_rebuilt_indices) < best_size:
                    S_indices = S_rebuilt_indices
                    S = [_all_pts[i] for i in S_indices]
                    S_set = set(S)
                    best_size = len(S)
                    best_S = list(S)
                    addable = compute_addable_indices(S_indices)
    
    return best_S

def construct():
    start_time = time.time()
    
    best_set = None
    best_size = float('inf')
    all_points = _all_pts
    
    # First, try row-major greedy as baseline
    row_major_order = list(itertools.product(range(N), repeat=2))
    S1 = _build_greedy_order(row_major_order)
    best_set = S1
    best_size = len(S1)
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS:
        return best_set
    
    # Try minimum-degree greedy with different seeds, then local search
    attempts = 50
    for seed in range(attempts):
        if time.time() - start_time > DEADLINE_SECONDS - 10:
            break
        S = _build_minimal_greedy(order_seed=seed)
        
        # Apply local search refinement
        remaining = DEADLINE_SECONDS - 5 - (time.time() - start_time)
        if remaining > 5:
            S_refined = _local_search(S, all_points, max_iterations=100, time_limit=min(remaining, 20))
            if len(S_refined) < best_size:
                best_size = len(S_refined)
                best_set = S_refined
        elif len(S) < best_size:
            best_size = len(S)
            best_set = S
    
    # Try column-major order
    col_major_order = list(itertools.product(range(N), repeat=2))
    col_major_order.sort(key=lambda p: (p[1], p[0]))
    if time.time() - start_time < DEADLINE_SECONDS - 5:
        S2 = _build_greedy_order(col_major_order)
        S2_refined = _local_search(S2, all_points, max_iterations=100, time_limit=15)
        if len(S2_refined) < best_size:
            best_size = len(S2_refined)
            best_set = S2_refined
    
    # Try diagonal orderings
    if time.time() - start_time < DEADLINE_SECONDS - 5:
        diag_order = sorted(itertools.product(range(N), repeat=2), 
                          key=lambda p: (abs(p[0]-p[1]), p[0], p[1]))
        S3 = _build_greedy_order(diag_order)
        S3_refined = _local_search(S3, all_points, max_iterations=100, time_limit=15)
        if len(S3_refined) < best_size:
            best_size = len(S3_refined)
            best_set = S3_refined
    
    # Random orderings with local search
    rng_global = random.Random(42)
    for _ in range(10):
        if time.time() - start_time > DEADLINE_SECONDS - 5:
            break
        random_order = list(itertools.product(range(N), repeat=2))
        rng_global.shuffle(random_order)
        S = _build_greedy_order(random_order)
        remaining = DEADLINE_SECONDS - 5 - (time.time() - start_time)
        if remaining > 5:
            S_refined = _local_search(S, all_points, max_iterations=50, time_limit=min(remaining, 10))
            if len(S_refined) < best_size:
                best_size = len(S_refined)
                best_set = S_refined
    
    return best_set

if __name__ == "__main__":
    result = construct()
    print(f"Set size: {len(result)}")

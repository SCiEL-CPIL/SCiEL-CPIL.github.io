"""Combined approach: minimum-degree greedy + local search refinement.
Builds a maximal no-isosceles set via greedy heuristic, then refines via
point-swap local search to escape local optima and reduce set size."""

import time
import random
import itertools
import numpy as np

N = 16
DEADLINE_SECONDS = 290  # safety margin before 300s kill

# Precompute all points and distance matrix
ALL_POINTS = list(itertools.product(range(N), repeat=2))
N_POINTS = N * N
POINT_INDEX = {p: i for i, p in enumerate(ALL_POINTS)}

# Precompute squared distance matrix
_coords = np.array(ALL_POINTS, dtype=np.float64)
_diff = _coords[:, np.newaxis, :] - _coords[np.newaxis, :, :]
DIST_MATRIX = (_diff ** 2).sum(axis=2)  # N_POINTS x N_POINTS

def _can_add_fast(p_idx, S_indices):
    """True if p can be added to S (given as list of indices) without creating isosceles triangle."""
    if not S_indices:
        return True
    S_arr = np.array(S_indices)
    d_p = DIST_MATRIX[p_idx, S_arr]  # distances from p to each point in S
    d_S = DIST_MATRIX[S_arr[:, np.newaxis], S_arr[np.newaxis, :]]  # pairwise distances in S
    # Check: for any pair (i,j) in S, d(p,S_i) == d(p,S_j) or d(p,S_i) == d(S_i,S_j) or d(p,S_j) == d(S_i,S_j)
    # d_p == d_p for same index, so check d_p[:, i] == d_p[j] for i != j
    # And d_p[i] == d_S[i,j] for any i != j
    # Efficient check:
    n = len(S_arr)
    if n < 2:
        return True
    # Check if any two distances from p to S are equal
    # Use sorting: if any two values in d_p are equal, return False
    sorted_d = np.sort(d_p)
    if np.any(np.diff(sorted_d) == 0):
        return False
    # Check if d_p[i] == d_S[i,j] for any i != j
    # d_S[i,j] where j != i
    # For each i, check if d_p[i] appears in d_S[i, j] for j != i
    for i in range(n):
        row = d_S[i]
        # Exclude d_S[i,i] = 0
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        if d_p[i] in row[mask]:
            return False
    return True

def _build_minimal_greedy(order_seed=0):
    """
    Greedy construction using minimum-degree heuristic with fast distance checks.
    At each step, among all addable points, pick the one that blocks the fewest others.
    """
    rng = random.Random(order_seed)
    S_indices = []
    addable_indices = set(range(N_POINTS))
    
    while addable_indices:
        best_p = None
        best_score = float('inf')
        
        candidates = list(addable_indices)
        rng.shuffle(candidates)
        
        for p_idx in candidates:
            if not _can_add_fast(p_idx, S_indices):
                continue
            
            new_S = S_indices + [p_idx]
            block_count = 0
            for q_idx in addable_indices:
                if q_idx == p_idx:
                    continue
                if not _can_add_fast(q_idx, new_S):
                    block_count += 1
            
            score = block_count + rng.random() * 0.01
            if score < best_score:
                best_score = score
                best_p = p_idx
        
        if best_p is None:
            break
        
        S_indices.append(best_p)
        addable_indices.discard(best_p)
        
        new_addable = set()
        for q_idx in addable_indices:
            if _can_add_fast(q_idx, S_indices):
                new_addable.add(q_idx)
        addable_indices = new_addable
    
    return [ALL_POINTS[i] for i in S_indices]

def _build_greedy_order(order):
    """Simple greedy in a given order."""
    S_indices = []
    for p in order:
        p_idx = POINT_INDEX[p]
        if _can_add_fast(p_idx, S_indices):
            S_indices.append(p_idx)
    return [ALL_POINTS[i] for i in S_indices]

def _local_search(S, all_points, max_iterations=200, time_limit=30):
    """
    Local search refinement: try to reduce set size by:
    1. Removing a point and checking if set remains maximal
    2. Swapping one point for another to maintain maximality
    3. Removing two points and checking if set remains maximal
    """
    S_indices = [POINT_INDEX[p] for p in S]
    S_set = set(S_indices)
    start_time = time.time()
    best_size = len(S_indices)
    best_S_indices = list(S_indices)
    
    def compute_addable_indices(S_idx_list):
        S_s = set(S_idx_list)
        addable = set()
        for p_idx in range(N_POINTS):
            if p_idx not in S_s and _can_add_fast(p_idx, S_idx_list):
                addable.add(p_idx)
        return addable
    
    def is_maximal(S_idx_list):
        """Check if no point can be added to S_idx_list."""
        S_s = set(S_idx_list)
        for p_idx in range(N_POINTS):
            if p_idx not in S_s and _can_add_fast(p_idx, S_idx_list):
                return False
        return True
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        
        # Try removing each point in S
        for idx in range(len(S_indices)):
            if time.time() - start_time > time_limit:
                break
            
            S_trial = S_indices[:idx] + S_indices[idx+1:]
            
            if is_maximal(S_trial):
                S_indices = S_trial
                S_set = set(S_indices)
                best_size = len(S_indices)
                best_S_indices = list(S_indices)
                improved = True
                break
        
        if improved:
            continue
        
        # Try removing two points
        if len(S_indices) >= 2:
            for i in range(len(S_indices)):
                if time.time() - start_time > time_limit:
                    break
                for j in range(i+1, len(S_indices)):
                    S_trial = S_indices[:i] + S_indices[i+1:j] + S_indices[j+1:]
                    if is_maximal(S_trial):
                        S_indices = S_trial
                        S_set = set(S_indices)
                        best_size = len(S_indices)
                        best_S_indices = list(S_indices)
                        improved = True
                        break
                if improved:
                    break
        
        if improved:
            continue
        
        # Random perturbation: remove random point, greedily rebuild
        if random.random() < 0.4 and len(S_indices) > 0:
            idx = random.randrange(len(S_indices))
            S_trial = S_indices[:idx] + S_indices[idx+1:]
            S_rebuilt = list(S_trial)
            addable = compute_addable_indices(S_rebuilt)
            rng_local = random.Random(iteration)
            add_list = list(addable)
            rng_local.shuffle(add_list)
            for p_idx in add_list:
                if _can_add_fast(p_idx, S_rebuilt):
                    S_rebuilt.append(p_idx)
                    addable = compute_addable_indices(S_rebuilt)
                    if not addable:
                        break
            if len(S_rebuilt) < best_size:
                S_indices = S_rebuilt
                S_set = set(S_indices)
                best_size = len(S_indices)
                best_S_indices = list(S_indices)
    
    return [ALL_POINTS[i] for i in best_S_indices]

def construct():
    start_time = time.time()
    
    best_set = None
    best_size = float('inf')
    all_points = list(itertools.product(range(N), repeat=2))
    
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

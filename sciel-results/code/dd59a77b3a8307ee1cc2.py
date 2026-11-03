"""Packing 50 equal circles in a unit circle via discrete lattice search + continuous polish."""
import numpy as np
from scipy.optimize import minimize
import time
import heapq

N = 50

def _score(points):
    """Compute the objective: minimum of half-min-pair-distance and min-margin."""
    if len(points) != N:
        return -1.0
    if not np.all(np.isfinite(points)):
        return -1.0
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    norms = np.sqrt(np.sum(points**2, axis=1))
    min_margin = np.min(1.0 - norms)
    return min(min_pair_dist / 2.0, min_margin)

def _hex_lattice_points(spacing):
    """Generate a hexagonal lattice of points within the unit circle."""
    points = []
    # Approximate radius to cover unit circle
    R = 1.2 
    # Hex grid parameters
    dx = spacing
    dy = spacing * np.sqrt(3) / 2
    
    i = int(-R / dx) - 2
    while i * dx <= R:
        j = int(-R / dy) - 2
        while j * dy <= R:
            x = i * dx
            y = j * dy + (i % 2) * (dy)
            if x*x + y*y <= 1.0:
                points.append([x, y])
            j += 1
        i += 1
    return np.array(points) if points else np.array([[0.0, 0.0]])

def _find_independent_set(candidates, r, max_iters=2000, seed=42):
    """
    Use Simulated Annealing to find N non-overlapping points from candidates.
    Candidates are points in the lattice.
    We represent the state as a boolean mask or list of indices.
    Objective: Maximize number of points placed (target N).
    Penalty: Overlaps.
    """
    np.random.seed(seed)
    n_cand = len(candidates)
    if n_cand < N:
        return None
        
    # Precompute distances between all candidates? 
    # n_cand might be large. Let's estimate.
    # If spacing = 2*r ~ 0.25, area ~ pi. Points ~ pi / (0.25^2 * 0.866) ~ 57.
    # So n_cand is small (~100-200). Precomputation is feasible.
    
    dists = np.zeros((n_cand, n_cand))
    for i in range(n_cand):
        diff = candidates[i] - candidates
        dists[i] = np.sqrt(np.sum(diff**2, axis=1))
        
    # Conflict matrix: 1 if dist < 2*r
    conflict = dists < (2 * r - 1e-9)
    np.fill_diagonal(conflict, False)
    
    # State: list of indices of selected points
    # Start with greedy selection
    selected = []
    available = np.ones(n_cand, dtype=bool)
    
    # Greedy init
    for _ in range(N):
        if not np.any(available):
            break
        # Pick random available
        idx = np.where(available)[0]
        pick = np.random.choice(idx)
        selected.append(pick)
        available[pick] = False
        # Remove neighbors
        neighbors = np.where(conflict[pick])[0]
        available[neighbors] = False
        
    current_count = len(selected)
    best_selected = selected.copy()
    best_count = current_count
    
    if best_count == N:
        return candidates[best_selected]
        
    # SA to fix overlaps or increase count
    # We maintain a set of selected indices.
    # Move: Swap a selected index with an unselected index, or add/remove.
    # Simpler: Randomly pick one selected, remove it, then try to add a new one.
    
    temp = 1.0
    cooling = 0.99
    
    for it in range(max_iters):
        if not np.any(available):
             # This logic is flawed if we just track 'available'. 
             # We need a dynamic state.
             pass
             
    # Let's use a simpler local search on the set of indices.
    # State: set of indices S.
    # Score: - (N - |S|) * 100 + (Number of conflicts in S)
    # We want to minimize score (maximize size, minimize conflicts).
    
    S = set(best_selected)
    current_score = (N - len(S)) * 100 + sum(1 for i in S for j in S if i < j and conflict[i][j])
    
    for it in range(max_iters):
        if len(S) == N and not any(conflict[i][j] for i in S for j in S if i < j):
             return candidates[list(S)]
             
        # Generate neighbor
        new_S = set(S)
        if np.random.rand() < 0.5 and S:
            # Remove a random element
            to_remove = np.random.choice(list(S))
            new_S.remove(to_remove)
        else:
            # Add a random element
            candidates_list = [i for i in range(n_cand) if i not in S]
            if candidates_list:
                to_add = np.random.choice(candidates_list)
                new_S.add(to_add)
                
        # Calculate new score
        new_count = len(new_S)
        new_conflicts = 0
        S_list = list(new_S)
        for i_idx in range(len(S_list)):
            i = S_list[i_idx]
            for j_idx in range(i_idx + 1, len(S_list)):
                j = S_list[j_idx]
                if conflict[i][j]:
                    new_conflicts += 1
        new_score = (N - new_count) * 100 + new_conflicts
        
        delta = new_score - current_score
        if delta < 0 or np.random.rand() < np.exp(-delta / max(temp, 1e-9)):
            S = new_S
            current_score = new_score
            
        temp *= cooling
        
    # Return best found
    # Find best S from history? We only kept current.
    # For simplicity, return current if it's good, else None.
    if len(S) == N:
         # Check conflicts
         S_list = list(S)
         for i_idx in range(len(S_list)):
             i = S_list[i_idx]
             for j_idx in range(i_idx + 1, len(S_list)):
                 j = S_list[j_idx]
                 if conflict[i][j]:
                     return None
         return candidates[S_list]
    return None

def _polish_with_smooth(points, deadline):
    """Polish using smooth objective with L-BFGS-B."""
    best = points.copy()
    best_score = _score(best)
    
    def _soft_min_objective(points_flat, beta=100.0):
        points_2d = points_flat.reshape(N, 2)
        diff = points_2d[:, np.newaxis, :] - points_2d[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, k=1)
        pair_dists = dists[iu]
        radii = np.hypot(points_2d[:, 0], points_2d[:, 1])
        boundary_dists = 1.0 - radii
        all_dists = np.concatenate([pair_dists, boundary_dists])
        scaled_dists = all_dists / 2.0
        neg_beta_scaled = -beta * scaled_dists
        max_val = np.max(neg_beta_scaled)
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_scaled - max_val)))
        soft_min = -log_sum_exp / beta
        return -soft_min

    try:
        for beta in [50.0, 100.0, 200.0]:
            if time.time() > deadline - 2.0:
                break
            result = minimize(
                _soft_min_objective,
                best.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            optimized_points = result.x.reshape(N, 2)
            if not np.all(np.isfinite(optimized_points)):
                continue
            norms = np.linalg.norm(optimized_points, axis=1)
            if np.any(norms > 1.0):
                continue
            new_score = _score(optimized_points)
            if new_score > best_score:
                best_score = new_score
                best = optimized_points.copy()
    except Exception:
        pass
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    best_points = None
    best_score = -1.0
    
    # Binary search on radius
    low = 0.10
    high = 0.13
    best_r = low
    
    # We will try to find a valid packing for several r values
    # and keep the one with the highest score after polishing.
    
    # Strategy:
    # 1. Try r = 0.125, 0.126, etc.
    # 2. For each r, generate lattice, find independent set, polish.
    
    r_candidates = np.linspace(0.120, 0.128, 20)
    
    for r in r_candidates:
        if time.time() > deadline - 10.0:
            break
            
        # Lattice spacing slightly smaller than 2r to allow flexibility?
        # Actually, for independent set, we need dist >= 2r.
        # If lattice spacing is 2r, neighbors are exactly 2r.
        # Let's use spacing = 2 * r.
        spacing = 2 * r
        candidates = _hex_lattice_points(spacing)
        
        if len(candidates) < N:
            continue
            
        # Find independent set
        found = _find_independent_set(candidates, r, max_iters=1000, seed=int(r*1000))
        
        if found is not None:
            # Polish
            polished = _polish_with_smooth(found, deadline)
            score = _score(polished)
            if score > best_score:
                best_score = score
                best_points = polished.copy()
                best_r = r
                
    if best_points is None:
        # Fallback to simple random if discrete search failed
        # This is unlikely but ensures validity
        best_points = np.random.uniform(-0.8, 0.8, (N, 2))
        # Ensure inside
        norms = np.linalg.norm(best_points, axis=1)
        mask = norms > 1.0
        best_points[mask] *= (1.0 - 1e-9) / norms[mask]
        
    return best_points

"""Optimized 50-sphere packing in unit cube using simulated annealing with vectorized gradients."""
N = 50
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import sys

# Deadline guard: kill after 135s (15s margin from 150s limit)
DEADLINE = time.time() + 135.0

def _time_left():
    return DEADLINE - time.time()

def _pairwise_distances(pts):
    """Compute pairwise distances for N points, shape (N, N)."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))

def _score(pts):
    """Compute the radius score: min of half-pairwise-dist and min-wall-dist."""
    # Wall distances: 1/2 - |coord| for each coord
    wall_dists = 0.5 - np.abs(pts)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    pd = _pairwise_distances(pts)
    np.fill_diagonal(pd, np.inf)
    min_pair = np.min(pd) / 2.0
    
    return min(min_pair, min_wall)

def _neg_score(pts):
    """Negative score for minimization."""
    return -_score(pts)

def _grad_neg_score(pts):
    """Gradient of negative score (for L-BFGS-B)."""
    n = len(pts)
    grad = np.zeros_like(pts)
    
    # Wall gradient
    wall_dists = 0.5 - np.abs(pts)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    pd = _pairwise_distances(pts)
    np.fill_diagonal(pd, np.inf)
    min_pair = np.min(pd) / 2.0
    
    r = min(min_pair, min_wall)
    
    # Gradient contribution from walls
    if min_wall <= min_pair:
        # Gradient of -min_wall = -min(0.5 - |pts|)
        # d/dpts_k (0.5 - |pts_k|) = -sign(pts_k)
        # So gradient of -min_wall w.r.t. pts = sign(pts) for the argmin coords
        for i in range(n):
            for k in range(3):
                if wall_dists[i, k] <= min_wall + 1e-12:
                    grad[i, k] += 1.0 if pts[i, k] >= 0 else -1.0
    
    # Gradient contribution from pairs
    if min_pair <= min_wall:
        # Find all pairs achieving min_pair
        for i in range(n):
            for j in range(i + 1, n):
                if pd[i, j] / 2.0 <= min_pair + 1e-12:
                    diff = pts[i] - pts[j]
                    dist = pd[i, j]
                    if dist > 1e-15:
                        # d/dpts_i (|pts_i - pts_j|/2) = (pts_i - pts_j) / (2*dist)
                        # gradient of -min_pair includes this
                        grad[i] += diff / (2.0 * dist)
                        grad[j] -= diff / (2.0 * dist)
    
    return grad

def _bcc_lattice_init():
    """Initialize with BCC-like lattice structure."""
    # BCC has 8 corner + 1 center per unit cell
    # For 50 spheres, use a 3x3x3 grid of BCC cells = 27 cells * 2 = 54, take 50
    # Or use a denser arrangement
    
    # Try a 4x4x4 simple cubic first, then perturb
    m = 4
    side = 1.0 / m
    centers = []
    for ix in range(m):
        for iy in range(m):
            for iz in range(m):
                x = -0.5 + (ix + 0.5) * side
                y = -0.5 + (iy + 0.5) * side
                z = -0.5 + (iz + 0.5) * side
                centers.append([x, y, z])
    
    centers = np.array(centers[:N])
    return centers

def _random_init():
    """Random initialization within cube."""
    return np.random.uniform(-0.45, 0.45, size=(N, 3))

def _refine_local(pts, max_iter=200):
    """Local refinement using L-BFGS-B."""
    if _time_left() < 5.0:
        return pts
    
    # Use smooth surrogate: soft-min via log-sum-exp for better gradients
    def _smooth_neg_score(pts, beta=50.0):
        """Smooth surrogate using log-sum-exp for min."""
        # Wall distances
        wall_dists = 0.5 - np.abs(pts).flatten()
        
        # Pairwise distances
        pd = _pairwise_distances(pts)
        np.fill_diagonal(pd, np.inf)
        pair_dists = pd / 2.0
        pair_dists = pair_dists[pair_dists < np.inf].flatten()
        
        # Soft min via -logsumexp(-beta * x)
        # min(x) ≈ -logsumexp(-beta * x) / beta for large beta
        all_dists = np.concatenate([wall_dists, pair_dists])
        
        # For numerical stability
        max_val = np.max(all_dists)
        soft_min = max_val - np.log(np.sum(np.exp(-beta * (all_dists - max_val)))) / beta
        
        return -soft_min
    
    def _smooth_grad(pts, beta=50.0):
        """Gradient of smooth surrogate."""
        wall_dists = 0.5 - np.abs(pts).flatten()
        
        pd = _pairwise_distances(pts)
        np.fill_diagonal(pd, np.inf)
        pair_dists = pd / 2.0
        valid_pairs = pair_dists < np.inf
        pair_dists_flat = pair_dists[valid_pairs].flatten()
        
        all_dists = np.concatenate([wall_dists, pair_dists_flat])
        
        max_val = np.max(all_dists)
        exp_vals = np.exp(-beta * (all_dists - max_val))
        sum_exp = np.sum(exp_vals)
        
        # d(-soft_min)/d(all_dists) = -exp_vals / sum_exp
        grad_all = -exp_vals / sum_exp
        
        grad = np.zeros_like(pts)
        
        # Wall gradient
        n_wall = len(wall_dists)
        grad_wall = grad_all[:n_wall].reshape(N, 3)
        # d wall_dists[i,k] / d pts[i,k] = -sign(pts[i,k])
        signs = np.sign(pts)
        signs[signs == 0] = 1.0
        grad -= grad_wall * signs
        
        # Pair gradient
        grad_pair_flat = grad_all[n_wall:]
        grad_pair = grad_pair_flat.reshape(-1, 1)
        # Need to map back to pairs
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                if valid_pairs[i, j]:
                    g = grad_pair_flat[idx]
                    diff = pts[i] - pts[j]
                    dist = pd[i, j]
                    if dist > 1e-15:
                        # d(pair_dist)/d(pts[i]) = diff/dist, d(pair_dist)/d(pts[j]) = -diff/dist
                        grad[i] += g * diff / dist
                        grad[j] -= g * diff / dist
                    idx += 1
        
        return grad
    
    best_pts = pts.copy()
    best_score = _score(pts)
    
    for attempt in range(3):
        if _time_left() < 3.0:
            break
        
        beta = 30.0 + attempt * 20.0
        result = minimize(
            lambda p: _smooth_neg_score(p, beta=beta),
            pts,
            jac=lambda p: _smooth_grad(p, beta=beta),
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (N * 3),
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        if result.x is not None and np.all(np.isfinite(result.x)):
            candidate = result.x.reshape(N, 3)
            s = _score(candidate)
            if s > best_score:
                best_score = s
                best_pts = candidate
                pts = candidate.copy()
    
    return best_pts

def _simulated_annealing(pts, n_restarts=5):
    """Simulated annealing with local refinement."""
    best_pts = pts.copy()
    best_score = _score(pts)
    
    for restart in range(n_restarts):
        if _time_left() < 10.0:
            break
        
        current_pts = best_pts.copy()
        current_score = best_score
        
        # Annealing schedule
        T_start = 0.1
        T_end = 0.001
        n_iter = 500
        
        for it in range(n_iter):
            if _time_left() < 5.0:
                break
            
            T = T_start * (T_end / T_start) ** (it / n_iter)
            
            # Perturb a random subset of points
            n_perturb = max(1, N // 4)
            idx = np.random.choice(N, n_perturb, replace=False)
            noise = np.random.normal(0, 0.02, size=(n_perturb, 3))
            
            new_pts = current_pts.copy()
            new_pts[idx] += noise
            # Clamp to cube
            new_pts = np.clip(new_pts, -0.499, 0.499)
            
            new_score = _score(new_pts)
            delta = new_score - current_score
            
            if delta > 0 or np.random.random() < np.exp(delta / T):
                current_pts = new_pts
                current_score = new_score
        
        # Local refinement
        refined = _refine_local(current_pts)
        refined_score = _score(refined)
        
        if refined_score > best_score:
            best_score = refined_score
            best_pts = refined.copy()
    
    return best_pts

def construct():
    """Construct 50 sphere centers in unit cube maximizing minimum radius."""
    start_time = time.time()
    
    # Multiple initialization strategies
    inits = []
    
    # BCC lattice
    inits.append(_bcc_lattice_init())
    
    # Random inits
    for _ in range(3):
        inits.append(_random_init())
    
    # Structured grid with perturbation
    m = 4
    side = 1.0 / m
    grid_pts = []
    for ix in range(m):
        for iy in range(m):
            for iz in range(m):
                x = -0.5 + (ix + 0.5) * side
                y = -0.5 + (iy + 0.5) * side
                z = -0.5 + (iz + 0.5) * side
                grid_pts.append([x, y, z])
    grid_pts = np.array(grid_pts[:N])
    # Add small random perturbation
    grid_pts += np.random.normal(0, 0.01, size=(N, 3))
    grid_pts = np.clip(grid_pts, -0.499, 0.499)
    inits.append(grid_pts)
    
    best_pts = None
    best_score = -np.inf
    
    for init in inits:
        if _time_left() < 15.0:
            break
        
        # Local refinement first
        refined = _refine_local(init.copy())
        s = _score(refined)
        
        if s > best_score:
            best_score = s
            best_pts = refined.copy()
        
        # Simulated annealing if time permits
        if _time_left() > 30.0:
            sa_result = _simulated_annealing(refined, n_restarts=2)
            sa_score = _score(sa_result)
            if sa_score > best_score:
                best_score = sa_score
                best_pts = sa_result.copy()
    
    # Final high-precision polish
    if best_pts is not None and _time_left() > 5.0:
        best_pts = _refine_local(best_pts, max_iter=300)
    
    if best_pts is None:
        # Fallback: simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_pts = grid[:N]
    
    # Ensure valid output
    best_pts = np.clip(best_pts, -0.5 + 1e-9, 0.5 - 1e-9)
    
    return best_pts

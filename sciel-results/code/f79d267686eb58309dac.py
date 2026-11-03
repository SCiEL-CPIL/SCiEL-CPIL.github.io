"""
Packing 100 equal spheres in a unit cube using a discrete lattice-based strategy.

Strategy:
1. Generate a dense 12x12x12 lattice of candidate centers within the cube.
2. Use a greedy algorithm to select a maximal independent set (non-overlapping spheres) for a target radius.
3. Use Simulated Annealing to refine the selection of lattice points to maximize the minimum distance.
4. Perform a final local continuous polish using L-BFGS-B on the best discrete configuration.
5. Deadline guard to ensure completion within 150s.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 100

def _score(points):
    """Return the maximal radius for the given set of centres."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Inside the cube [-0.5, 0.5]^3
    margin = 0.5 - np.abs(pts).max(axis=1)
    r_margin = margin.min()
    if r_margin <= 0:
        return 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = dists.min()
    r_pair = min_dist / 2.0
    return min(r_pair, r_margin)

def _generate_lattice(nx=12, ny=12, nz=12):
    """Generate a lattice of candidate points."""
    # Slightly inset to ensure points are well inside
    x = np.linspace(-0.48, 0.48, nx)
    y = np.linspace(-0.48, 0.48, ny)
    z = np.linspace(-0.48, 0.48, nz)
    points = np.array([(x, y, z) for z in z for y in y for x in x])
    return points

def _greedy_select(lattice_points, target_radius, N_target=100):
    """
    Greedily select N_target points from the lattice such that pairwise distances
    are at least 2 * target_radius.
    Returns the selected indices.
    """
    selected = []
    available = list(range(len(lattice_points)))
    
    # Sort available points by "centrality" or just randomize for diversity?
    # Randomize to avoid bias towards corners/edges in greedy
    random.shuffle(available)
    
    min_dist = 2.0 * target_radius
    
    while available and len(selected) < N_target:
        # Pick a random available point
        idx = available.pop()
        p = lattice_points[idx]
        
        # Check against already selected points
        valid = True
        for s_idx in selected:
            s_p = lattice_points[s_idx]
            if np.linalg.norm(p - s_p) < min_dist:
                valid = False
                break
        
        if valid:
            selected.append(idx)
            
    # If we didn't get enough points, we failed for this radius
    if len(selected) < N_target:
        return None
        
    return selected

def _discrete_energy(selected_indices, lattice_points):
    """
    Energy function for the discrete problem.
    We want to maximize the minimum distance.
    Energy = -min_dist + penalty if min_dist < 2r (but we are maximizing min_dist directly here).
    Actually, for SA, let's define cost as -min_dist.
    """
    if len(selected_indices) != N:
        return 1e10
        
    pts = lattice_points[selected_indices]
    # Check bounds (lattice is inside, so this is safe, but just in case)
    if np.any(np.abs(pts) > 0.5):
        return 1e10
        
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = dists.min()
    
    # Also consider margin to walls
    margin = 0.5 - np.abs(pts).max(axis=1)
    r_margin = margin.min()
    
    # The effective radius is min(min_dist/2, r_margin)
    # We want to maximize this. So minimize -radius.
    r = min(min_dist / 2.0, r_margin)
    return -r

def _local_polish(points, maxiter=100):
    """
    Final continuous polish using L-BFGS-B.
    We optimize the smooth approximation of the minimum radius.
    """
    x0 = points.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    def neg_radius(x):
        pts = x.reshape(N, 3)
        if np.any(np.abs(pts) > 0.5 + 1e-9):
            return 1e10
        margin = 0.5 - np.abs(pts).max(axis=1)
        r_margin = margin.min()
        if r_margin <= 0:
            return 1e10
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=2))
        np.fill_diagonal(dists, np.inf)
        r_pair = dists.min() / 2.0
        # Smooth min
        beta = 100.0
        m = np.minimum(r_pair, r_margin)
        return -(m - np.logaddexp(-beta * (r_pair - m), -beta * (r_margin - m)) / beta)
        
    def grad_neg_radius(x):
        pts = x.reshape(N, 3)
        margin = 0.5 - np.abs(pts).max(axis=1)
        r_margin = margin.min()
        if r_margin <= 0:
            return np.zeros_like(x)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=2))
        np.fill_diagonal(dists, np.inf)
        r_pair = dists.min() / 2.0
        
        beta = 100.0
        m = np.minimum(r_pair, r_margin)
        w_pair = np.exp(-beta * (r_pair - m)) / (np.exp(-beta * (r_pair - m)) + np.exp(-beta * (r_margin - m)))
        w_margin = 1.0 - w_pair
        
        # Gradient of r_pair
        idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = idx
        d_ij = pts[i] - pts[j]
        dist_ij = np.linalg.norm(d_ij)
        if dist_ij > 1e-12:
            grad_dist = d_ij / dist_ij
        else:
            grad_dist = np.zeros(3)
            
        grad_r_pair = np.zeros_like(x)
        grad_r_pair[i*3:(i+1)*3] = grad_dist / 2.0
        grad_r_pair[j*3:(j+1)*3] = -grad_dist / 2.0
        
        # Gradient of r_margin
        min_idx = np.argmin(margin)
        min_pt = pts[min_idx]
        min_coord = np.argmax(np.abs(min_pt))
        
        grad_r_margin = np.zeros_like(x)
        sign = 1.0 if min_pt[min_coord] >= 0 else -1.0
        grad_r_margin[min_idx*3 + min_coord] = -sign
        
        grad = -w_pair * grad_r_pair - w_margin * grad_r_margin
        return grad

    try:
        result = minimize(neg_radius, x0, jac=grad_neg_radius, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-7})
        if result.fun < neg_radius(x0):
            return result.x.reshape(N, 3)
        else:
            return points
    except Exception:
        return points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    random.seed(42)
    np.random.seed(42)
    
    # 1. Generate Lattice
    lattice = _generate_lattice(12, 12, 12)
    lattice_size = len(lattice)
    
    # 2. Find a good starting radius using greedy selection
    # The best known radius is ~0.111. Let's try to find a feasible discrete config.
    # We will use a binary search or step-down to find the max radius for which greedy succeeds.
    # But greedy is sensitive to order. Let's just run SA on a set of points selected by a slightly loose greedy.
    
    # Start with a radius slightly lower than optimal to ensure we can pick 100 points
    # 0.1113 * 2 = 0.2226 is min dist.
    # Let's try to pick 100 points with min dist 0.22.
    
    best_points = None
    best_score = 0.0
    
    # Attempt 1: Greedy with a few random shuffles to find a good initial set
    for attempt in range(20):
        if time.time() > deadline - 60:
            break
        selected = _greedy_select(lattice, 0.11, N)
        if selected is not None:
            pts = lattice[selected]
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
    
    if best_points is None:
        # Fallback: just use a grid if greedy fails completely (should not happen)
        xs = np.linspace(-0.4, 0.4, 5)
        ys = np.linspace(-0.4, 0.4, 5)
        zs = np.linspace(-0.375, 0.375, 4)
        best_points = np.array([(x, y, z) for z in zs for y in ys for x in xs])
        best_score = _score(best_points)
        
    # 3. Simulated Annealing on the discrete lattice
    # State: set of N indices into lattice
    # Move: swap one selected index with one unselected index
    
    if time.time() < deadline - 20:
        current_indices = []
        # Reconstruct indices from best_points (they are lattice points)
        for p in best_points:
            # Find closest lattice point
            dists = np.linalg.norm(lattice - p, axis=1)
            current_indices.append(np.argmin(dists))
            
        current_indices = np.array(current_indices)
        best_indices = current_indices.copy()
        
        T = 0.05
        T_min = 0.001
        cooling_rate = 0.995
        
        while T > T_min and time.time() < deadline - 10:
            # Pick a random index to swap out
            i_out = random.randint(0, N - 1)
            old_idx = current_indices[i_out]
            
            # Pick a random index to swap in (not currently selected)
            # Efficiently pick a random unselected index
            # For small lattice (1728 points) and N=100, we can just pick randomly and check
            # Or maintain a set of available indices
            available = set(range(lattice_size)) - set(current_indices)
            if not available:
                break
            i_in = random.choice(list(available))
            
            # Perform swap
            new_indices = current_indices.copy()
            new_indices[i_out] = i_in
            
            # Calculate energy change
            # To speed up, we can approximate, but for 100 points, full calc is fast enough
            # Let's just call _discrete_energy
            E_old = _discrete_energy(current_indices, lattice)
            E_new = _discrete_energy(new_indices, lattice)
            
            delta_E = E_new - E_old
            
            if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                current_indices = new_indices
                E_curr = E_new
                if E_curr < _discrete_energy(best_indices, lattice):
                    best_indices = current_indices.copy()
            else:
                # Revert (current_indices is already unchanged in memory if we used copy, 
                # but we modified new_indices, so current_indices is safe)
                pass
                
            T *= cooling_rate
            
        # Update best_points from best_indices
        pts_from_sa = lattice[best_indices]
        score_sa = _score(pts_from_sa)
        if score_sa > best_score:
            best_points = pts_from_sa
            best_score = score_sa

    # 4. Local Polish
    if time.time() < deadline - 5:
        polished = _local_polish(best_points, maxiter=200)
        polished_score = _score(polished)
        if polished_score > best_score:
            best_points = polished
            best_score = polished_score

    return best_points

# EVOLVE-BLOCK-END

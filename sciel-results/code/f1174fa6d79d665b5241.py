"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
N = 21
import numpy as np
import time
from scipy.spatial import Voronoi, QhullError

DEADLINE = 150.0  # seconds wall-clock limit
SAFETY_MARGIN = 5.0  # seconds before deadline to stop and return best


def _compute_radii(centers, W, H):
    """Compute the maximum radius for each circle given centers and rectangle bounds.
    
    Uses Voronoi diagram to find the inradius of each cell (distance to nearest
    other circle center or to rectangle boundary).
    """
    n = len(centers)
    radii = np.zeros(n)
    if n < 3:
        # Fallback for small n
        for i in range(n):
            cx, cy = centers[i]
            r = min(cx, W - cx, cy, H - cy)
            for j in range(n):
                if i != j:
                    d = np.sqrt((cx - centers[j][0])**2 + (cy - centers[j][1])**2)
                    r = min(r, d / 2.0)
            radii[i] = r
        return radii
    
    # Build Voronoi with boundary points far away to handle edges
    # We'll compute distance to boundary and to nearest neighbor directly
    for i in range(n):
        cx, cy = centers[i]
        # Distance to rectangle boundaries
        r = min(cx, W - cx, cy, H - cy)
        # Distance to other circles
        for j in range(n):
            if i != j:
                d = np.sqrt((cx - centers[j][0])**2 + (cy - centers[j][1])**2)
                r = min(r, d / 2.0)
        radii[i] = max(r, 1e-10)
    return radii


def _compute_radii_fast(centers, W, H):
    """Vectorized computation of radii using pairwise distances."""
    n = len(centers)
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    # Nearest neighbor distance / 2
    min_nn = np.min(dists, axis=1) / 2.0
    # Distance to boundaries
    cx, cy = centers[:, 0], centers[:, 1]
    boundary_dist = np.minimum.reduce([cx, W - cx, cy, H - cy])
    radii = np.maximum(np.minimum(min_nn, boundary_dist), 1e-10)
    return radii


def _sum_radii(centers, W, H):
    """Objective: sum of radii."""
    radii = _compute_radii_fast(centers, W, H)
    return np.sum(radii)


def _validate(centers, W, H):
    """Check if the configuration is valid (disjoint interiors, inside rectangle)."""
    n = len(centers)
    radii = _compute_radii_fast(centers, W, H)
    # Check circles are inside rectangle
    for i in range(n):
        cx, cy = centers[i]
        r = radii[i]
        if cx - r < -1e-9 or cx + r > W + 1e-9 or cy - r < -1e-9 or cy + r > H + 1e-9:
            return False
    # Check disjoint interiors (distance between centers >= sum of radii)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((centers[i][0] - centers[j][0])**2 + (centers[i][1] - centers[j][1])**2)
            if d < radii[i] + radii[j] - 1e-9:
                return False
    return True


def _initial_grid(N, W, H):
    """Create an initial grid placement."""
    side = int(np.ceil(np.sqrt(N)))
    centers = []
    for i in range(side):
        for j in range(side):
            if len(centers) >= N:
                break
            cx = (i + 0.5) * W / side
            cy = (j + 0.5) * H / side
            centers.append([cx, cy])
    return np.array(centers[:N])


def _initial_random(N, W, H, seed=None):
    """Create a random initial placement."""
    rng = np.random.default_rng(seed)
    margin = 0.05
    centers = rng.uniform(margin, 1 - margin, size=(N, 2))
    centers[:, 0] *= W
    centers[:, 1] *= H
    return centers


def _local_refine(centers, W, H, max_iters=2000, tol=1e-10, time_limit=None):
    """Local refinement using gradient-free optimization (Nelder-Mead style via scipy)."""
    from scipy.optimize import minimize
    
    def objective(x):
        c = x.reshape(-1, 2)
        # Penalty for out-of-bounds
        cx, cy = c[:, 0], c[:, 1]
        penalty = 0.0
        for i in range(len(c)):
            if cx[i] < 0: penalty += (100.0 - cx[i])**2
            if cx[i] > W: penalty += (cx[i] - 100.0)**2
            if cy[i] < 0: penalty += (100.0 - cy[i])**2
            if cy[i] > H: penalty += (cy[i] - 100.0)**2
        # Overlap penalty
        for i in range(len(c)):
            for j in range(i+1, len(c)):
                d = np.sqrt((c[i][0]-c[j][0])**2 + (c[i][1]-c[j][1])**2)
                # We want to maximize sum of radii, so minimize negative sum
                # But for local refine, just maximize sum of radii
                pass
        radii = _compute_radii_fast(c, W, H)
        return -np.sum(radii) + penalty
    
    x0 = centers.flatten()
    
    start_time = time.time()
    best_val = objective(x0)
    best_x = x0.copy()
    
    it = 0
    while it < max_iters:
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 50, 'xatol': tol, 'fatol': tol, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
            x0 = res.x.copy()
            it += 1
        except Exception:
            break
    
    return best_x.reshape(-1, 2), -best_val


def _sa_optimize(centers, W, H, time_limit=None, seed=None):
    """Simulated annealing to optimize circle positions."""
    if time_limit is None:
        time_limit = 30.0
    
    rng = np.random.default_rng(seed)
    n = len(centers)
    
    start_time = time.time()
    elapsed = 0.0
    
    # Initial temperature
    T = 0.01
    T_min = 1e-8
    cooling_rate = 0.9995
    step_size = 0.01
    
    current = centers.copy()
    current_val = _sum_radii(current, W, H)
    best = current.copy()
    best_val = current_val
    
    # Adaptive step size
    accepted = 0
    total = 0
    
    while elapsed < time_limit:
        # Propose a move: perturb one or two circles
        n_move = 1 if rng.random() < 0.7 else 2
        indices = rng.choice(n, size=n_move, replace=False)
        
        proposal = current.copy()
        for idx in indices:
            proposal[idx, 0] += rng.normal(0, step_size)
            proposal[idx, 1] += rng.normal(0, step_size)
            # Keep within bounds
            proposal[idx, 0] = np.clip(proposal[idx, 0], 0, W)
            proposal[idx, 1] = np.clip(proposal[idx, 1], 0, H)
        
        proposal_val = _sum_radii(proposal, W, H)
        delta = proposal_val - current_val
        
        if delta > 0 or rng.random() < np.exp(delta / T):
            current = proposal
            current_val = proposal_val
            accepted += 1
            if current_val > best_val:
                best = current.copy()
                best_val = current_val
        
        total += 1
        
        # Adaptive step size
        if total % 100 == 0:
            accept_rate = accepted / total
            if accept_rate > 0.5:
                step_size *= 1.2
            elif accept_rate < 0.1:
                step_size *= 0.8
            step_size = np.clip(step_size, 1e-5, 0.1)
            accepted = 0
            total = 0
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
        
        elapsed = time.time() - start_time
    
    return best, best_val


def _optimize_for_aspect(W, H, time_limit=None, seed=None):
    """Optimize for a given aspect ratio (W x H)."""
    if time_limit is None:
        time_limit = 25.0
    
    start_time = time.time()
    best_centers = None
    best_val = 0.0
    
    # Try multiple initializations
    n_restarts = 3
    restart_time = time_limit / (n_restarts + 1)
    
    for r in range(n_restarts):
        if time.time() - start_time > time_limit - 2:
            break
        
        # Grid initialization
        centers = _initial_grid(N, W, H)
        val = _sum_radii(centers, W, H)
        if val > best_val:
            best_val = val
            best_centers = centers.copy()
        
        # SA optimization
        remaining = time_limit - (time.time() - start_time) - 2
        if remaining < 1:
            break
        
        centers_sa, val_sa = _sa_optimize(centers, W, H, time_limit=min(remaining, 15), seed=seed + r)
        if val_sa > best_val:
            best_val = val_sa
            best_centers = centers_sa.copy()
        
        # Local refinement
        remaining = time_limit - (time.time() - start_time) - 1
        if remaining > 1 and best_centers is not None:
            refined, val_ref = _local_refine(best_centers, W, H, max_iters=500, time_limit=min(remaining, 8))
            if val_ref > best_val:
                best_val = val_ref
                best_centers = refined.copy()
    
    return best_centers, best_val


def construct():
    """Main entry point: pack 21 circles in a perimeter-4 rectangle, maximize sum of radii."""
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    best_centers = None
    best_val = 0.0
    best_W = None
    best_H = None
    
    # Perimeter constraint: 2*(W+H) = 4 => W+H = 2
    # Try different aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        
        # W + H = 2, W/H = ar => W = 2*ar/(1+ar), H = 2/(1+ar)
        W = 2.0 * ar / (1.0 + ar)
        H = 2.0 / (1.0 + ar)
        
        remaining = deadline - time.time()
        if remaining < 3:
            break
        
        # Allocate time: more time for earlier aspect ratios
        time_alloc = min(remaining * 0.3, 30.0)
        
        centers, val = _optimize_for_aspect(W, H, time_limit=time_alloc, seed=42)
        
        if centers is not None and val > best_val:
            best_val = val
            best_centers = centers.copy()
            best_W = W
            best_H = H
    
    # Final validation and adjustment
    if best_centers is None:
        # Fallback: grid in square
        W, H = 1.0, 1.0
        best_centers = _initial_grid(N, W, H)
        best_W, best_H = W, H
    
    # Final local refinement with remaining time
    remaining = deadline - time.time()
    if remaining > 2:
        refined, val_ref = _local_refine(best_centers, best_W, best_H, 
                                         max_iters=1000, time_limit=min(remaining - 1, 15))
        if val_ref > best_val:
            best_val = val_ref
            best_centers = refined
    
    # Construct output: (x, y, r) for each circle
    radii = _compute_radii_fast(best_centers, best_W, best_H)
    
    # Ensure validity: shrink radii slightly to avoid numerical issues
    radii = radii * (1 - 1e-6)
    
    result = np.column_stack([best_centers[:, 0], best_centers[:, 1], radii])
    
    # Verify perimeter
    # The bounding rectangle is defined by the problem: W x H with perimeter 4
    # We need to ensure circles fit within [0, W] x [0, H]
    
    return result


# Quick self-test
if __name__ == "__main__":
    result = construct()
    print(f"Number of circles: {len(result)}")
    print(f"Sum of radii: {np.sum(result[:, 2]):.6f}")
    print(f"First few circles: {result[:3]}")

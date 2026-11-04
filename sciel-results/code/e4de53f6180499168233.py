"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(pts):
    """Compute the packing radius for given centers."""
    # Boundary distance
    d_bound = np.min(np.minimum(
        0.5 - np.abs(pts[:, 0]),
        0.5 - np.abs(pts[:, 1])
    ))
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    d_pair = np.min(dists) / 2.0
    return min(d_bound, d_pair)

def _negative_score(pts):
    """Negative of score for minimization."""
    return -_score(pts)

def _smooth_objective(pts, beta=50.0):
    """Smooth surrogate: soft-min of distances using log-sum-exp."""
    pts = pts.reshape(-1, 2)
    # Boundary
    d_bound = np.minimum(
        0.5 - np.abs(pts[:, 0]),
        0.5 - np.abs(pts[:, 1])
    )
    # Pairwise
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # Soft min of all constraints
    all_d = np.concatenate([d_bound, dists[~np.eye(len(pts), dtype=bool)].ravel()])
    # Use log-sum-exp for soft min: -log(sum(exp(-beta * d)))
    # But for large beta this approximates min
    min_d = np.min(all_d)
    # Soft min approximation
    soft_min = -np.log(np.sum(np.exp(-beta * all_d))) / beta
    return -soft_min

def _initial_grid():
    """Create a 7x7 grid plus one extra circle, perturbed."""
    # 7x7 grid gives r=1/14≈0.0714, but only 49 circles
    # We need 50, so use a slightly different arrangement
    # Start with 8x7 grid (56 spots) and pick 50, or use hexagonal
    # Let's use a 7x8 grid arrangement
    m = 8
    n = 7
    xs = (np.arange(m) + 0.5) / m - 0.5
    ys = (np.arange(n) + 0.5) / n - 0.5
    grid = np.array([(x, y) for y in ys for x in xs])
    # That's 56 points, take 50
    return grid[:50].copy()

def _initial_hexagonal():
    """Hexagonal-like packing initial guess."""
    pts = []
    # Hexagonal packing: rows offset
    r_approx = 0.071
    dy = 2 * r_approx
    dx = 2 * r_approx
    row = 0
    while len(pts) < N:
        offset = (dx / 2) if (row % 2) else 0
        x = -0.5 + r_approx + offset
        while x < 0.5 - r_approx and len(pts) < N:
            pts.append([x, -0.5 + r_approx + row * dy * np.sqrt(3) / 2])
            x += dx
        row += 1
    return np.array(pts[:N])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_pts = _initial_grid()
    best_score = _score(best_pts)
    
    # Try multiple initializations and local optimizations
    rng = np.random.default_rng(42)
    
    def try_optimize(init_pts):
        """Run local optimization on initial points."""
        pts = init_pts.copy()
        # Use scipy minimize with smooth objective
        try:
            res = minimize(
                _smooth_objective,
                pts,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * 2 * N,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8},
                args=(50.0,)
            )
            if res.success or _score(res.x) > _score(pts):
                return res.x
        except:
            pass
        return pts
    
    # Try different initializations
    inits = [
        _initial_grid(),
        _initial_hexagonal(),
    ]
    
    # Add perturbed versions
    for _ in range(3):
        base = _initial_grid()
        pert = base + rng.normal(0, 0.01, size=base.shape)
        inits.append(pert)
    
    for init in inits:
        if time.time() > deadline - 10:
            break
        result = try_optimize(init)
        s = _score(result)
        if s > best_score:
            best_score = s
            best_pts = result
    
    # Additional random restarts if time permits
    for _ in range(10):
        if time.time() > deadline - 5:
            break
        # Random initialization within bounds
        init = rng.uniform(-0.4, 0.4, size=(N, 2))
        result = try_optimize(init)
        s = _score(result)
        if s > best_score:
            best_score = s
            best_pts = result
    
    # Final check
    if time.time() > deadline - 2:
        pass  # Return immediately
    
    return best_pts
# EVOLVE-BLOCK-END

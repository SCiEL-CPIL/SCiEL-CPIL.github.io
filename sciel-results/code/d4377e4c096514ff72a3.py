"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _pairwise_sqr_distances(pts):
    """Compute squared pairwise distances for points of shape (n, d)."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sum(diff ** 2, axis=-1)

def _score(pts):
    """Compute (max/min)^2 of pairwise distances."""
    d2 = _pairwise_sqr_distances(pts)
    iu = np.triu_indices_from(d2, k=1)
    vals = d2[iu]
    mn = vals.min()
    mx = vals.max()
    if mn <= 0:
        return 1e10
    return (mx / mn) ** 2

def _objective(pts_flat):
    """Objective function for optimization: minimize (max/min)^2 with penalty for distinctness."""
    pts = pts_flat.reshape(-1, D)
    d2 = _pairwise_sqr_distances(pts)
    iu = np.triu_indices_from(d2, k=1)
    vals = d2[iu]
    mn = vals.min()
    mx = vals.max()
    if mn < 1e-12:
        return 1e10 + (1e-12 - mn) ** 2
    # Smooth surrogate: use log-sum-exp for max and -log-sum-exp for min
    # This makes the objective differentiable
    beta = 50.0
    log_mx = np.log(np.sum(np.exp(vals * beta / mx)) ) / beta if mx > 0 else 0
    # Better: use soft max and soft min
    # soft_max = (1/beta) * log(sum(exp(beta * x)))
    # soft_min = -(1/beta) * log(sum(exp(-beta * x)))
    soft_max = (1.0 / beta) * np.log(np.sum(np.exp(beta * vals / (mx + 1e-12)))) * (mx + 1e-12)
    soft_min = -(1.0 / beta) * np.log(np.sum(np.exp(-beta * vals / (mn + 1e-12)))) * (mn + 1e-12)
    
    # Actually let's just use the raw ratio with a penalty for small min
    ratio_sq = (mx / mn) ** 2
    # Penalty for points being too close
    penalty = 0.0
    for v in vals:
        if v < 1e-6:
            penalty += (1e-6 - v) ** 2
    return ratio_sq + penalty

def _objective_smooth(pts_flat):
    """Smooth objective using log-sum-exp for differentiability."""
    pts = pts_flat.reshape(-1, D)
    d2 = _pairwise_sqr_distances(pts)
    iu = np.triu_indices_from(d2, k=1)
    vals = d2[iu]
    
    # Ensure all values are positive
    vals = np.maximum(vals, 1e-12)
    
    beta = 100.0
    # soft_max = (1/beta) * log(sum(exp(beta * vals)))
    # soft_min = -(1/beta) * log(sum(exp(-beta * vals)))
    # ratio^2 = soft_max^2 / soft_min^2
    
    # For numerical stability
    max_val = np.max(vals)
    min_val = np.min(vals)
    
    soft_max = (1.0 / beta) * np.log(np.sum(np.exp(beta * (vals - max_val)))) + max_val
    soft_min = -(1.0 / beta) * np.log(np.sum(np.exp(-beta * (vals - min_val)))) + min_val
    
    if soft_min < 1e-12:
        return 1e10
    
    return (soft_max / soft_min) ** 2

def _make_tetrahedron_init():
    """Create initial configuration based on tetrahedron vertices and edge midpoints."""
    # Regular tetrahedron vertices
    a = 1.0
    tetra = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ]) * a / np.sqrt(3)
    
    # Edge midpoints (6 edges)
    edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    midpoints = []
    for i, j in edges:
        mid = (tetra[i] + tetra[j]) / 2
        midpoints.append(mid)
    midpoints = np.array(midpoints)
    
    # We need 14 points: 4 tetra vertices + 6 midpoints + 4 more
    # Add 4 more points - maybe face centers or other symmetric points
    # Face centers (4 faces)
    faces = [(0,1,2), (0,1,3), (0,2,3), (1,2,3)]
    face_centers = []
    for f in faces:
        fc = (tetra[f[0]] + tetra[f[1]] + tetra[f[2]]) / 3
        face_centers.append(fc)
    face_centers = np.array(face_centers)
    
    pts = np.vstack([tetra, midpoints, face_centers])
    return pts

def _random_init(scale=1.0):
    """Random initial configuration."""
    return np.random.randn(N, D) * scale

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_pts = None
    best_score = np.inf
    
    # Try multiple initializations
    inits = []
    
    # Tetrahedron-based init
    inits.append(_make_tetrahedron_init())
    
    # Random inits at various scales
    for scale in [0.5, 1.0, 2.0]:
        for _ in range(5):
            inits.append(_random_init(scale))
    
    # Fibonacci sphere-like init
    n_pts = N
    golden_ratio = (1 + np.sqrt(5)) / 2
    fib_pts = np.zeros((n_pts, D))
    for i in range(n_pts):
        z = 1 - 2 * i / (n_pts - 1)
        r = np.sqrt(1 - z**2)
        theta = 2 * np.pi * i / golden_ratio
        fib_pts[i] = [r * np.cos(theta), r * np.sin(theta), z]
    inits.append(fib_pts)
    
    for init in inits:
        if time.time() > deadline:
            break
        
        init_flat = init.flatten()
        
        # Use L-BFGS-B for optimization
        try:
            res = minimize(
                _objective_smooth,
                init_flat,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'maxfun': 5000}
            )
            
            pts = res.x.reshape(-1, D)
            score = _score(pts)
            
            if score < best_score:
                best_score = score
                best_pts = pts.copy()
        except Exception:
            pass
    
    # If no valid result, return grid
    if best_pts is None or best_score >= 1e9:
        side = int(np.ceil(N ** (1 / D)))
        import itertools
        grid = list(itertools.product(range(side), repeat=D))[:N]
        return np.array(grid, dtype=float)
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 5.0:
        try:
            res = minimize(
                _objective_smooth,
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 2000, 'maxfun': 10000, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            pts = res.x.reshape(-1, D)
            score = _score(pts)
            if score < best_score:
                best_pts = pts.copy()
                best_score = score
        except Exception:
            pass
    
    return best_pts
# EVOLVE-BLOCK-END

"""Improved packing for 50 equal circles in a unit square [-1/2, 1/2]^2."""
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

N = 50

def _score(pts):
    """Compute the max radius for a set of points (N,2)."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Check bounds: all points must be strictly inside or on the boundary
    margins = np.minimum(
        np.minimum(0.5 - pts[:, 0], 0.5 + pts[:, 0]),
        np.minimum(0.5 - pts[:, 1], 0.5 + pts[:, 1])
    )
    wall_r = np.min(margins)
    if wall_r <= 0:
        return 0.0
    # Pairwise distances
    d = pdist(pts, metric='euclidean')
    min_pair = np.min(d)
    if min_pair <= 0:
        return 0.0
    return min(min_pair / 2.0, wall_r)

def _neg_score(pts):
    return -_score(pts)

def _simulated_annealing(pts, start_time, deadline, rng):
    """Simulated annealing to improve circle packing."""
    current = pts.copy()
    current_score = _score(current)
    best = current.copy()
    best_score = current_score
    
    # Start temperature based on initial score
    T = max(0.001, current_score * 0.1)
    T_min = 1e-6
    cooling_rate = 0.9995
    
    step_size = 0.05
    
    while time.time() < deadline - 5 and T > T_min:
        # Choose a random point to move
        i = rng.integers(N)
        
        # Random displacement
        dx = rng.normal(0, step_size)
        dy = rng.normal(0, step_size)
        
        new_pts = current.copy()
        new_pts[i, 0] += dx
        new_pts[i, 1] += dy
        
        # Clamp to valid range
        new_pts[i, 0] = np.clip(new_pts[i, 0], -0.499999, 0.499999)
        new_pts[i, 1] = np.clip(new_pts[i, 1], -0.499999, 0.499999)
        
        new_score = _score(new_pts)
        
        if new_score > current_score:
            current = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best = current.copy()
        else:
            # Accept with probability exp((new - current) / T)
            delta = new_score - current_score
            if delta < 0:
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    current = new_pts
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best = current.copy()
        
        T *= cooling_rate
        
        # Occasionally try moving multiple points
        if rng.random() < 0.1:
            for _ in range(rng.integers(1, 4)):
                j = rng.integers(N)
                dx = rng.normal(0, step_size)
                dy = rng.normal(0, step_size)
                current[j, 0] += dx
                current[j, 1] += dy
                current[j, 0] = np.clip(current[j, 0], -0.499999, 0.499999)
                current[j, 1] = np.clip(current[j, 1], -0.499999, 0.499999)
    
    return best, best_score

def _local_refinement(pts, start_time, deadline, maxiter=1000):
    """Local refinement using Nelder-Mead on the objective."""
    x0 = pts.ravel()
    
    def objective(x):
        return _neg_score(x.reshape(N, 2))
    
    res = minimize(objective, x0, method='Nelder-Mead',
                   options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12})
    return res.x.reshape(N, 2)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_pts = None
    best_score = 0.0
    rng = np.random.default_rng(42)
    
    def try_update(pts):
        nonlocal best_pts, best_score
        pts = np.asarray(pts, dtype=float).reshape(N, 2)
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
    
    # Create good initial configurations
    def make_hexagonal_grid(n_cols, n_rows):
        """Create hexagonal (close-packed) grid arrangement."""
        pts = []
        spacing_x = 1.0 / (n_cols - 1 + 0.5)
        spacing_y = spacing_x * np.sqrt(3) / 2
        
        for row in range(n_rows):
            offset = 0.5 * spacing_x if row % 2 == 1 else 0
            for col in range(n_cols):
                x = -0.5 + spacing_x/2 + col * spacing_x + offset
                y = -0.5 + spacing_y/2 + row * spacing_y
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    pts.append([x, y])
        
        pts = np.array(pts[:N])
        if len(pts) < N:
            # Pad with random points if needed
            while len(pts) < N:
                pts = np.vstack([pts, [rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4)]])
        return pts[:N]
    
    def make_grid(cols, rows):
        xs = np.linspace(-0.5 + 0.5/cols, 0.5 - 0.5/cols, cols)
        ys = np.linspace(-0.5 + 0.5/rows, 0.5 - 0.5/rows, rows)
        gx, gy = np.meshgrid(xs, ys, indexing='xy')
        pts = np.column_stack([gx.ravel(), gy.ravel()])
        return pts[:N]
    
    # Try multiple initializations
    inits = []
    
    # Square grid 5x10
    inits.append(make_grid(5, 10))
    
    # Hexagonal grid
    inits.append(make_hexagonal_grid(8, 8))
    inits.append(make_hexagonal_grid(7, 9))
    inits.append(make_hexagonal_grid(9, 7))
    
    # 7x7 grid + center
    m = 7
    xs = (np.arange(m) + 0.5) / m - 0.5
    ys = (np.arange(m) + 0.5) / m - 0.5
    gx, gy = np.meshgrid(xs, ys, indexing='xy')
    pts_7x7 = np.column_stack([gx.ravel(), gy.ravel()])
    center = np.array([[0.0, 0.0]])
    inits.append(np.vstack([pts_7x7, center])[:N])
    
    # 6x9 grid
    inits.append(make_grid(6, 9))
    
    # Random initialization
    inits.append(rng.uniform(-0.4, 0.4, size=(N, 2)))
    
    # Evaluate all initializations
    for init in inits:
        try_update(init)
    
    # Run simulated annealing from the best initial
    if best_pts is not None:
        sa_pts, sa_score = _simulated_annealing(best_pts, start_time, deadline - 20, rng)
        try_update(sa_pts)
    
    # Local refinement from best
    if time.time() < deadline - 10 and best_pts is not None:
        refined = _local_refinement(best_pts, start_time, deadline - 5, maxiter=2000)
        try_update(refined)
    
    # Second round of SA from refined result
    if time.time() < deadline - 15 and best_pts is not None:
        sa_pts2, sa_score2 = _simulated_annealing(best_pts, start_time, deadline - 10, rng)
        try_update(sa_pts2)
    
    # Final local refinement
    if time.time() < deadline - 5 and best_pts is not None:
        refined2 = _local_refinement(best_pts, start_time, deadline - 2, maxiter=500)
        try_update(refined2)
    
    # Final safety: ensure result is valid
    if best_pts is not None:
        best_pts = np.clip(best_pts, -0.4999999, 0.4999999)
        s = _score(best_pts)
        if s > 0:
            return best_pts
        else:
            return inits[0][:N]
    else:
        return inits[0][:N]

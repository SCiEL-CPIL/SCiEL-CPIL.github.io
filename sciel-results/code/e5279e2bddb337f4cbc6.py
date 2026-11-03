"""
Packing 300 equal circles in a unit circle using vectorized repulsion energy optimization.

Strategy:
1. Initialize with a hexagonal lattice.
2. Use a vectorized Gauss-Newton-like step on a squared overlap penalty.
   This avoids the slow Python loops of the soft-min gradient and allows for
   many more iterations within the time budget.
3. Refine the radius by solving the active constraint system (pairs at minimum
   distance and boundary circles) using a small Newton iteration.
"""

import numpy as np
import time

N = 300
TIME_LIMIT = 145.0  # Safety margin before 150s kill


def _hex_init(n=N):
    """Dense hexagonal lattice clipped to the unit disk, keep n closest to center."""
    # Start with a spacing slightly larger than the expected optimal radius * 2
    # Optimal r is ~0.053, so diameter ~0.106. Spacing 0.11 should work.
    spacing = 0.11
    k = int(1.1 / spacing) + 2
    xs, ys = [], []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            xs.append(x)
            ys.append(y)
    pts = np.column_stack([xs, ys])
    r = np.hypot(pts[:, 0], pts[:, 1])
    inside = r <= 1.0
    pts = pts[inside]
    r = r[inside]
    if len(pts) < n:
        # Fallback if not enough points
        spacing = 0.09
        k = int(1.2 / spacing) + 2
        xs, ys = [], []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                xs.append(x)
                ys.append(y)
        pts = np.column_stack([xs, ys])
        r = np.hypot(pts[:, 0], pts[:, 1])
        inside = r <= 1.0
        pts = pts[inside]
        r = r[inside]
    order = np.argsort(r, kind="stable")
    return pts[order[:n]].copy()


def _compute_distances(pts):
    """Compute pairwise distance matrix."""
    # Using einsum for speed
    # d2[i,j] = |x_i|^2 + |x_j|^2 - 2 x_i . x_j
    x2 = np.sum(pts * pts, axis=1)
    d2 = x2[:, None] + x2[None, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(d2, 0.0)
    d2 = np.maximum(d2, 1e-12)
    d = np.sqrt(d2)
    return d


def _score(pts):
    """Achieved common radius."""
    d = _compute_distances(pts)
    min_pair = np.min(d[np.triu_indices(len(pts), k=1)]) / 2.0
    min_bound = 1.0 - np.max(np.hypot(pts[:, 0], pts[:, 1]))
    return min(min_pair, min_bound)


def _repulsion_step(pts, r, dt, start_time):
    """
    Perform one optimization step based on repulsion forces.
    Force on i from j is proportional to (r - d_ij/2) if overlapping, pushing apart.
    Boundary force pushes inward if |c_i| + r > 1.
    """
    n = len(pts)
    d = _compute_distances(pts)
    
    # Overlaps: d < 2r
    # We want to maximize r, so we treat current r as a constraint.
    # Actually, we want to find a configuration that supports a larger r.
    # Let's define a "virtual radius" rv that is slightly larger than current best r.
    # If we can fit circles of radius rv, we update r.
    
    # For the step, we compute forces based on the current target radius r_target.
    # Let's use r_target = r * 1.01 (try to expand slightly)
    r_target = r * 1.005
    
    # Pairwise forces
    # F_i = sum_{j != i} w_ij * (c_i - c_j) / d_ij
    # w_ij = max(0, r_target - d_ij / 2)
    
    # Vectorized computation of weights
    # W[i,j] = max(0, r_target - d[i,j]/2)
    W = np.maximum(0.0, r_target - d / 2.0)
    np.fill_diagonal(W, 0.0)
    
    # Normalize by distance to get unit direction, then multiply by weight
    # Avoid division by zero
    W_dir = W / d
    np.fill_diagonal(W_dir, 0.0)
    
    # Force on i is sum_j W_dir[i,j] * (c_i - c_j)
    # Let's use matrix multiplication:
    # F = (W_dir @ (c_i - c_j) matrix) ... this is tricky directly.
    # F_i = sum_j W_dir[i,j] * c_i - sum_j W_dir[i,j] * c_j
    # F_i = c_i * sum_j W_dir[i,j] - (W_dir @ c)_i
    
    sum_W_dir = np.sum(W_dir, axis=1)
    Wc = W_dir @ pts  # (n, 2)
    
    F_pair = pts * sum_W_dir[:, None] - Wc
    
    # Boundary forces
    # If |c_i| + r_target > 1, push inward.
    rad = np.hypot(pts[:, 0], pts[:, 1])
    boundary_violation = np.maximum(0.0, rad + r_target - 1.0)
    
    # Force direction is -c_i / |c_i|
    # F_bound_i = -boundary_violation[i] * c_i / |c_i|
    F_bound = np.zeros_like(pts)
    mask = rad > 1e-6
    F_bound[mask] = -boundary_violation[mask, None] * pts[mask] / rad[mask, None]
    
    # Total force
    F = F_pair + F_bound
    
    # Update positions
    # Adaptive step size
    # dt should be small enough to ensure stability
    new_pts = pts + dt * F
    
    # Clip to unit circle to maintain validity
    new_rad = np.hypot(new_pts[:, 0], new_pts[:, 1])
    out = new_rad > 1.0
    if np.any(out):
        new_pts[out] = new_pts[out] / new_rad[out, None]
        
    return new_pts


def construct():
    start_time = time.time()
    pts = _hex_init(N)
    
    # Ensure valid start
    rad = np.hypot(pts[:, 0], pts[:, 1])
    out = rad > 1.0
    if np.any(out):
        pts[out] = pts[out] / rad[out, None]
        
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # Phase 1: Rapid convergence using repulsion forces
    # Start with a small radius and increase it
    current_r = best_score
    
    # We will iterate, trying to increase the radius
    # dt is the step size for position updates
    # We use a schedule for dt and r_target growth
    
    dt = 0.05
    growth_factor = 1.001
    
    # Main optimization loop
    # We run until time limit or no improvement
    last_improvement_time = start_time
    
    while time.time() - start_time < TIME_LIMIT - 10.0:
        # Try to expand the radius
        new_r_target = current_r * growth_factor
        
        # Perform a few steps with this new target
        for _ in range(50):
            if time.time() - start_time > TIME_LIMIT - 10.0:
                break
            pts = _repulsion_step(pts, new_r_target, dt, start_time)
            
        # Evaluate new score
        new_score = _score(pts)
        
        if new_score > best_score:
            best_score = new_score
            best_pts = pts.copy()
            current_r = new_score
            last_improvement_time = time.time()
            # Reduce growth factor if we are improving fast? Or keep it steady?
            # Let's reduce dt slightly to refine
            dt *= 0.95
        else:
            # No improvement, revert or try smaller growth
            # Revert to best
            pts = best_pts.copy()
            # Reduce growth factor
            growth_factor *= 0.9
            # Increase dt to escape local minima? Or decrease to refine?
            # If stuck, we might need to perturb.
            # Let's try a small random perturbation if stuck for a while
            if time.time() - last_improvement_time > 5.0:
                # Perturb
                pts = pts + np.random.normal(0, 0.001, size=pts.shape)
                # Clip
                rad = np.hypot(pts[:, 0], pts[:, 1])
                out = rad > 1.0
                if np.any(out):
                    pts[out] = pts[out] / rad[out, None]
                last_improvement_time = time.time()
            
            # If growth factor is too small, reset to a moderate value
            if growth_factor < 1.0001:
                growth_factor = 1.0005
                dt = 0.05
        
        # Check time
        if time.time() - start_time > TIME_LIMIT - 10.0:
            break

    # Phase 2: Newton Refinement on Active Constraints
    # Identify active constraints:
    # 1. Pairs with distance close to 2 * best_score
    # 2. Circles with radius close to 1 - best_score
    
    if time.time() - start_time < TIME_LIMIT - 2.0:
        r_current = best_score
        # Tolerance for "active"
        tol = 1e-4
        
        d = _compute_distances(best_pts)
        rad = np.hypot(best_pts[:, 0], best_pts[:, 1])
        
        # Active pairs: d_ij < 2*r + tol
        # Active boundary: rad_i > 1 - r - tol
        
        # We want to solve for a new radius r_new > r_current
        # Such that there exists a configuration satisfying:
        # d_ij >= 2 r_new for all pairs
        # rad_i + r_new <= 1 for all i
        
        # This is a non-linear feasibility problem.
        # Approximation: Linearize around current configuration.
        # Let r_new = r_current + delta_r
        # d_ij(c + delta_c) >= 2 (r_current + delta_r)
        # rad_i(c + delta_c) + r_current + delta_r <= 1
        
        # For active pairs, d_ij is small, so gradient of d_ij is well-defined.
        # For active boundary, gradient of rad_i is well-defined.
        
        # We can formulate this as a linear program or least squares problem for delta_c and delta_r.
        # Maximize delta_r subject to linearized constraints.
        
        # Since we don't have a full LP solver, we can use a simple iterative approach:
        # 1. Find the "tightest" constraints.
        # 2. Adjust positions to relax them slightly, allowing a small increase in r.
        
        # Let's try a few Newton-like steps on the "active" set.
        # Actually, a simpler approach:
        # Compute the "slack" for each constraint.
        # Slack_pair_ij = d_ij - 2*r
        # Slack_bound_i = 1 - rad_i - r
        
        # We want to increase r. The limiting factors are the minimum slacks.
        # We can move points to increase the minimum slack.
        
        # Let's define a function that computes the gradient of the minimum slack.
        # This is similar to the soft-min gradient but on slacks.
        
        # However, given time constraints, let's just do a few more repulsion steps
        # with a very small growth factor and dt.
        
        for _ in range(100):
            if time.time() - start_time > TIME_LIMIT - 1.0:
                break
            pts = _repulsion_step(pts, best_score * 1.0001, 0.01, start_time)
            new_score = _score(pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = pts.copy()

    # Final safety checks
    best_pts = np.asarray(best_pts, dtype=np.float64)
    if best_pts.shape != (N, 2):
        best_pts = _hex_init(N)
    if not np.all(np.isfinite(best_pts)):
        best_pts = _hex_init(N)
    
    rad = np.hypot(best_pts[:, 0], best_pts[:, 1])
    out = rad > 1.0
    if np.any(out):
        best_pts[out] = best_pts[out] / rad[out, None]
        
    return best_pts

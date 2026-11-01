"""Pack 50 equal circles in the unit disk via L-BFGS-B with a smooth violation potential."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 110.0  # seconds; evaluator kills at 150s, keep safety margin


def _score(points):
    """Compute the packing score: min(pairwise dist/2, 1 - |c_i|)."""
    if len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    # Container constraint
    r_max = np.max(np.hypot(points[:, 0], points[:, 1]))
    if r_max >= 1.0:
        return 0.0
    container_margin = 1.0 - r_max
    # Pairwise distances
    from scipy.spatial.distance import cdist
    D = cdist(points, points)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D) / 2.0
    return min(min_pair, container_margin)


def _objective_and_grad(x, r_target):
    """
    Objective: maximize the minimum of (pairwise dist/2, 1-|c_i|).
    We use a soft-min (log-sum-exp) of the violations below r_target.
    Minimize: sum of soft violations.
    
    x: (2N,) flattened points
    r_target: the radius we're trying to achieve
    """
    pts = x.reshape(N, 2)
    
    # Container violations: for each point, violation = max(0, |c_i| + r_target - 1)
    dists_origin = np.hypot(pts[:, 0], pts[:, 1])
    container_viol = np.maximum(0.0, dists_origin + r_target - 1.0)
    
    # Pairwise violations: for each pair (i,j), violation = max(0, r_target - |c_i - c_j|/2)
    # We compute all pairwise distances
    # For efficiency, compute pairwise squared distances
    # D_ij = |c_i - c_j|
    # Use broadcasting
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Pairwise violation for each pair
    pair_viol = np.maximum(0.0, r_target - dists / 2.0)
    # Only upper triangle to avoid double counting
    iu = np.triu_indices(N, k=1)
    pair_viol_vals = pair_viol[iu]
    
    # Soft min: use log-sum-exp for smoothness
    beta = 50.0  # sharpness parameter
    
    # Combine all violations
    all_viol = np.concatenate([container_viol, pair_viol_vals])
    
    if len(all_viol) == 0:
        return 0.0, np.zeros_like(x)
    
    # Soft max (we want to minimize the max violation)
    # Use log-sum-exp: log(sum(exp(beta * v))) / beta
    # This approximates max(v) as beta -> inf
    max_v = np.max(all_viol)
    if max_v < 1e-12:
        return 0.0, np.zeros_like(x)
    
    exp_v = np.exp(beta * (all_viol - max_v))
    soft_max = max_v + np.log(np.sum(exp_v)) / beta
    
    # Gradient
    # d(soft_max)/d(all_viol) = exp(beta * (v - max_v)) / sum(exp(beta * (v - max_v)))
    weights = exp_v / np.sum(exp_v)
    
    grad_all = np.zeros_like(all_viol)
    grad_all[:] = weights * beta  # d(soft_max)/dv = beta * exp(beta*(v-max)) / sum(...)
    
    # Actually: d/dv [max_v + log(sum(exp(beta*(vi - max_v))))/beta]
    # = (1/beta) * d/dv log(sum(exp(beta*(vi-max_v))))
    # = (1/beta) * sum_j [beta * exp(beta*(vj-max_v)) / sum_j exp(beta*(vj-max_v))] * d(vi)/dv
    # For vi itself: (1/beta) * beta * exp(beta*(vi-max_v)) / sum = exp(beta*(vi-max_v)) / sum
    # So d(soft_max)/d(vi) = exp(beta*(vi-max_v)) / sum(exp(beta*(vj-max_v)))
    
    grad_all = exp_v / np.sum(exp_v)
    
    # Now back-propagate to x
    grad_x = np.zeros_like(x)
    
    # Container part: v_i = max(0, |c_i| + r_target - 1)
    # d(v_i)/d(c_i) = c_i / |c_i| (if |c_i| + r_target > 1)
    for i in range(N):
        w = grad_all[i]
        if w > 1e-15:
            d = dists_origin[i]
            if d > 1e-12:
                grad_x[2*i] += w * pts[i, 0] / d
                grad_x[2*i+1] += w * pts[i, 1] / d
    
    # Pairwise part
    for idx, (ii, jj) in enumerate(zip(iu[0], iu[1])):
        w = grad_all[N + idx]
        if w > 1e-15:
            d = dists[ii, jj]
            if d > 1e-12:
                # v = max(0, r_target - d/2)
                # d(v)/d(c_i) = -0.5 * (c_i - c_j) / d
                # d(v)/d(c_j) = +0.5 * (c_i - c_j) / d
                dx = (pts[ii] - pts[jj]) / d
                grad_x[2*ii] += w * (-0.5 * dx[0])
                grad_x[2*ii+1] += w * (-0.5 * dx[1])
                grad_x[2*jj] += w * (0.5 * dx[0])
                grad_x[2*jj+1] += w * (0.5 * dx[1])
    
    return soft_max, grad_x


def _hex_init():
    """Generate an initial hexagonal packing configuration."""
    # Start with a hex grid and select N points
    pts = []
    r = 0.05
    while len(pts) < N + 20:
        for j in range(-int(3/r), int(3/r)+1):
            for i in range(-int(3/r), int(3/r)+1):
                x = (i + 0.5*(j % 2)) * r * np.sqrt(3)
                y = j * r * 1.5
                if np.hypot(x, y) < 0.95:
                    pts.append([x, y])
        r *= 1.05
    pts = np.array(pts[:N])
    return pts


def _ring_init(n_rings=3):
    """Generate points on concentric rings."""
    pts = [[0.0, 0.0]]
    for ring in range(1, n_rings + 1):
        n_pts = 6 * ring
        radius = ring * 0.15
        for k in range(n_pts):
            theta = 2 * np.pi * k / n_pts
            pts.append([radius * np.cos(theta), radius * np.sin(theta)])
    # Add more if needed
    while len(pts) < N:
        theta = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0, 0.9)
        pts.append([r * np.cos(theta), r * np.sin(theta)])
    return np.array(pts[:N])


def construct():
    t_start = time.time()
    
    # Try multiple initializations
    best_pts = None
    best_score = 0.0
    
    inits = [
        _hex_init(),
        _ring_init(3),
        _ring_init(4),
        np.random.randn(N, 2) * 0.5,
    ]
    
    # Also try a structured spiral
    theta = np.linspace(0, 12 * np.pi, N)
    r_spiral = np.linspace(0.1, 0.85, N)
    inits.append(np.column_stack([r_spiral * np.cos(theta), r_spiral * np.sin(theta)]))
    
    for init_pts in inits:
        if time.time() - t_start > DEADLINE - 30:
            break
        
        # Binary search on radius
        lo = 0.05
        hi = 0.15
        
        # First, try to find a feasible radius via L-BFGS-B
        # We'll do a few iterations of: pick r, minimize violation, check feasibility
        
        current_r = 0.10
        x0 = init_pts.flatten()
        
        for _iter in range(5):
            if time.time() - t_start > DEADLINE - 20:
                break
            
            # Try to solve for radius current_r
            res = minimize(
                _objective_and_grad,
                x0,
                args=(current_r,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-10},
            )
            
            if res.success or res.fun < 1e-6:
                # Feasible for current_r, try higher
                lo = current_r
                hi = min(hi * 1.1, 0.16)
                current_r = (lo + hi) / 2
                x0 = res.x
            else:
                # Not feasible, try lower
                hi = current_r
                lo = (lo + hi) / 2
                current_r = (lo + hi) / 2
                x0 = res.x
        
        # Final evaluation
        final_pts = x0.reshape(N, 2)
        # Clip to ensure inside unit disk
        norms = np.hypot(final_pts[:, 0], final_pts[:, 1])
        mask = norms > 1.0
        if np.any(mask):
            final_pts[mask] = final_pts[mask] / norms[mask, np.newaxis] * 0.999
        
        sc = _score(final_pts)
        if sc > best_score:
            best_score = sc
            best_pts = final_pts.copy()
    
    # If we didn't find anything good, use hex init
    if best_pts is None or best_score < 0.05:
        best_pts = _hex_init()
        best_score = _score(best_pts)
    
    return best_pts

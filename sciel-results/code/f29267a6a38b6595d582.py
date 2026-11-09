import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.spatial import cKDTree

N = 300

# EVOLVE-BLOCK-START

def _get_score(points):
    """
    Computes the score (common radius r) for a given set of points.
    r = min( min_{i<j} ||c_i - c_j|| / 2, min_i (1 - ||c_i||) )
    """
    if points is None or len(points) != N:
        return 0.0
    
    if not np.all(np.isfinite(points)):
        return 0.0

    norms = np.linalg.norm(points, axis=1)
    boundary_r = 1.0 - np.max(norms)
    
    if boundary_r < 0:
        return 0.0

    d = pdist(points)
    if len(d) == 0:
        pair_r = float('inf')
    else:
        pair_r = np.min(d) / 2.0

    r = min(pair_r, boundary_r)
    return max(0.0, r)

def _hexagonal_lattice(spacing):
    """
    Generate a hexagonal lattice of points inside the unit disk.
    """
    k = int(1.0 / spacing) + 5
    pts = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0:
                pts.append((x, y))
    
    if len(pts) < N:
        return None
    
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists, kind="stable")[:N]
    return pts[idx]

def _concentric_rings(n_rings):
    """Generate concentric ring configuration."""
    if n_rings <= 0:
        return None
    if n_rings == 1:
        angles = 2 * np.pi * np.arange(N) / N
        return np.column_stack([0.5 * np.cos(angles), 0.5 * np.sin(angles)])
    pts = [[0.0, 0.0]]
    remaining = N - 1
    ring_radii = np.linspace(0.1, 0.95, n_rings - 1)
    weights = ring_radii
    weights /= np.sum(weights)
    counts = np.round(weights * remaining).astype(int)
    diff = remaining - np.sum(counts)
    if diff != 0:
        idx = np.argmax(counts) if diff > 0 else np.argmax(-counts)
        counts[idx] += diff
    for i, count in enumerate(counts):
        if count <= 0:
            continue
        r = ring_radii[i]
        angles = 2 * np.pi * np.arange(count) / count
        offset = 0 if i % 2 == 0 else np.pi / count
        x = r * np.cos(angles + offset)
        y = r * np.sin(angles + offset)
        for j in range(count):
            pts.append([x[j], y[j]])
    if len(pts) != N:
        return None
    return np.array(pts)

def _smooth_obj_and_grad(x, alpha, k_nn=12):
    """
    Smooth objective using only k nearest neighbors per point.
    Objective: -log(sum over nn pairs of exp(-alpha * d_ij)) + boundary penalty
    This is much faster than all-pairs for N=300.
    """
    points = x.reshape(-1, 2)
    n = len(points)
    
    # Build KDTree for nearest neighbors
    tree = cKDTree(points)
    # Query k_nn+1 neighbors (including self)
    dists_nn, idxs_nn = tree.query(points, k=k_nn+1)
    
    # Collect unique pairs (i, j) where j is a neighbor of i and i < j
    pair_set = set()
    for i in range(n):
        for j_idx in range(1, k_nn+1):  # skip self (index 0)
            j = idxs_nn[i, j_idx]
            if i < j:
                pair_set.add((i, j))
    
    pairs = list(pair_set)
    n_pairs = len(pairs)
    
    if n_pairs == 0:
        return 0.0, np.zeros_like(x)
    
    # Compute distances for these pairs
    pi = np.array([p[0] for p in pairs])
    pj = np.array([p[1] for p in pairs])
    
    diff = points[pi] - points[pj]  # shape (n_pairs, 2)
    d = np.linalg.norm(diff, axis=1)  # shape (n_pairs,)
    d_safe = np.maximum(d, 1e-10)
    
    # exp terms
    exp_vals = np.exp(-alpha * d)
    S = np.sum(exp_vals)
    
    if S < 1e-300:
        return 0.0, np.zeros_like(x)
    
    obj_val = -np.log(S)
    
    # Gradient: for each pair (i,j) with weight w_ij = exp(-alpha*d_ij)/S
    # d/dx_i (-log S) = -1/S * sum_{j in nn(i)} exp(-alpha*d_ij) * (-alpha) * (x_i - x_j)/d_ij
    # = alpha/S * sum_{j in nn(i)} exp(-alpha*d_ij) * (x_i - x_j)/d_ij
    
    weights = alpha * exp_vals / S  # shape (n_pairs,)
    
    grad_points = np.zeros_like(points)
    for k in range(n_pairs):
        i, j = pairs[k]
        w = weights[k]
        dir_vec = diff[k] / d_safe[k]
        grad_points[i] += w * dir_vec
        grad_points[j] -= w * dir_vec
    
    # Boundary penalty
    norms = np.linalg.norm(points, axis=1)
    violation = np.maximum(0.0, norms - 1.0)
    obj_val += 50.0 * np.sum(violation ** 2)
    
    b_mask = norms > 1.0
    if np.any(b_mask):
        norms_safe = np.where(norms[b_mask] < 1e-8, 1e-8, norms[b_mask])
        grad_points[b_mask] += 100.0 * (norms[b_mask] - 1.0) * (points[b_mask] / norms_safe[:, np.newaxis])
    
    return obj_val, grad_points.flatten()

def _constrained_objective(x, r_target):
    """
    Objective for constrained optimization: minimize sum of constraint violations.
    Constraints: ||c_i - c_j|| >= 2*r_target for all i<j, ||c_i|| + r_target <= 1
    """
    points = x.reshape(-1, 2)
    
    # Pairwise distances
    d = pdist(points)
    pair_violations = np.maximum(0, 2*r_target - d)
    
    # Boundary violations
    norms = np.linalg.norm(points, axis=1)
    boundary_violations = np.maximum(0, norms + r_target - 1.0)
    
    return np.sum(pair_violations**2) + np.sum(boundary_violations**2)

def _constrained_gradient(x, r_target):
    """Gradient of the constraint violation objective."""
    points = x.reshape(-1, 2)
    n = len(points)
    
    # Pairwise
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    dists_safe = np.where(dists < 1e-8, 1e-8, dists)
    
    mask = ~np.eye(n, dtype=bool)
    
    # Pairwise gradient contribution
    # For pair (i,j): violation = max(0, 2r - d_ij)
    # d/dx_i violation^2 = 2*violation * (-1) * d/dx_i(2r - d_ij) = 2*violation * (x_i - x_j)/d_ij
    # But only if violation > 0, i.e., d_ij < 2r
    
    pair_grad = np.zeros_like(points)
    for i in range(n):
        for j in range(i+1, n):
            d_ij = dists[i, j]
            if d_ij < 2 * r_target:
                viol = 2 * r_target - d_ij
                dir_vec = (points[i] - points[j]) / dists_safe[i, j]
                pair_grad[i] += 2.0 * viol * dir_vec
                pair_grad[j] -= 2.0 * viol * dir_vec
    
    # Boundary gradient
    norms = np.linalg.norm(points, axis=1)
    boundary_grad = np.zeros_like(points)
    b_mask = norms + r_target > 1.0
    if np.any(b_mask):
        norms_safe = np.where(norms[b_mask] < 1e-8, 1e-8, norms[b_mask])
        viol = norms[b_mask] + r_target - 1.0
        boundary_grad[b_mask] = 2.0 * viol * (points[b_mask] / norms_safe[:, np.newaxis])
    
    return (pair_grad + boundary_grad).flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_points = None
    best_score = 0.0
    
    # Generate initial candidates
    spacings = [0.105, 0.106, 0.107, 0.108, 0.110]
    
    candidates = []
    for sp in spacings:
        pts = _hexagonal_lattice(sp)
        if pts is not None:
            noise = np.random.normal(0, 0.002, pts.shape)
            pts_noisy = pts + noise
            candidates.append(pts_noisy)
    
    for nr in [10, 15, 20]:
        pts = _concentric_rings(nr)
        if pts is not None:
            noise = np.random.normal(0, 0.002, pts.shape)
            candidates.append(pts + noise)
    
    if not candidates:
        angles = 2 * np.pi * np.arange(N) / N
        return np.column_stack([0.5 * np.cos(angles), 0.5 * np.sin(angles)])
    
    # Phase 1: Fast smooth optimization with L-BFGS-B using k-NN
    alpha = 30.0
    options_fast = {
        'maxiter': 150,
        'ftol': 1e-10,
        'gtol': 1e-6,
        'maxls': 30
    }
    
    for idx, init_pts in enumerate(candidates):
        if time.time() > deadline - 60.0:
            break
            
        x0 = init_pts.flatten()
        
        try:
            result = minimize(
                _smooth_obj_and_grad, 
                x0, 
                args=(alpha,),
                method='L-BFGS-B', 
                jac=True, 
                options=options_fast
            )
            
            opt_points = result.x.reshape(-1, 2)
            score = _get_score(opt_points)
            
            if score > best_score:
                best_score = score
                best_points = opt_points.copy()
                
        except Exception:
            pass
    
    # Phase 2: Constrained refinement using SLSQP on the best candidate
    if best_points is not None and time.time() < deadline - 30.0:
        # Try to increase radius using constrained optimization
        current_r = best_score
        # Start with a slightly smaller target to ensure feasibility
        r_target = current_r * 0.99
        
        for attempt in range(3):
            if time.time() > deadline - 10.0:
                break
            
            # Binary search for maximum feasible r
            r_lo = current_r * 0.95
            r_hi = current_r * 1.05
            best_feasible_r = current_r
            best_feasible_pts = best_points.copy()
            
            for _ in range(8):
                if time.time() > deadline - 10.0:
                    break
                    
                r_mid = (r_lo + r_hi) / 2.0
                
                try:
                    res = minimize(
                        _constrained_objective,
                        best_points.flatten(),
                        args=(r_mid,),
                        jac=_constrained_gradient,
                        method='SLSQP',
                        options={'maxiter': 50, 'ftol': 1e-8}
                    )
                    
                    opt_pts = res.x.reshape(-1, 2)
                    actual_score = _get_score(opt_pts)
                    
                    # Check if constraints are satisfied
                    if actual_score >= r_mid * 0.999:
                        # Feasible - try larger r
                        r_lo = r_mid
                        best_feasible_r = r_mid
                        best_feasible_pts = opt_pts.copy()
                    else:
                        r_hi = r_mid
                        
                except Exception:
                    r_hi = r_mid
                
                if r_hi - r_lo < 1e-5:
                    break
            
            if best_feasible_r > best_score:
                best_score = best_feasible_r
                best_points = best_feasible_pts.copy()
                current_r = best_score
            else:
                break
    
    # Phase 3: Final local refinement with higher alpha
    if best_points is not None and time.time() < deadline - 5.0:
        alpha_high = 80.0
        try:
            result = minimize(
                _smooth_obj_and_grad, 
                best_points.flatten(), 
                args=(alpha_high,),
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-7, 'maxls': 20}
            )
            opt_points = result.x.reshape(-1, 2)
            score = _get_score(opt_points)
            if score > best_score:
                best_score = score
                best_points = opt_points.copy()
        except Exception:
            pass
    
    if best_points is None:
        best_points = candidates[0]
        
    return best_points

# EVOLVE-BLOCK-END

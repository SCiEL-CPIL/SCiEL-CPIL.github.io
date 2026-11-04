import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 300

def _hex_init(n_points, seed=0):
    """Generate a hexagonal lattice and select n_points best points."""
    rng = np.random.default_rng(seed)
    best_pts = None
    best_r = 0.0
    for attempt in range(12):
        # Vary spacing slightly to explore different lattice densities
        spacing = 2.0 / (np.sqrt(n_points) + 1.0) * (0.92 + 0.03 * attempt)
        k = int(1.0 / spacing) + 5
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                if r <= 1.0 - spacing / 2.0:
                    rows.append((x, y, r))
        if len(rows) < n_points:
            continue
        pts = np.array([(r[0], r[1]) for r in rows])
        radii = np.array([r[2] for r in rows])
        order = np.argsort(radii, kind="stable")
        selected = pts[order[:n_points]]
        if attempt > 0:
            # Add small noise to break symmetry and escape local minima
            noise_scale = spacing * (0.02 + 0.01 * attempt)
            selected = selected + rng.normal(0, noise_scale, selected.shape)
            norms = np.linalg.norm(selected, axis=1)
            mask = norms < 1.0
            selected = selected[mask]
            if len(selected) < n_points:
                continue
            selected = selected[:n_points]
        d = np.linalg.norm(selected[:, np.newaxis, :] - selected[np.newaxis, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair = d.min() / 2.0
        min_bound = 1.0 - np.linalg.norm(selected, axis=1).max()
        r = min(min_pair, min_bound)
        if r > best_r:
            best_r = r
            best_pts = selected.copy()
    if best_pts is None:
        spacing = 2.0 / (np.sqrt(n_points) + 1.0)
        k = int(1.0 / spacing) + 4
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                    rows.append((x, y))
        pts = np.array(rows)
        order = np.argsort(np.linalg.norm(pts, axis=1), kind="stable")
        best_pts = pts[order[:n_points]]
    return best_pts


def _compute_radius(pts):
    """Compute the packing radius for given points."""
    norms = np.linalg.norm(pts, axis=1)
    min_bound = 1.0 - norms.max()
    if min_bound <= 0:
        return 0.0
    d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    min_pair = d.min() / 2.0
    return min(min_pair, min_bound)


def _force_relax(pts, max_iter=200, dt=0.002):
    """
    Force-based relaxation: push apart close pairs, pull inward boundary circles.
    Uses KD-tree for efficient neighbor lookup.
    """
    pts = pts.copy()
    n = len(pts)
    r_current = _compute_radius(pts)
    
    for it in range(max_iter):
        if time.time() > _deadline - 5.0:
            break
        
        # Build KD-tree for neighbor queries
        tree = cKDTree(pts)
        
        # Query neighbors within 4*r_current (to catch all close pairs)
        query_radius = max(4.0 * r_current, 0.1)
        neighbors = tree.query_pairs(query_radius)
        
        # Compute forces
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion for close pairs
        for i, j in neighbors:
            diff = pts[i] - pts[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                # Random small push for coincident points
                forces[i] += np.random.randn(2) * 0.01
                forces[j] -= np.random.randn(2) * 0.01
                continue
            if dist < 2.0 * r_current * 1.05:
                # Repulsive force proportional to overlap
                overlap = 2.0 * r_current - dist
                unit = diff / dist
                force_mag = 0.5 * overlap
                forces[i] += force_mag * unit
                forces[j] -= force_mag * unit
        
        # Boundary force: push inward if too close to boundary
        norms = np.linalg.norm(pts, axis=1)
        for i in range(n):
            if norms[i] > 1.0 - r_current * 1.05:
                if norms[i] < 1e-10:
                    forces[i] += np.random.randn(2) * 0.01
                else:
                    # Pull toward center
                    pull = (1.0 - r_current - norms[i]) * 0.5
                    forces[i] += pull * pts[i] / norms[i]
        
        # Apply forces with damping
        pts += dt * forces
        
        # Clamp to stay inside unit circle
        norms = np.linalg.norm(pts, axis=1)
        mask = norms > 1.0 - 1e-10
        if np.any(mask):
            pts[mask] *= (1.0 - 1e-10) / norms[mask, np.newaxis]
        
        new_r = _compute_radius(pts)
        if new_r > r_current:
            r_current = new_r
        else:
            # If no improvement after some iterations, reduce step size
            if it > 50:
                dt *= 0.9
    
    return pts


def _get_active_edges(pts, r, max_edges=500):
    """Get the tightest pairwise constraints and boundary constraints."""
    n = len(pts)
    norms = np.linalg.norm(pts, axis=1)
    
    # Pairwise distances
    d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    
    # Get indices of smallest distances
    flat_idx = np.argsort(d.ravel())[:max_edges]
    rows = flat_idx // n
    cols = flat_idx % n
    # Only keep i < j
    mask = rows < cols
    rows = rows[mask]
    cols = cols[mask]
    
    # Boundary violations
    boundary_idx = np.argsort(norms)[-min(50, n):]
    
    return rows, cols, boundary_idx


def _local_polish(pts, r_target, max_iter=100):
    """
    Local polish focusing on the tightest constraints using L-BFGS-B
    with a very low temperature soft-min.
    """
    if time.time() > _deadline - 3.0:
        return pts
    
    T = 0.00005
    x0 = pts.ravel()
    
    try:
        result = minimize(
            _objective_only,
            x0,
            args=(T,),
            jac=_grad_only,
            method='L-BFGS-B',
            options={
                'maxiter': max_iter,
                'ftol': 1e-14,
                'gtol': 1e-12,
                'maxls': 100,
                'maxfun': 500,
            }
        )
        new_pts = result.x.reshape(-1, 2)
        norms = np.linalg.norm(new_pts, axis=1)
        if np.all(norms < 1.0) and np.all(np.isfinite(new_pts)):
            new_r = _compute_radius(new_pts)
            if new_r >= r_target - 1e-10:
                return new_pts
    except Exception:
        pass
    
    return pts


def _anneal_restart(pts, r_current, rng, strength=0.02):
    """
    Perturb the packing and try to recover or improve.
    """
    if time.time() > _deadline - 10.0:
        return pts, r_current
    
    # Perturb a random subset of points
    n = len(pts)
    n_perturb = max(10, n // 10)
    indices = rng.choice(n, size=n_perturb, replace=False)
    perturbed = pts.copy()
    perturbed[indices] += rng.normal(0, strength, (n_perturb, 2))
    
    # Clamp to unit circle
    norms = np.linalg.norm(perturbed, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        perturbed[mask] *= (1.0 - 1e-10) / norms[mask, np.newaxis]
    
    # Try to recover via force relaxation
    relaxed = _force_relax(perturbed, max_iter=100, dt=0.001)
    r_relaxed = _compute_radius(relaxed)
    
    if r_relaxed > r_current:
        return relaxed, r_relaxed
    
    # Try local polish from perturbed state
    polished = _local_polish(perturbed, r_current, max_iter=50)
    r_polished = _compute_radius(polished)
    
    if r_polished > r_current:
        return polished, r_polished
    
    return pts, r_current


def _soft_min_logsumexp(values, T):
    """Stable soft minimum: -T * logsumexp(-values/T)"""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    neg_vals = -values / T
    max_neg = np.max(neg_vals)
    if max_neg > 700:
        return values.min()
    exp_shifted = np.exp(neg_vals - max_neg)
    log_sum = np.log(np.sum(exp_shifted)) + max_neg
    return -T * log_sum


def _objective_and_grad(x, T):
    """
    Objective: minimize -softmin(constraints) where constraints are:
      - pairwise half-distances d_ij/2 for i<j
      - boundary distances 1 - |c_i|
    
    Uses log-sum-exp soft-min for differentiability.
    Optimized with einsum and reused distance matrix.
    """
    pts = x.reshape(-1, 2)
    n = len(pts)
    
    # Compute differences once
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    
    # Boundary constraints: 1 - |c_i|
    norms = np.linalg.norm(pts, axis=1)
    boundary = 1.0 - norms  # shape (n,)
    
    # Pairwise distances
    d = np.linalg.norm(diff, axis=2)
    # Take upper triangle (i < j)
    iu = np.triu_indices(n, k=1)
    pair_half = d[iu] / 2.0
    
    # All constraint values
    all_vals = np.concatenate([pair_half, boundary])
    
    # Soft minimum
    obj = -_soft_min_logsumexp(all_vals, T)
    
    # Gradient computation
    neg_vals_T = -all_vals / T
    max_nv = np.max(neg_vals_T)
    if max_nv > 700:
        min_idx = np.argmin(all_vals)
        grad_all = np.zeros_like(all_vals)
        grad_all[min_idx] = -1.0
    else:
        exp_shifted = np.exp(neg_vals_T - max_nv)
        weights = exp_shifted / np.sum(exp_shifted)
        grad_all = -weights
    
    # Gradient for pairwise constraints
    n_pairs = len(pair_half)
    
    # Unit vectors for pairwise differences
    safe_dists = np.where(d > 1e-10, d, 1.0)
    unit_vecs = diff / safe_dists[:, :, np.newaxis]  # (n, n, 2)
    
    # Build weight matrix for pairs
    W = np.zeros((n, n))
    W[iu] = grad_all[:n_pairs]
    W[iu[1], iu[0]] = grad_all[:n_pairs]
    
    # Compute pairwise gradient contribution using einsum
    # grad_pairs[i] = 0.5 * sum_j W[i,j] * unit_vecs[i,j]
    weighted_unit = W[:, :, np.newaxis] * unit_vecs  # (n, n, 2)
    grad_pairs = 0.5 * np.sum(weighted_unit, axis=1)  # (n, 2)
    
    # Boundary gradient
    safe_norms = np.where(norms > 1e-10, norms, 1.0)
    grad_boundary = -pts / safe_norms[:, np.newaxis] * grad_all[n_pairs:][:, np.newaxis]
    
    # Total gradient
    grad_pts = grad_pairs + grad_boundary
    grad_obj = grad_pts.ravel()
    
    return obj, grad_obj


def _objective_only(x, T):
    """Objective function without gradient (for methods that don't use jac)."""
    return _objective_and_grad(x, T)[0]


def _grad_only(x, T):
    """Gradient function."""
    return _objective_and_grad(x, T)[1]


# Global deadline for force_relax to check
_deadline = None


def construct():
    global _deadline
    start_time = time.time()
    _deadline = start_time + 145.0  # 5s safety margin before 150s limit
    
    rng = np.random.default_rng(12345)
    
    # Initial hexagonal packing with multiple seeds
    best_pts = None
    best_r = 0.0
    for seed in [42, 0, 123, 7]:
        if time.time() > _deadline - 40.0:
            break
        pts = _hex_init(N, seed=seed)
        r = _compute_radius(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    
    if best_pts is None:
        best_pts = _hex_init(N, seed=42)
        best_r = _compute_radius(best_pts)
    
    # Force-based relaxation to quickly improve
    best_pts = _force_relax(best_pts, max_iter=300, dt=0.003)
    best_r = _compute_radius(best_pts)
    
    x0 = best_pts.ravel()
    
    # Temperature annealing schedule
    temperatures = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002]
    
    for T in temperatures:
        if time.time() > _deadline - 15.0:
            break
        
        try:
            # Use L-BFGS-B with gradient
            result = minimize(
                _objective_only,
                x0,
                args=(T,),
                jac=_grad_only,
                method='L-BFGS-B',
                options={
                    'maxiter': 200,
                    'ftol': 1e-13,
                    'gtol': 1e-11,
                    'maxls': 50,
                    'maxfun': 600,
                }
            )
            if time.time() > _deadline - 10.0:
                break
            new_pts = result.x.reshape(-1, 2)
            # Check validity
            norms = np.linalg.norm(new_pts, axis=1)
            if np.any(norms >= 1.0) or not np.all(np.isfinite(new_pts)):
                continue
            new_r = _compute_radius(new_pts)
            if new_r > best_r:
                best_r = new_r
                best_pts = new_pts.copy()
                x0 = new_pts.ravel()
        except Exception:
            continue
    
    # Force relaxation after optimization
    if time.time() < _deadline - 10.0:
        best_pts = _force_relax(best_pts, max_iter=200, dt=0.002)
        best_r = _compute_radius(best_pts)
        x0 = best_pts.ravel()
    
    # Anneal restarts to escape local minima
    for restart in range(5):
        if time.time() > _deadline - 12.0:
            break
        
        strength = 0.015 * (1.0 - restart * 0.15)
        new_pts, new_r = _anneal_restart(best_pts, best_r, rng, strength=strength)
        
        if new_r > best_r:
            best_r = new_r
            best_pts = new_pts
            x0 = new_pts.ravel()
            
            # Polish after improvement
            if time.time() < _deadline - 8.0:
                best_pts = _local_polish(best_pts, best_r, max_iter=80)
                best_r = _compute_radius(best_pts)
                x0 = best_pts.ravel()
    
    # Final high-precision polish
    if time.time() < _deadline - 5.0:
        best_pts = _local_polish(best_pts, best_r, max_iter=150)
        best_r = _compute_radius(best_pts)
    
    # Ultra-final polish
    if time.time() < _deadline - 3.0:
        try:
            T_final = 0.00001
            result = minimize(
                _objective_only,
                best_pts.ravel(),
                args=(T_final,),
                jac=_grad_only,
                method='L-BFGS-B',
                options={
                    'maxiter': 200,
                    'ftol': 1e-14,
                    'gtol': 1e-13,
                    'maxls': 100,
                    'maxfun': 800,
                }
            )
            new_pts = result.x.reshape(-1, 2)
            norms = np.linalg.norm(new_pts, axis=1)
            if np.all(norms < 1.0) and np.all(np.isfinite(new_pts)):
                new_r = _compute_radius(new_pts)
                if new_r > best_r:
                    best_r = new_r
                    best_pts = new_pts.copy()
        except Exception:
            pass
    
    # Ensure validity
    norms = np.linalg.norm(best_pts, axis=1)
    mask = norms < 1.0
    if not np.all(mask) or not np.all(np.isfinite(best_pts)):
        best_pts = _hex_init(N, seed=0)
        best_r = _compute_radius(best_pts)
    
    return best_pts

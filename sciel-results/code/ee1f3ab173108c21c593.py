"""Improved packing of 50 equal circles in a unit circle using soft-min optimization."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
_DEADLINE = time.time() + 130.0  # 20s safety margin before 150s kill


def _score(pts):
    """Compute the valid packing radius (or 0 if invalid)."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    
    dists_origin = np.hypot(pts[:, 0], pts[:, 1])
    if np.any(dists_origin > 1.0 + 1e-12):
        return 0.0
    
    r_cont = 1.0 - np.max(dists_origin)
    if r_cont < -1e-12:
        return 0.0
    
    if N < 2:
        return max(0.0, r_cont)
        
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, np.inf)
    r_pair = 0.5 * np.min(dists)
    
    return max(0.0, min(r_cont, r_pair))


def _ring_init():
    """
    Generate ring-based initialization matching known optimal structure:
    1 (center) + 6 + 12 + 18 + 13 = 50
    """
    rings = [
        (0.0, 1),    # center
        (0.2516, 6),    # 6 circles
        (0.5032, 12),   # 12 circles
        (0.7548, 18),   # 18 circles
        (0.945, 13),   # 13 circles (outermost, partial ring)
    ]
    
    pts = []
    for ring_idx, (radius, count) in enumerate(rings):
        if count == 1:
            pts.append([0.0, 0.0])
        else:
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
            # Add angular offset per ring to stagger
            offset = ring_idx * (np.pi / 6.0)
            for a in angles:
                a += offset
                pts.append([radius * np.cos(a), radius * np.sin(a)])
    
    pts = np.array(pts[:N])
    return pts


def _hex_lattice_init():
    """Generate a hexagonal lattice packing as initial guess."""
    best_pts = None
    best_score = 0.0
    
    # Search over different lattice spacings
    for spacing in np.linspace(0.22, 0.30, 40):
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points = points[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            r = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(r, kind="stable")[:N]
            pts = points[idx]
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
    
    if best_pts is not None:
        return best_pts
    
    # Fallback: random
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 0.81, N))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _random_init(seed=0):
    """Random initialization in unit disk."""
    rng = np.random.RandomState(seed)
    angles = rng.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(rng.uniform(0, 0.9, N))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _jittered_ring_init():
    """Ring init with small random perturbations to break symmetry."""
    base = _ring_init()
    rng = np.random.RandomState(7)
    jitter = rng.uniform(-0.02, 0.02, base.shape)
    pts = base + jitter
    # Project back to unit disk
    radii = np.hypot(pts[:, 0], pts[:, 1])
    mask = radii > 1.0
    if np.any(mask):
        pts[mask] = pts[mask] / radii[mask, np.newaxis]
    return pts


def _soft_min_obj_and_grad(x, beta):
    """
    Soft-min objective with gradient for L-BFGS-B.
    Minimizes -soft_min(min_pair_dist/2, 1 - max_dist_to_origin)
    Uses vectorized computation for speed.
    """
    pts = x.reshape(N, 2)
    
    # Pairwise distances (vectorized)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, np.inf)
    
    # Container distance
    dists_origin = np.hypot(pts[:, 0], pts[:, 1])
    r_cont = 1.0 - np.max(dists_origin)
    
    # Soft-min of pairwise distances
    finite_mask = np.isfinite(dists)
    finite_dists = dists[finite_mask]
    
    if len(finite_dists) == 0:
        return 0.0, np.zeros_like(x)
    
    # soft_min = -beta * log(sum(exp(-d/beta)))
    val = -finite_dists / beta
    max_val = np.max(val)
    log_sum_exp = max_val + np.log(np.sum(np.exp(val - max_val)))
    soft_min_pair = -beta * log_sum_exp
    
    r_pair = 0.5 * soft_min_pair
    combined = min(r_pair, r_cont)
    
    # Gradient computation (vectorized)
    # Build weight matrix: w[i,j] = exp(-d_ij/beta) for i != j
    w = np.zeros((N, N))
    np.fill_diagonal(dists, 0)  # temporary for indexing
    w = np.exp(-dists / beta)
    np.fill_diagonal(w, 0)
    
    w_sum = np.sum(w)
    grad_obj = np.zeros((N, 2))
    
    if w_sum > 0:
        # grad_soft_min[i] = beta * sum_j w[i,j] * (pts[i] - pts[j]) / (d_ij * w_sum)
        # Vectorized: for each i, sum over j
        # diff[i,j] = pts[i] - pts[j]
        # We need: grad[i] = beta/w_sum * sum_j w[i,j] * diff[i,j] / d_ij
        
        # Safe division
        safe_dists = np.where(dists > 1e-12, dists, 1.0)
        weights = w / safe_dists  # w[i,j] / d_ij
        np.fill_diagonal(weights, 0)
        
        # grad[i] = beta/w_sum * sum_j weights[i,j] * (pts[i] - pts[j])
        # = beta/w_sum * (sum_j weights[i,j]*pts[i] - sum_j weights[i,j]*pts[j])
        # = beta/w_sum * (pts[i]*sum_j weights[i,j] - weights[i,:] @ pts)
        
        row_sums = np.sum(weights, axis=1, keepdims=True)  # (N, 1)
        grad_soft = beta / w_sum * (pts * row_sums - weights @ pts)
        
        # Objective is -combined; gradient of -r_pair = -0.5 * grad_soft_min
        grad_obj = -0.5 * grad_soft
    
    # Gradient of -r_cont
    # r_cont = 1 - max(dists_origin), so -r_cont = max(dists_origin) - 1
    # d(max_dist)/d(pts_i) = pts_i/|pts_i| for the argmax point
    idx_max = np.argmax(dists_origin)
    if dists_origin[idx_max] > 1e-10:
        grad_obj[idx_max] -= pts[idx_max] / dists_origin[idx_max]
    
    # Boundary penalty for points outside unit disk
    outside_mask = dists_origin > 1.0
    if np.any(outside_mask):
        for i in np.where(outside_mask)[0]:
            pen = 10.0 * (dists_origin[i] - 1.0)
            grad_obj[i] += pen * pts[i] / dists_origin[i]
            # Also add to objective
            combined += pen
    
    return -combined, grad_obj.flatten()


def _optimize_from_start(pts0, time_limit, betas=None):
    """
    Progressive soft-min optimization with L-BFGS-B.
    """
    if betas is None:
        betas = [0.15, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
    
    pts = pts0.copy()
    best_score = _score(pts)
    best_pts = pts.copy()
    
    for beta in betas:
        if time.time() > time_limit:
            break
        
        def obj(x):
            val, _ = _soft_min_obj_and_grad(x, beta)
            return val
        
        def grad(x):
            _, g = _soft_min_obj_and_grad(x, beta)
            return g
        
        try:
            # Determine maxiter based on remaining time
            remaining = max(1.0, time_limit - time.time())
            maxiter = max(50, min(500, int(remaining * 10)))
            
            res = minimize(
                obj,
                grad,
                pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            new_pts = res.x.reshape(N, 2)
            
            # Project back to unit disk
            radii = np.hypot(new_pts[:, 0], new_pts[:, 1])
            mask = radii > 1.0
            if np.any(mask):
                new_pts[mask] = new_pts[mask] / radii[mask, np.newaxis]
            
            new_score = _score(new_pts)
            
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
                pts = new_pts.copy()
        except Exception:
            pass
    
    return best_pts


def _local_search(pts0, time_limit):
    """
    Local search: try small perturbations and keep improvements.
    """
    pts = pts0.copy()
    best_score = _score(pts)
    best_pts = pts.copy()
    rng = np.random.RandomState(12345)
    
    step_size = 0.05
    no_improve = 0
    
    while time.time() < time_limit and no_improve < 200:
        # Generate random perturbation
        perturb = rng.uniform(-step_size, step_size, pts.shape)
        new_pts = pts + perturb
        
        # Project to unit disk
        radii = np.hypot(new_pts[:, 0], new_pts[:, 1])
        mask = radii > 1.0
        if np.any(mask):
            new_pts[mask] = new_pts[mask] / radii[mask, np.newaxis]
        
        new_score = _score(new_pts)
        
        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts.copy()
            pts = new_pts.copy()
            no_improve = 0
            # Slightly increase step size on improvement
            step_size = min(0.1, step_size * 1.1)
        else:
            no_improve += 1
            # Decrease step size
            step_size *= 0.95
    
    return best_pts


def _force_directed(pts0, time_limit, iters=2000):
    """
    Force-directed layout for rapid improvement.
    """
    pts = pts0.copy()
    best_score = _score(pts)
    best_pts = pts.copy()
    
    lr = 0.005
    decay = 0.9995
    
    for it in range(iters):
        if time.time() > time_limit:
            break
        
        # Compute forces (vectorized)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, np.inf)
        
        r_est = 0.5 * np.min(dists[dists < np.inf])
        if r_est < 1e-6:
            r_est = 0.05
        
        forces = np.zeros_like(pts)
        
        # Repulsion: F = 1/d^4 for close pairs
        # Vectorized approach
        close_mask = (dists < 4.0 * r_est) & (dists > 1e-10)
        if np.any(close_mask):
            # For each pair, add force
            for i in range(N):
                for j in range(i + 1, N):
                    if close_mask[i, j]:
                        d = dists[i, j]
                        f_mag = 1.0 / (d ** 4)
                        direction = (pts[i] - pts[j]) / d
                        forces[i] += f_mag * direction
                        forces[j] -= f_mag * direction
        
        # Boundary force
        dists_origin = np.hypot(pts[:, 0], pts[:, 1])
        margin = 1.0 - dists_origin
        for i in range(N):
            if margin[i] < r_est and dists_origin[i] > 1e-10:
                forces[i] -= 2.0 * (r_est - margin[i]) * pts[i] / dists_origin[i]
        
        # Weak centering
        forces -= 0.001 * pts
        
        # Update
        pts += lr * forces
        
        # Project to unit disk
        radii = np.hypot(pts[:, 0], pts[:, 1])
        mask = radii > 1.0
        if np.any(mask):
            pts[mask] = pts[mask] / radii[mask, np.newaxis]
        
        pts = np.nan_to_num(pts, nan=0.0, posinf=0.9, neginf=-0.9)
        
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
        
        lr *= decay
    
    return best_pts


def construct():
    start_time = time.time()
    
    # 1. Generate multiple initializations
    initializations = [
        _ring_init(),
        _jittered_ring_init(),
        _hex_lattice_init(),
        _random_init(42),
        _random_init(123),
        _random_init(7),
    ]
    
    # Score all initializations
    init_scores = [_score(p) for p in initializations]
    best_init_idx = np.argmax(init_scores)
    best_pts = initializations[best_init_idx].copy()
    best_score = init_scores[best_init_idx]
    
    # 2. Force-directed from best init (quick improvement)
    fd_deadline = _DEADLINE - 90.0
    if time.time() < fd_deadline:
        try:
            fd_pts = _force_directed(best_pts, fd_deadline, iters=1500)
            fd_score = _score(fd_pts)
            if fd_score > best_score:
                best_pts = fd_pts
                best_score = fd_score
        except Exception:
            pass
    
    # 3. Try force-directed from other good inits
    if time.time() < _DEADLINE - 80.0:
        for i, init in enumerate(initializations):
            if i == best_init_idx:
                continue
            if init_scores[i] < best_score * 0.92:
                continue
            if time.time() > _DEADLINE - 70.0:
                break
            try:
                fd_pts = _force_directed(init, _DEADLINE - 70.0, iters=800)
                fd_score = _score(fd_pts)
                if fd_score > best_score:
                    best_pts = fd_pts
                    best_score = fd_score
            except Exception:
                pass
    
    # 4. Progressive soft-min polish from best
    polish_deadline = _DEADLINE - 25.0
    if time.time() < polish_deadline:
        try:
            polished_pts = _optimize_from_start(best_pts, polish_deadline)
            polished_score = _score(polished_pts)
            if polished_score > best_score:
                best_pts = polished_pts
                best_score = polished_score
        except Exception:
            pass
    
    # 5. Local search to escape local optima
    ls_deadline = _DEADLINE - 15.0
    if time.time() < ls_deadline:
        try:
            ls_pts = _local_search(best_pts, ls_deadline)
            ls_score = _score(ls_pts)
            if ls_score > best_score:
                best_pts = ls_pts
                best_score = ls_score
        except Exception:
            pass
    
    # 6. Final high-precision polish
    final_deadline = _DEADLINE - 5.0
    if time.time() < final_deadline:
        try:
            fine_betas = [0.001, 0.0005, 0.0002, 0.0001]
            final_pts = _optimize_from_start(best_pts, final_deadline, betas=fine_betas)
            final_score = _score(final_pts)
            if final_score > best_score:
                best_pts = final_pts
                best_score = final_score
        except Exception:
            pass
    
    # Final safety check
    final_score = _score(best_pts)
    if final_score <= 0:
        return initializations[best_init_idx]
    
    # Ensure all points are strictly inside
    radii = np.hypot(best_pts[:, 0], best_pts[:, 1])
    mask = radii > 1.0
    if np.any(mask):
        best_pts[mask] = best_pts[mask] * 0.999 / radii[mask, np.newaxis]
    
    return best_pts

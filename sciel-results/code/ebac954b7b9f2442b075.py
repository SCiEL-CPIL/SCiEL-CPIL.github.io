import time
import numpy as np
from scipy.optimize import minimize

N = 50
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin before 150s kill

def _score(pts):
    """Compute the valid packing radius (or 0 if invalid)."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    
    # Check if inside unit circle
    dists_origin = np.linalg.norm(pts, axis=1)
    if np.any(dists_origin > 1.0 + 1e-9):
        return 0.0
    
    # Calculate min distance to boundary
    r_cont = 1.0 - np.max(dists_origin)
    if r_cont < 0:
        return 0.0
        
    # Calculate min pairwise distance
    if N < 2:
        return r_cont
        
    # Efficient pairwise distance
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_mat = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist_mat, np.inf)
    r_pair = 0.5 * np.min(dist_mat)
    
    return min(r_cont, r_pair)

def _soft_min_obj_and_grad(flat, beta):
    """
    Compute the soft-min objective and gradient for L-BFGS-B.
    Objective: minimize f(x) = (1/beta) * ln(sum(exp(-beta * x_k)))
    where x_k are the critical distances (pairwise and boundary).
    Minimizing this is equivalent to maximizing the soft-min of x_k.
    """
    pts = flat.reshape(N, 2)
    
    # Calculate boundary distances: 1 - |c_i|
    dists_origin = np.linalg.norm(pts, axis=1)
    boundary_dists = 1.0 - dists_origin
    
    # Calculate pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_mat = np.linalg.norm(diff, axis=-1)
    
    # We only need upper triangle for unique pairs
    iu = np.triu_indices(N, k=1)
    pair_dists = dist_mat[iu]
    
    # Combine all critical distances
    all_dists = np.concatenate([boundary_dists, pair_dists / 2.0])
    
    # Soft min calculation
    y = -beta * all_dists
    max_y = np.max(y)
    y_shifted = y - max_y
    sum_exp = np.sum(np.exp(y_shifted))
    log_sum_exp = max_y + np.log(sum_exp)
    obj_val = log_sum_exp / beta
    
    # Gradient
    weights = np.exp(y_shifted) / sum_exp
    
    grad_pts = np.zeros_like(pts)
    
    # 1. Boundary terms
    w_boundary = weights[0:N]
    safe_dists = np.where(dists_origin < 1e-8, 1.0, dists_origin)
    boundary_grad = (w_boundary / safe_dists)[:, np.newaxis] * pts
    grad_pts += boundary_grad
    
    # 2. Pairwise terms
    w_pairs = weights[N:]
    i_idx, m_idx = iu
    d_pairs = dist_mat[i_idx, m_idx]
    safe_d_pairs = np.where(d_pairs < 1e-8, 1.0, d_pairs)
    vecs = pts[i_idx] - pts[m_idx]
    unit_vecs = vecs / safe_d_pairs[:, np.newaxis]
    
    grad_contrib_i = -0.5 * w_pairs[:, np.newaxis] * unit_vecs
    grad_contrib_m = 0.5 * w_pairs[:, np.newaxis] * unit_vecs
    
    np.add.at(grad_pts, i_idx, grad_contrib_i)
    np.add.at(grad_pts, m_idx, grad_contrib_m)
    
    grad_flat = grad_pts.flatten()
    
    return obj_val, grad_flat

def _force_directed_relax(pts, n_steps=200, target_r=0.12):
    """
    Force-directed relaxation to improve packing.
    Pushes overlapping circles apart and pulls them toward the center.
    """
    pts = pts.copy()
    target_d = 2.0 * target_r
    
    for step in range(n_steps):
        # Compute forces
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion for overlapping circles
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_mat = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dist_mat, np.inf)
        
        # Only consider pairs that are too close
        for i in range(N):
            for j in range(i + 1, N):
                d = dist_mat[i, j]
                if d < target_d and d > 1e-10:
                    # Repulsive force proportional to overlap
                    overlap = target_d - d
                    direction = (pts[i] - pts[j]) / d
                    force_mag = overlap * 0.5
                    forces[i] += force_mag * direction
                    forces[j] -= force_mag * direction
        
        # Boundary constraint: push circles inward if they're outside
        dists_origin = np.linalg.norm(pts, axis=1)
        for i in range(N):
            if dists_origin[i] > 1.0 - target_r:
                # Push inward
                direction = -pts[i] / dists_origin[i]
                excess = dists_origin[i] - (1.0 - target_r)
                forces[i] += excess * direction
        
        # Update positions with damping
        lr = 0.01 * (1.0 - step / n_steps) + 0.001
        pts += lr * forces
        
        # Project to unit disk
        dists = np.linalg.norm(pts, axis=1)
        mask = dists > 1.0
        if np.any(mask):
            pts[mask] = pts[mask] / dists[mask, np.newaxis]
    
    return pts

def _hex_lattice_init():
    """Generate a hexagonal lattice packing as initial guess."""
    best_pts = None
    best_score = -1.0
    
    for spacing in [0.24, 0.25, 0.26, 0.27, 0.28]:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        
        mask = radius <= 1.0
        points = points[mask]
        radius = radius[mask]
        
        if len(points) >= N:
            idx = np.argsort(radius, kind="stable")[:N]
            pts = points[idx]
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts
    
    if best_pts is not None:
        return best_pts
        
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 0.9, N))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

def _ring_init():
    """Generate a ring-based initialization."""
    # Try a few ring configurations
    best_pts = None
    best_score = -1.0
    
    # Configuration 1: center + rings
    for n_rings in [3, 4, 5]:
        pts_list = [[0.0, 0.0]]
        remaining = N - 1
        per_ring = [remaining // n_rings + (1 if i < remaining % n_rings else 0) for i in range(n_rings)]
        
        for ring_idx, count in enumerate(per_ring):
            r = (ring_idx + 1) * 0.3
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
            for a in angles:
                pts_list.append([r * np.cos(a), r * np.sin(a)])
        
        pts = np.array(pts_list[:N])
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts
    
    if best_pts is not None:
        return best_pts
    return _hex_lattice_init()

def _random_init():
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 0.95, N))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

def _perturb(pts, sigma=0.01):
    """Add small random perturbation to escape local minima."""
    noise = np.random.normal(0, sigma, pts.shape)
    return pts + noise

def _project_to_disk(pts):
    """Project points to unit disk."""
    dists = np.linalg.norm(pts, axis=1)
    mask = dists > 1.0
    if np.any(mask):
        pts[mask] = pts[mask] / dists[mask, np.newaxis]
    return pts

def construct():
    # Initializations
    inits = []
    inits.append(_hex_lattice_init())
    inits.append(_ring_init())
    for _ in range(4):
        inits.append(_random_init())
        
    best_pts = inits[0]
    best_score = _score(best_pts)
    
    # Use force-directed relaxation to improve initial configurations
    # Estimate target radius from best initial
    target_r = max(best_score, 0.12)
    
    improved_inits = []
    for init_pts in inits:
        if time.time() > _DEADLINE - 60.0:
            break
        fd_pts = _force_directed_relax(init_pts, n_steps=150, target_r=target_r)
        fd_pts = _project_to_disk(fd_pts)
        fd_score = _score(fd_pts)
        if fd_score > best_score:
            best_score = fd_score
            best_pts = fd_pts.copy()
        improved_inits.append(fd_pts)
    
    # Combine original and force-directed improved inits
    all_inits = inits + improved_inits
    
    betas = [10, 50, 200, 1000]
    
    for init_pts in all_inits:
        current_pts = init_pts.copy()
        
        for beta in betas:
            # Check time
            if time.time() > _DEADLINE - 15.0:
                break
                
            try:
                res = minimize(
                    _soft_min_obj_and_grad,
                    current_pts.flatten(),
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=[(-1.0, 1.0)] * (2 * N),
                    options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
                )
                
                new_pts = res.x.reshape(N, 2)
                new_pts = _project_to_disk(new_pts)
                
                new_score = _score(new_pts)
                
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
                    current_pts = new_pts.copy()
                    
            except Exception:
                pass
        
        # Check time
        if time.time() > _DEADLINE - 15.0:
            break
            
    # Multiple final polish passes with very high beta on the best found
    # Use more aggressive betas and tighter tolerances
    polish_betas = [2000, 5000, 10000, 20000, 50000, 100000]
    for beta in polish_betas:
        if time.time() > _DEADLINE - 8.0:
            break
        try:
            res = minimize(
                _soft_min_obj_and_grad,
                best_pts.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-1.0, 1.0)] * (2 * N),
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-14, 'disp': False}
            )
            new_pts = res.x.reshape(N, 2)
            new_pts = _project_to_disk(new_pts)
            
            new_score = _score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts
        except Exception:
            pass
            
    # Perturbation-based local search near the best solution
    if time.time() < _DEADLINE - 12.0:
        for sigma in [0.02, 0.01, 0.005, 0.002]:
            if time.time() > _DEADLINE - 8.0:
                break
            for attempt in range(4):
                if time.time() > _DEADLINE - 8.0:
                    break
                try:
                    perturbed = _perturb(best_pts, sigma)
                    perturbed = _project_to_disk(perturbed)
                    
                    # Quick polish with moderate beta
                    res = minimize(
                        _soft_min_obj_and_grad,
                        perturbed.flatten(),
                        args=(1000,),
                        method='L-BFGS-B',
                        jac=True,
                        bounds=[(-1.0, 1.0)] * (2 * N),
                        options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
                    )
                    new_pts = res.x.reshape(N, 2)
                    new_pts = _project_to_disk(new_pts)
                    
                    new_score = _score(new_pts)
                    if new_score > best_score:
                        best_score = new_score
                        best_pts = new_pts
                except Exception:
                    pass
    
    return best_pts

# EVOLVE-BLOCK-END

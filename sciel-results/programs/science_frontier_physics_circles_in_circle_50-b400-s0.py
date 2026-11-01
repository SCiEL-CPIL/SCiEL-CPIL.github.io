import time
import numpy as np
from scipy.optimize import minimize

N = 50
_DEADLINE = time.time() + 140.0  # 10s safety margin before 150s kill

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
    # We need the gradient with respect to pts
    # Let's compute the full distance matrix for values, and gradients carefully.
    
    # Values for soft-min
    # To avoid overflow in exp, we can shift by max, but here beta is moderate.
    # However, if beta is large, exp(-beta * x) can underflow/overflow.
    # Standard trick: log-sum-exp.
    # LSE(y) = max(y) + log(sum(exp(y - max(y))))
    # Here y_k = -beta * x_k.
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_mat = np.linalg.norm(diff, axis=-1)
    
    # We only need upper triangle for unique pairs
    iu = np.triu_indices(N, k=1)
    pair_dists = dist_mat[iu]
    
    # Combine all critical distances
    # x_vector contains boundary_dists and pair_dists
    # Note: The score is min(boundary_dists, pair_dists/2).
    # So the "critical distances" for the min are:
    # x_k = boundary_dists[i]
    # x_k = pair_dists[j] / 2.0
    # We want to maximize min(x_k).
    
    all_dists = np.concatenate([boundary_dists, pair_dists / 2.0])
    
    # Soft min calculation
    # y = -beta * all_dists
    y = -beta * all_dists
    max_y = np.max(y)
    # Prevent overflow
    y_shifted = y - max_y
    sum_exp = np.sum(np.exp(y_shifted))
    log_sum_exp = max_y + np.log(sum_exp)
    obj_val = -log_sum_exp / beta  # This is - (1/beta) LSE(-beta * x) = soft_min(x) approx?
    # Wait. 
    # min(x) = lim_{beta->inf} -1/beta ln(sum exp(-beta x))
    # We want to MAXIMIZE min(x). So we MINIMIZE -min(x).
    # -min(x) approx 1/beta ln(sum exp(-beta x)).
    # So objective to minimize is f(x) = 1/beta ln(sum exp(-beta x_k)).
    # My calculation above: log_sum_exp = LSE(-beta * x).
    # f(x) = log_sum_exp / beta.
    
    obj_val = log_sum_exp / beta
    
    # Gradient
    # d/dx_j [ 1/beta ln(sum_k exp(-beta x_k)) ]
    # = 1/beta * 1/sum(exp(-beta x_k)) * sum_k d/dx_j exp(-beta x_k)
    # = 1/beta * 1/sum(exp(-beta x_k)) * sum_k exp(-beta x_k) * (-beta) * d/dx_j x_k
    # = - sum_k [ exp(-beta x_k) / sum(exp(-beta x_k)) ] * d/dx_j x_k
    # Let w_k = exp(-beta x_k) / sum(exp(-beta x_k)).
    # Gradient w.r.t x_k is -w_k.
    
    weights = np.exp(y_shifted) / sum_exp  # These are w_k
    
    # Now we need d/d(pts) x_k for each k.
    # x_k are either boundary_dist[i] or pair_dist[j]/2.
    
    grad_pts = np.zeros_like(pts)
    
    # 1. Boundary terms
    # x_b_i = 1 - |p_i|
    # d/dp_i x_b_i = - p_i / |p_i|
    # Contribution: - w_b_i * (- p_i / |p_i|) = w_b_i * p_i / |p_i|
    # Note: weights for boundary terms are weights[0:N]
    w_boundary = weights[0:N]
    
    # Avoid division by zero for points at origin (probable measure 0, but safe)
    safe_dists = np.where(dists_origin < 1e-8, 1.0, dists_origin)
    boundary_grad = (w_boundary / safe_dists)[:, np.newaxis] * pts
    grad_pts += boundary_grad
    
    # 2. Pairwise terms
    # x_p_j = 0.5 * |p_i - p_m| for pair (i, m)
    # d/dp_i x_p_j = 0.5 * (p_i - p_m) / |p_i - p_m|
    # d/dp_m x_p_j = 0.5 * (p_m - p_i) / |p_i - p_m| = - d/dp_i x_p_j
    # Contribution to grad_pts[i] from pair j: - w_p_j * (0.5 * (p_i - p_m) / d_ij)
    # Contribution to grad_pts[m] from pair j: - w_p_j * (0.5 * (p_m - p_i) / d_ij)
    
    w_pairs = weights[N:]
    
    # We have iu indices
    i_idx, m_idx = iu
    # dist_mat[i_idx, m_idx] are the distances
    d_pairs = dist_mat[i_idx, m_idx]
    
    # Safe division
    safe_d_pairs = np.where(d_pairs < 1e-8, 1.0, d_pairs)
    
    # Vector (p_i - p_m)
    vecs = pts[i_idx] - pts[m_idx]
    
    # Unit vectors
    unit_vecs = vecs / safe_d_pairs[:, np.newaxis]
    
    # Gradient for point i: - w_j * 0.5 * unit_vec
    # Gradient for point m: - w_j * 0.5 * (-unit_vec) = w_j * 0.5 * unit_vec
    
    # Accumulate gradients
    # We need to add to grad_pts[i] and grad_pts[m]
    # Use np.add.at for repeated indices
    grad_contrib_i = -0.5 * w_pairs[:, np.newaxis] * unit_vecs
    grad_contrib_m = 0.5 * w_pairs[:, np.newaxis] * unit_vecs
    
    np.add.at(grad_pts, i_idx, grad_contrib_i)
    np.add.at(grad_pts, m_idx, grad_contrib_m)
    
    # Flatten gradient
    grad_flat = grad_pts.flatten()
    
    return obj_val, grad_flat

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

def construct():
    start_time = time.time()
    
    # Initializations
    inits = []
    inits.append(_hex_lattice_init())
    inits.append(_ring_init())
    for _ in range(4):
        inits.append(_random_init())
        
    best_pts = inits[0]
    best_score = _score(best_pts)
    
    betas = [10, 50, 200, 1000]
    
    for init_pts in inits:
        current_pts = init_pts.copy()
        
        for beta in betas:
            # Check time
            if time.time() > _DEADLINE - 10.0:
                break
                
            try:
                res = minimize(
                    _soft_min_obj_and_grad,
                    current_pts.flatten(),
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=[(-1.0, 1.0)] * (2 * N),
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
                )
                
                new_pts = res.x.reshape(N, 2)
                
                # Project back to unit disk to ensure validity if it drifted out
                dists = np.linalg.norm(new_pts, axis=1)
                mask = dists > 1.0
                if np.any(mask):
                    new_pts[mask] = new_pts[mask] / dists[mask, np.newaxis]
                    
                new_score = _score(new_pts)
                
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
                    current_pts = new_pts.copy()
                    
            except Exception:
                pass
        
        # Check time
        if time.time() > _DEADLINE - 10.0:
            break
            
    # Multiple final polish passes with very high beta on the best found
    polish_betas = [2000, 5000, 10000, 20000, 50000]
    for beta in polish_betas:
        if time.time() > _DEADLINE - 5.0:
            break
        try:
            res = minimize(
                _soft_min_obj_and_grad,
                best_pts.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-1.0, 1.0)] * (2 * N),
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'disp': False}
            )
            new_pts = res.x.reshape(N, 2)
            dists = np.linalg.norm(new_pts, axis=1)
            mask = dists > 1.0
            if np.any(mask):
                new_pts[mask] = new_pts[mask] / dists[mask, np.newaxis]
            
            new_score = _score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts
        except Exception:
            pass
            
    # Perturbation-based local search near the best solution
    if time.time() < _DEADLINE - 10.0:
        for sigma in [0.02, 0.01, 0.005]:
            if time.time() > _DEADLINE - 5.0:
                break
            for attempt in range(3):
                if time.time() > _DEADLINE - 5.0:
                    break
                try:
                    perturbed = _perturb(best_pts, sigma)
                    # Project to unit disk
                    dists = np.linalg.norm(perturbed, axis=1)
                    mask = dists > 1.0
                    if np.any(mask):
                        perturbed[mask] = perturbed[mask] / dists[mask, np.newaxis]
                    
                    # Quick polish with moderate beta
                    res = minimize(
                        _soft_min_obj_and_grad,
                        perturbed.flatten(),
                        args=(500,),
                        method='L-BFGS-B',
                        jac=True,
                        bounds=[(-1.0, 1.0)] * (2 * N),
                        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
                    )
                    new_pts = res.x.reshape(N, 2)
                    dists = np.linalg.norm(new_pts, axis=1)
                    mask = dists > 1.0
                    if np.any(mask):
                        new_pts[mask] = new_pts[mask] / dists[mask, np.newaxis]
                    
                    new_score = _score(new_pts)
                    if new_score > best_score:
                        best_score = new_score
                        best_pts = new_pts
                except Exception:
                    pass
    
    return best_pts

# EVOLVE-BLOCK-END

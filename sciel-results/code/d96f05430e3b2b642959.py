import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N = 300
# EVOLVE-BLOCK-START

def _true_score(points):
    """Compute the exact feasible radius."""
    if points is None or len(points) < 2:
        return 0.0
    norms = np.linalg.norm(points, axis=1)
    min_boundary = float(1.0 - np.max(norms))
    if min_boundary <= 0:
        return 0.0
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    min_pair = float(np.min(dists[:, 1])) / 2.0
    return min(min_boundary, min_pair)

def _soft_min_objective(x, beta):
    """
    Objective function for L-BFGS-B: minimize -soft_min(radius).
    x: flattened array of shape (2*N,)
    beta: sharpness parameter for soft-min
    """
    pts = x.reshape(N, 2)
    
    # Boundary terms: 1 - |c_i|
    norms = np.linalg.norm(pts, axis=1)
    boundary_dists = 1.0 - norms
    
    # Pairwise terms: d_ij / 2
    # Compute pairwise distances
    # To save memory/time, we can compute upper triangle
    # But for N=300, full matrix is 300x300, which is small.
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    pair_dists = dists[mask]
    pair_terms = pair_dists / 2.0
    
    # Combine all terms
    all_terms = np.concatenate([pair_terms, boundary_dists])
    
    # Soft min: -1/beta * log(sum(exp(-beta * terms)))
    # For numerical stability, subtract max term
    max_term = np.max(all_terms)
    log_sum = np.log(np.sum(np.exp(-beta * (all_terms - max_term))))
    soft_min_val = max_term - (1.0 / beta) * log_sum
    
    # We want to maximize soft_min_val, so minimize -soft_min_val
    return -soft_min_val

def _soft_min_gradient(x, beta):
    """
    Gradient of -soft_min_objective.
    """
    pts = x.reshape(N, 2)
    
    # Boundary terms
    norms = np.linalg.norm(pts, axis=1)
    boundary_dists = 1.0 - norms
    
    # Pairwise terms
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    mask = ~np.eye(N, dtype=bool)
    pair_dists = dists[mask]
    pair_terms = pair_dists / 2.0
    
    all_terms = np.concatenate([pair_terms, boundary_dists])
    
    max_term = np.max(all_terms)
    weights = np.exp(-beta * (all_terms - max_term))
    Z = np.sum(weights)
    
    # Gradient of soft_min_val w.r.t. terms
    # d(soft_min)/d(term_k) = (weights[k] / Z) * (-1) ? 
    # Let S = -1/beta * log(sum(exp(-beta * t)))
    # dS/dt_k = -1/beta * (1/sum) * sum' * (-beta * exp(-beta * t_k))
    # sum' = -beta * exp(-beta * t_k)
    # dS/dt_k = -1/beta * (1/Z) * (-beta * w_k) = w_k / Z
    # So gradient of S w.r.t. x is sum_k (w_k/Z) * grad(t_k)
    # We are minimizing -S, so gradient is - sum_k (w_k/Z) * grad(t_k)
    
    grad_x = np.zeros_like(pts)
    
    # Gradient of boundary terms: t_i = 1 - |c_i|
    # dt_i/dc_i = - c_i / |c_i|
    # w_b = weights[N:]
    w_b = weights[N:]
    # grad_boundary = - sum_i (w_b[i]/Z) * (-c_i/|c_i|) = sum_i (w_b[i]/Z) * (c_i/|c_i|)
    # Note: We are computing grad of -S. 
    # S = soft_min. grad S = sum w/Z * grad t.
    # grad (-S) = - sum w/Z * grad t.
    # grad t_i = - c_i / |c_i|
    # So grad (-S) boundary part = - sum (w_b[i]/Z) * (-c_i/|c_i|) = sum (w_b[i]/Z) * (c_i/|c_i|)
    
    # Handle division by zero for center
    safe_norms = np.where(norms > 1e-10, norms, 1.0)
    grad_boundary_contrib = (w_b / Z)[:, np.newaxis] * (pts / safe_norms[:, np.newaxis])
    grad_x += grad_boundary_contrib
    
    # Gradient of pair terms: t_{ij} = d_{ij} / 2
    # dt_{ij}/d c_i = (c_i - c_j) / (2 * d_{ij})
    # dt_{ij}/d c_j = (c_j - c_i) / (2 * d_{ij})
    # w_p = weights[:N*(N-1)/2]
    w_p = weights[:N*(N-1)/2]
    
    # We need to map the flat array w_p back to indices.
    # This is expensive to do explicitly in a loop.
    # Alternative: Compute gradient via vectorized operations on the full matrix.
    
    # Let's compute the full gradient matrix G where G[i,j] is contribution to c_i from pair (i,j)
    # Actually, let's just compute the force on each point.
    # F_i = sum_{j != i} (w_{ij}/Z) * (c_i - c_j) / (2 * d_{ij})  <-- Wait, sign?
    # S = soft_min. grad S_i = sum_j (w_{ij}/Z) * grad t_{ij}
    # grad t_{ij} w.r.t c_i = (c_i - c_j) / (2 d_{ij})
    # We want grad (-S) = - grad S.
    # So contribution to grad_x[i] from pair (i,j) is - (w_{ij}/Z) * (c_i - c_j) / (2 d_{ij})
    
    # Let's construct the full weight matrix W_mat of shape (N, N)
    # W_mat[i,j] corresponds to weight of pair (i,j) if i<j, and symmetric.
    # The flat array w_p is upper triangle.
    W_mat = np.zeros((N, N))
    iu = np.triu_indices(N, k=1)
    W_mat[iu] = w_p
    W_mat = W_mat + W_mat.T
    
    # Now compute forces
    # diff = pts[:, None, :] - pts[None, :, :]  # (N, N, 2)
    # dists_mat = np.linalg.norm(diff, axis=2) # (N, N)
    # We already have diff and dists from objective, but they are local. Let's recompute or pass them?
    # Recalculation is cheap enough.
    
    # Avoid division by zero
    safe_dists = np.where(dists > 1e-10, dists, 1.0)
    
    # term = (c_i - c_j) / (2 * d_{ij})
    # grad_vec = W_mat / (2 * safe_dists)  # (N, N)
    # grad_x[i] += - sum_j grad_vec[i,j] * diff[i,j]
    
    grad_vec = W_mat / (2.0 * safe_dists)
    # grad_vec[i,j] is scalar weight for the vector (c_i - c_j)
    
    # grad_x[i] = - sum_j grad_vec[i,j] * (c_i - c_j)
    # = - sum_j grad_vec[i,j] * diff[i,j]
    
    # diff is (N, N, 2). grad_vec is (N, N).
    # We need to multiply each row i of diff by grad_vec[i, :]
    # np.einsum('ij,ijc->ic', grad_vec, diff) gives sum over j.
    grad_pair_contrib = -np.einsum('ij,ijc->ic', grad_vec, diff)
    
    grad_x += grad_pair_contrib
    
    return grad_x.flatten()

def _refine_lbfgs(points, deadline, beta=100.0, max_iter=200):
    """
    Refine points using L-BFGS-B on the soft-min objective.
    """
    x0 = points.flatten()
    
    # Check time
    if time.time() > deadline:
        return points, _true_score(points)
    
    start_time = time.time()
    
    # L-BFGS-B options
    options = {
        'maxiter': max_iter,
        'ftol': 1e-10,
        'gtol': 1e-8,
        'disp': False,
    }
    
    try:
        res = minimize(
            _soft_min_objective,
            x0,
            args=(beta,),
            jac=_soft_min_gradient,
            method='L-BFGS-B',
            options=options
        )
    except Exception:
        return points, _true_score(points)
    
    if time.time() > deadline:
        return points, _true_score(points)
    
    if res.success or res.x is not None:
        new_points = res.x.reshape(N, 2)
        # Ensure no NaNs
        if not np.all(np.isfinite(new_points)):
            return points, _true_score(points)
        
        new_score = _true_score(new_points)
        old_score = _true_score(points)
        
        if new_score > old_score:
            return new_points, new_score
    
    return points, _true_score(points)

def _build_hex_lattice(spacing):
    """Hex lattice points inside unit disk, pick N closest to center."""
    k = int(1.0 / spacing) + 4
    pts = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0 - spacing * 0.3:
                pts.append((x, y))
    if len(pts) < N:
        return None
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists, kind="stable")[:N]
    return pts[idx]

def _build_rings(spacing):
    """Concentric rings."""
    pts = []
    n_rings = int(1.0 / spacing) + 1
    for k in range(1, n_rings + 1):
        r = k * spacing
        if r > 1.0 - spacing * 0.5:
            break
        c = int(2 * np.pi * r / spacing)
        if c < 1:
            continue
        angles = 2 * np.pi * np.arange(c) / c
        pts.append(np.column_stack([r * np.cos(angles), r * np.sin(angles)]))
    
    if not pts:
        return None
    
    pts = np.vstack(pts)
    
    # If we have too many, trim from outside
    if len(pts) > N:
        dists = np.linalg.norm(pts, axis=1)
        idx = np.argsort(dists, kind="stable")[:N]
        pts = pts[idx]
    elif len(pts) < N:
        # Add center or random points?
        # Just pad with random points for now, or return None if too few
        # Better to return None if not enough
        return None
        
    return pts

def _simulated_annealing(points, deadline, max_iters=5000):
    """
    Simulated Annealing to escape local minima.
    """
    pts = points.copy()
    best_pts = pts.copy()
    best_r = _true_score(pts)
    
    T = 0.01
    cooling = 0.999
    rng = np.random.default_rng(42)
    
    for i in range(max_iters):
        if time.time() > deadline:
            break
            
        idx = rng.integers(0, N)
        orig = pts[idx].copy()
        
        # Random perturbation
        step = T * rng.uniform(0.5, 1.5)
        ang = rng.uniform(0, 2 * np.pi)
        new_pos = orig + step * np.array([np.cos(ang), np.sin(ang)])
        
        # Simple boundary check
        if np.linalg.norm(new_pos) > 1.0:
            continue
            
        pts[idx] = new_pos
        
        # Fast local check: KDTree
        tree = cKDTree(pts)
        dists, _ = tree.query(pts, k=2)
        min_pair = np.min(dists[:, 1]) / 2.0
        norms = np.linalg.norm(pts, axis=1)
        min_bound = 1.0 - np.max(norms)
        new_r = min(min_pair, min_bound)
        
        delta_r = new_r - best_r
        
        if delta_r > 0:
            best_r = new_r
            best_pts = pts.copy()
            # Don't cool if we improved? Or cool slightly?
            # Standard SA: accept if improved.
        else:
            # Accept with probability exp(delta_r / T)
            if rng.random() < np.exp(delta_r / T):
                pass # Accept
            else:
                pts[idx] = orig # Reject
                
        T *= cooling
        
    return best_pts, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    candidates = []
    
    # 1. Generate initial candidates
    # Hex lattices
    for spacing in [0.05, 0.052, 0.054, 0.056, 0.058, 0.06]:
        pts = _build_hex_lattice(spacing)
        if pts is not None:
            r = _true_score(pts)
            candidates.append((pts, r))
            
    # Rings
    for spacing in [0.05, 0.052, 0.054, 0.056, 0.058]:
        pts = _build_rings(spacing)
        if pts is not None:
            r = _true_score(pts)
            candidates.append((pts, r))
            
    # Random
    for _ in range(5):
        angles = 2 * np.pi * np.random.rand(N)
        radii = np.sqrt(np.random.rand(N)) * 0.9
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        r = _true_score(pts)
        candidates.append((pts, r))
        
    if not candidates:
        angles = 2 * np.pi * np.arange(N) / N
        return np.column_stack([0.5 * np.cos(angles), 0.5 * np.sin(angles)])
        
    # Sort candidates by score
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    best_points, best_r = candidates[0]
    
    # 2. Refine top candidates
    # Use SA for a few of them, then L-BFGS-B
    
    n_sa = min(3, len(candidates))
    for i in range(n_sa):
        if time.time() > deadline - 20.0:
            break
        cand_pts, _ = candidates[i]
        sa_pts, sa_r = _simulated_annealing(cand_pts, deadline - 10.0, max_iters=2000)
        if sa_r > best_r:
            best_points, best_r = sa_pts, sa_r
            
    # 3. Final Polish with L-BFGS-B
    # Use the best point found so far
    if time.time() < deadline - 5.0:
        # Try a few different betas for soft-min
        for beta in [50, 100, 200]:
            if time.time() > deadline - 5.0:
                break
            polished_pts, polished_r = _refine_lbfgs(best_points, deadline, beta=beta, max_iter=100)
            if polished_r > best_r:
                best_points, best_r = polished_pts, polished_r
                
    return best_points

# EVOLVE-BLOCK-END

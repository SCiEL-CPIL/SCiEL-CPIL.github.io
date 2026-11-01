"""
Circle packing: 50 equal circles in unit disk.
Strategy: Smooth surrogate optimization via L-BFGS with multi-start from geometric templates.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
_DEADLINE = time.time() + 145.0  # 5s safety margin before 150s kill

def _compute_score(pts):
    """Return the true packing radius."""
    n = len(pts)
    if n < 2:
        return 0.0
    # boundary distances
    boundary = 1.0 - np.linalg.norm(pts, axis=1)
    if np.any(boundary < -1e-12):
        return 0.0  # invalid: outside container
    # pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return min(min_pair, np.min(boundary))

def _softmin(vals, beta):
    """Soft-min approximation of min(vals)."""
    neg = -vals
    m = np.max(neg)
    if m > 1e12:
        return m
    exp_vals = np.exp(neg - m)
    log_sum = np.log(np.sum(exp_vals)) + m
    return -log_sum / beta

def _objective_and_grad(x, n, beta):
    """
    Objective: -softmin(constraints)
    Constraints: pairwise distances and boundary distances.
    We want to maximize the min of these, so we minimize the negative soft-min.
    """
    pts = x.reshape(n, 2)
    
    # Boundary distances: 1 - ||p_i||
    norms = np.linalg.norm(pts, axis=1)
    boundary_dists = 1.0 - norms
    
    # Pairwise distances: ||p_i - p_j||
    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Collect all constraints
    # We take the min of all pairwise distances and boundary distances.
    # To be efficient, we can just compute softmin over all of them.
    # But for gradient, we need to be careful.
    # Let's construct the vector of all constraints.
    # Number of pairwise constraints: n*(n-1)/2
    # Number of boundary constraints: n
    # Total: ~1275 for n=50.
    
    # Instead of building a huge vector, we can compute softmin on the fly 
    # or use a trick. But for L-BFGS, we need a scalar objective and gradient.
    # The soft-min is differentiable.
    
    # Let's build the vector of distances for softmin.
    # This is O(N^2) memory, which is fine for N=50 (2500 floats).
    pair_idx = np.triu_indices(n, k=1)
    pair_dists = dists[pair_idx]
    
    all_dists = np.concatenate([pair_dists, boundary_dists])
    
    # Soft-min
    neg_dists = -all_dists
    m = np.max(neg_dists)
    exp_vals = np.exp(neg_dists - m)
    sum_exp = np.sum(exp_vals)
    if sum_exp == 0:
        sum_exp = 1e-300
    weights = exp_vals / sum_exp
    
    soft_min_val = -np.sum(weights * all_dists)
    
    # Gradient
    # d(softmin)/dx = -sum(weights * d(constraint)/dx)
    # d(boundary_dist_i)/dx_i = -x_i / ||x_i||
    # d(pair_dist_ij)/dx_i = (x_i - x_j) / ||x_i - x_j||
    # d(pair_dist_ij)/dx_j = (x_j - x_i) / ||x_i - x_j||
    
    grad = np.zeros_like(pts)
    
    # Boundary gradient contribution
    # weights for boundary are the last n weights
    w_boundary = weights[n*(n-1)//2:]
    # d(boundary_i)/dx_i = -pts[i] / norms[i]
    # Contribution to grad[i] = -w_boundary[i] * (-pts[i] / norms[i]) = w_boundary[i] * pts[i] / norms[i]
    # Wait, objective is -softmin. So grad_obj = - grad_softmin.
    # grad_softmin = sum(w * d(constraint)/dx)
    # So grad_obj = - sum(w * d(constraint)/dx)
    
    # Let's compute grad_softmin first.
    grad_softmin = np.zeros_like(pts)
    
    # Boundary part
    # d(boundary_i)/dx_i = -pts[i] / norms[i]
    # grad_softmin[i] += w_boundary[i] * (-pts[i] / norms[i])
    safe_norms = np.where(norms > 1e-12, norms, 1.0)
    grad_softmin -= (w_boundary / safe_norms)[:, None] * pts
    
    # Pairwise part
    # w_pair = weights[:n*(n-1)//2]
    w_pair = weights[:n*(n-1)//2]
    # d(dist_ij)/dx_i = (x_i - x_j) / dist_ij
    # d(dist_ij)/dx_j = (x_j - x_i) / dist_ij
    # grad_softmin[i] += w_pair[k] * (x_i - x_j) / dist_ij
    # grad_softmin[j] += w_pair[k] * (x_j - x_i) / dist_ij
    
    # Vectorized update using np.add.at
    # pair_dists is the distance matrix flattened
    # diff is the difference matrix
    # We need to accumulate for i and j
    
    # Let's iterate or use advanced indexing. N=50 is small enough for a loop if vectorization is complex,
    # but let's try vectorization.
    
    # diff[i,j] = x_i - x_j
    # unit_vec = diff / dists[:,:,None]
    # unit_vec[i,j] = (x_i - x_j) / dist_ij
    
    safe_dists = np.where(dists > 1e-12, dists, 1.0)
    unit_vec = diff / safe_dists[:, :, None]
    
    # We only care about i < j
    # w_pair corresponds to the flattened upper triangle.
    # Let's create a weight matrix W where W[i,j] = w_pair[k] if i<j else 0.
    W = np.zeros((n, n))
    W[pair_idx] = w_pair
    
    # grad_softmin[i] += sum_j W[i,j] * unit_vec[i,j]
    # grad_softmin[j] += sum_i W[i,j] * unit_vec[j,i]  (Note: unit_vec[j,i] = -unit_vec[i,j])
    # Actually, d(dist_ij)/dx_j = - d(dist_ij)/dx_i
    # So grad_softmin[j] -= sum_i W[i,j] * unit_vec[i,j]
    
    # Let's compute:
    # term_i = sum_j W[i,j] * unit_vec[i,j]
    # term_j = - sum_i W[i,j] * unit_vec[i,j]
    
    # Using matrix multiplication:
    # term_i = W @ unit_vec  (row i, sum over j)
    # term_j = - W.T @ unit_vec (col j, sum over i)
    
    grad_softmin += W @ unit_vec
    grad_softmin -= W.T @ unit_vec
    
    # Objective is -softmin
    grad_obj = -grad_softmin
    
    return -soft_min_val, grad_obj.flatten()

def _hex_init(n):
    """Hexagonal lattice initialization."""
    best_pts = None
    for spacing in np.linspace(0.12, 0.25, 40):
        k = int(2.0 / spacing) + 2
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                    rows.append((x, y))
        pts = np.array(rows)
        if len(pts) >= n:
            r = np.linalg.norm(pts, axis=1)
            idx = np.argsort(r, kind="stable")[:n]
            best_pts = pts[idx]
            break
    if best_pts is None:
        best_pts = _random_init(n)
    return best_pts

def _ring_init(n):
    """Concentric rings initialization."""
    pts = np.zeros((n, 2))
    # Distribute points in rings
    # Ring 0: 1 point
    # Ring 1: 6 points
    # Ring 2: 12 points
    # Ring 3: 18 points
    # Ring 4: 13 points (total 50)
    # Radii: 0, 0.3, 0.6, 0.85, 0.95 (approx)
    
    rings = [
        (1, 0.0),
        (6, 0.35),
        (12, 0.65),
        (18, 0.85),
        (13, 0.95)
    ]
    
    idx = 0
    for count, radius in rings:
        if idx >= n:
            break
        if count == 1:
            pts[idx] = [0, 0]
            idx += 1
        else:
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
            pts[idx:idx+count, 0] = radius * np.cos(angles)
            pts[idx:idx+count, 1] = radius * np.sin(angles)
            idx += count
            
    return pts

def _random_init(n):
    """Random points in unit disk."""
    theta = np.random.uniform(0, 2 * np.pi, n)
    rho = np.sqrt(np.random.uniform(0, 1, n))
    return np.column_stack([rho * np.cos(theta), rho * np.sin(theta)])

def _optimize_single_start(pts, n, deadline):
    """Run L-BFGS from a single start point."""
    x0 = pts.flatten()
    
    # Multi-stage beta: start with low beta (smoother), increase to high beta (sharper)
    betas = [10.0, 50.0, 200.0, 1000.0, 5000.0]
    
    current_x = x0
    best_score = _compute_score(pts)
    best_x = x0.copy()
    
    for beta in betas:
        if time.time() > deadline - 5:
            break
            
        # Define objective and gradient for this beta
        def obj_grad(x):
            return _objective_and_grad(x, n, beta)
            
        # Unpack for scipy
        def obj(x):
            val, _ = obj_grad(x)
            return val
            
        def grad(x):
            _, g = obj_grad(x)
            return g
            
        try:
            result = minimize(
                obj, 
                current_x, 
                jac=grad, 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            if result.fun < 0: # It's negative softmin, so lower is better (more negative means larger min)
                # Wait, objective is -softmin. We want to maximize softmin, so minimize -softmin.
                # So result.fun should be decreasing.
                pass
                
            new_pts = result.x.reshape(n, 2)
            s = _compute_score(new_pts)
            if s > best_score:
                best_score = s
                best_x = result.x.copy()
                current_x = result.x
        except Exception:
            pass
            
    return best_x.reshape(n, 2), best_score

def construct():
    t_start = time.time()
    deadline = t_start + 145.0
    
    # List of initializations
    inits = [
        _hex_init(N),
        _ring_init(N),
        _random_init(N),
        _random_init(N),
        _random_init(N)
    ]
    
    # Add jittered versions of hex
    for i in range(3):
        jittered = _hex_init(N) + np.random.randn(N, 2) * 0.05
        # Clip to disk
        r = np.linalg.norm(jittered, axis=1)
        mask = r > 0.99
        if np.any(mask):
            jittered[mask] *= 0.99 / r[mask]
        inits.append(jittered)
        
    best_pts = None
    best_score = 0.0
    
    for init in inits:
        if time.time() > deadline - 10:
            break
            
        pts, score = _optimize_single_start(init, N, deadline)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
            
    # Final polish with very high beta on the best result
    if best_pts is not None and time.time() < deadline - 2:
        # One more L-BFGS with high beta
        x0 = best_pts.flatten()
        beta = 10000.0
        
        def obj(x):
            val, _ = _objective_and_grad(x, N, beta)
            return val
        def grad(x):
            _, g = _objective_and_grad(x, N, beta)
            return g
            
        try:
            result = minimize(
                obj, 
                x0, 
                jac=grad, 
                method='L-BFGS-B', 
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            new_pts = result.x.reshape(N, 2)
            s = _compute_score(new_pts)
            if s > best_score:
                best_score = s
                best_pts = new_pts
        except Exception:
            pass

    # Ensure validity
    if best_pts is None:
        best_pts = _hex_init(N)
        
    r = np.linalg.norm(best_pts, axis=1)
    if np.any(r > 1.0):
        best_pts[r > 1.0] *= 1.0 / r[r > 1.0]
        
    return best_pts

# Dummy main to allow testing if needed
if __name__ == "__main__":
    pts = construct()
    print(_compute_score(pts))

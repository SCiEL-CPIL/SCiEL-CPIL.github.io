"""
Strategy: Direct Min-Max Optimization using L-BFGS-B on a Smooth Surrogate.

The current program uses a physics-based force relaxation (gradient descent with
fixed step size) on a sum-of-potentials energy. This new program uses:
1. A smooth surrogate for the "minimum distance" objective: 
   f(pts) = -soft_min(dists, wall_margins)
   where soft_min(x, alpha) = -1/alpha * log(sum(exp(-alpha * x_i)))
   This is a differentiable approximation of min(x).
2. The scipy.optimize.minimize method with L-BFGS-B, which uses a quasi-Newton
   method to adaptively choose step sizes and directions, potentially escaping
   local minima better than fixed-step gradient descent.
3. Multiple random restarts with different initial configurations (not just a grid).
4. A final "polish" phase using a very high alpha (sharp min) to refine the solution.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
CUBE_HALF = 0.5

def _score(points):
    """Return the maximal radius for the given set of centres."""
    if len(points) != N:
        return 0.0
    pts = np.asarray(points, dtype=np.float64)
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Inside the cube [-0.5, 0.5]^3
    margin = 0.5 - np.abs(pts).max(axis=1)
    r_margin = margin.min()
    if r_margin <= 0:
        return 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = dists.min()
    r_pair = min_dist / 2.0
    return min(r_pair, r_margin)

def _soft_min(values, alpha):
    """
    Compute the soft minimum of an array of values.
    soft_min(x) = -1/alpha * log(sum(exp(-alpha * x_i)))
    As alpha -> infinity, this approaches min(x).
    """
    # For numerical stability, subtract the max (which is the min of -x)
    # We want min(x). Let y = -x. Then min(x) = -max(y).
    # max(y) = 1/alpha * log(sum(exp(alpha * y_i)))
    # So min(x) = -1/alpha * log(sum(exp(-alpha * x_i)))
    
    # To avoid overflow in exp, we can scale.
    # Let m = min(values). Then values - m >= 0.
    # sum(exp(-alpha * (v - m))) = exp(alpha * m) * sum(exp(-alpha * v))
    # log(sum(exp(-alpha * v))) = alpha * m + log(sum(exp(-alpha * (v - m))))
    # -1/alpha * log(...) = -m - 1/alpha * log(sum(exp(-alpha * (v - m))))
    
    m = np.min(values)
    shifted = values - m
    # exp(-alpha * shifted) is <= 1, so no overflow.
    sum_exp = np.sum(np.exp(-alpha * shifted))
    if sum_exp == 0:
        return m
    return m - (1.0 / alpha) * np.log(sum_exp)

def _objective_and_grad(flat_pts, alpha):
    """
    Objective function for L-BFGS-B.
    We want to MINIMIZE the negative of the soft-min radius.
    Radius r is limited by:
    1. Wall margins: w_i = 0.5 - max(|x_i|, |y_i|, |z_i|)
    2. Pairwise half-distances: d_ij / 2
    
    We create a vector of all these constraints:
    C = [w_1, ..., w_N, d_12/2, d_13/2, ..., d_{N-1,N}/2]
    
    Objective = -soft_min(C, alpha)
    Gradient = -grad_soft_min(C, alpha)
    """
    pts = flat_pts.reshape(N, 3)
    
    # 1. Wall margins
    abs_pts = np.abs(pts)
    # w_i = 0.5 - max(abs_pts[i])
    # Gradient of w_i w.r.t pts[i,k] is -sign(pts[i,k]) if k is argmax, else 0.
    # To make it differentiable, we can use a soft-max on the absolute values?
    # Or just use the hard max and rely on L-BFGS-B handling non-smoothness?
    # L-BFGS-B assumes smoothness. A soft-max is better.
    
    # Soft max of |x|, |y|, |z|
    # Let A = alpha_wall
    # soft_max_abs = 1/A * log(sum(exp(A * |x|)))
    # w_i = 0.5 - soft_max_abs_i
    
    alpha_wall = 100.0 # High alpha for sharp max
    exp_abs = np.exp(alpha_wall * abs_pts)
    log_sum_exp_abs = np.log(np.sum(exp_abs, axis=1))
    soft_max_abs = log_sum_exp_abs / alpha_wall
    wall_margins = 0.5 - soft_max_abs
    
    # Gradient of wall_margins w.r.t pts
    # d(soft_max_abs)/d(pts[i,k]) = (alpha_wall * exp(alpha_wall * |pts[i,k]|) * sign(pts[i,k])) / (alpha_wall * sum(exp(...)))
    # = exp(alpha_wall * |pts[i,k]|) * sign(pts[i,k]) / sum(exp(...))
    # d(wall_margins)/d(pts[i,k]) = - d(soft_max_abs)/d(pts[i,k])
    
    denom = np.sum(exp_abs, axis=1, keepdims=True)
    grad_wall = - (exp_abs * np.sign(pts)) / denom
    
    # 2. Pairwise distances
    # d_ij = ||p_i - p_j||
    # We need d_ij / 2
    # Gradient of d_ij w.r.t p_i = (p_i - p_j) / d_ij
    # Gradient of d_ij/2 w.r.t p_i = 0.5 * (p_i - p_j) / d_ij
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dists, 1e-10) # Avoid div by 0
    
    # We only care about upper triangle for the vector C
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists = dists[i_idx, j_idx]
    half_pair_dists = pair_dists / 2.0
    
    # Gradient of half_pair_dists w.r.t pts
    # For pair (i, j):
    # d(d_ij/2)/dp_i = 0.5 * (p_i - p_j) / d_ij
    # d(d_ij/2)/dp_j = 0.5 * (p_j - p_i) / d_ij = - d(d_ij/2)/dp_i
    
    grad_pairs = np.zeros_like(pts)
    vecs = diff[i_idx, j_idx]
    norms = pair_dists
    # Avoid div by zero
    norms_safe = np.where(norms < 1e-10, 1e-10, norms)
    unit_vecs = vecs / norms_safe[:, np.newaxis]
    
    # Accumulate gradients
    # For each pair k in i_idx, j_idx:
    # grad[i_idx[k]] += 0.5 * unit_vecs[k]
    # grad[j_idx[k]] -= 0.5 * unit_vecs[k]
    
    np.add.at(grad_pairs, i_idx, 0.5 * unit_vecs)
    np.add.at(grad_pairs, j_idx, -0.5 * unit_vecs)

    # Combine constraints
    # C = [wall_margins, half_pair_dists]
    C = np.concatenate([wall_margins, half_pair_dists])
    grad_C = np.concatenate([grad_wall.flatten(), grad_pairs.flatten()])
    
    # Soft min and its gradient
    # soft_min(C) = m - 1/alpha * log(sum(exp(-alpha * (C - m))))
    # Let S = sum(exp(-alpha * (C - m)))
    # d(soft_min)/dC_k = -1/alpha * d/dC_k [ log(S) ]
    # = -1/alpha * (1/S) * d(S)/dC_k
    # d(S)/dC_k = -alpha * exp(-alpha * (C_k - m))
    # So d(soft_min)/dC_k = (1/S) * exp(-alpha * (C_k - m))
    
    m = np.min(C)
    shifted = C - m
    exp_shifted = np.exp(-alpha * shifted)
    S = np.sum(exp_shifted)
    if S == 0:
        S = 1e-300
        
    weights = exp_shifted / S
    
    # Gradient of soft_min w.r.t C
    grad_soft_min_C = weights
    
    # Gradient of objective (-soft_min) w.r.t C is -weights
    # Chain rule: grad_obj = -weights @ grad_C
    
    grad_obj = - (weights @ grad_C)
    
    val = -soft_min(C, alpha)
    
    return val, grad_obj

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_points = None
    best_score = 0.0
    
    # Initial configurations: Grid + Random
    # 1. Grid
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.375, 0.375, nz)
    grid_pts = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    
    initializations = [grid_pts]
    
    # 2. Random initializations
    rng = np.random.default_rng(42)
    for _ in range(5):
        rand_pts = rng.uniform(-0.45, 0.45, (N, 3))
        initializations.append(rand_pts)
    
    # Optimization strategy:
    # Start with low alpha (smooth), increase alpha (sharp) to converge to min.
    alphas = [10.0, 50.0, 100.0, 200.0, 500.0]
    
    for init_idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 20:
            break
            
        current_pts = init_pts.copy()
        
        # Multi-stage optimization with increasing alpha
        for alpha in alphas:
            if time.time() > deadline - 10:
                break
                
            flat_pts = current_pts.flatten()
            
            try:
                res = minimize(
                    _objective_and_grad,
                    flat_pts,
                    args=(alpha,),
                    jac=True,
                    method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * (N * 3),
                    options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-5}
                )
                if res.success or res.fun < 1e6:
                    current_pts = res.x.reshape(N, 3)
            except Exception:
                pass
            
        score = _score(current_pts)
        if score > best_score:
            best_score = score
            best_points = current_pts.copy()
            
    # Final polish: Use the best points found and run a few more iterations with high alpha
    if best_points is not None and time.time() < deadline - 5:
        current_pts = best_points.copy()
        for alpha in [1000.0, 2000.0]:
            if time.time() > deadline - 2:
                break
            flat_pts = current_pts.flatten()
            try:
                res = minimize(
                    _objective_and_grad,
                    flat_pts,
                    args=(alpha,),
                    jac=True,
                    method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * (N * 3),
                    options={'maxiter': 20, 'ftol': 1e-10, 'gtol': 1e-6}
                )
                if res.success or res.fun < 1e6:
                    current_pts = res.x.reshape(N, 3)
            except Exception:
                pass
        
        score = _score(current_pts)
        if score > best_score:
            best_score = score
            best_points = current_pts.copy()
            
    if best_points is None:
        # Fallback to grid
        best_points = grid_pts
        
    return best_points

# EVOLVE-BLOCK-END

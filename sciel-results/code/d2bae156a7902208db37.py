import numpy as np
import time
from scipy.optimize import minimize

N = 50
TOTAL_TIME = 140.0  # safety margin before 150s kill


def _compute_radius(p):
    """Compute the minimum radius for given centers."""
    # Boundary constraint
    b = 0.5 - np.max(np.abs(p), axis=1)
    min_b = np.min(b)
    # Pairwise constraint
    diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_d = np.min(dists) / 2.0
    return min(min_b, min_d)


def _softmin_radius_and_grad(p, beta):
    """
    Smooth approximation of min radius using log-sum-exp.
    Returns (softmin_value, gradient).
    Higher beta = closer to exact min.
    """
    # Boundary terms: for each point i, dist to boundary
    # r_i_bound = 0.5 - max(|x_i|, |y_i|, |z_i|)
    # We want softmin over these
    abs_p = np.abs(p)
    max_abs = np.max(abs_p, axis=1)
    r_bound = 0.5 - max_abs  # shape (N,)
    
    # Pairwise terms: r_ij = |p_i - p_j| / 2 for i < j
    diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dists, np.inf)
    r_pair = dists / 2.0  # shape (N, N), symmetric, inf on diagonal
    
    # Collect all radius values
    # Boundary: N values
    # Pairwise: N*(N-1)/2 values (upper triangle)
    iu = np.triu_indices(N, k=1)
    r_vals = np.concatenate([r_bound, r_pair[iu]])  # shape (N + N*(N-1)/2,)
    
    # Log-sum-exp softmin: -log(sum(exp(-beta * r))) / beta
    # This approximates min(r) as beta -> inf
    neg_br = -beta * r_vals
    # Shift for numerical stability
    max_neg_br = np.max(neg_br)
    shifted = neg_br - max_neg_br
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    
    if sum_exp < 1e-300:
        softmin_val = np.min(r_vals)
    else:
        softmin_val = -(max_neg_br + np.log(sum_exp)) / beta
    
    # Gradient: d(softmin)/d(p) 
    # softmin = -log(sum(exp(-beta*r))) / beta
    # d(softmin)/dp = -1/beta * d(log(sum(exp(-beta*r))))/dp
    #                = -1/beta * sum_j (-beta * exp(-beta*r_j) / sum(exp(-beta*r))) * dr_j/dp
    #                = sum_j exp(-beta*r_j)/sum(exp(-beta*r)) * dr_j/dp
    #                = sum_j w_j * dr_j/dp  where w_j = exp(-beta*r_j)/sum(exp(-beta*r))
    
    weights = exp_shifted / sum_exp  # shape (n_vals,)
    w_bound = weights[:N]  # shape (N,)
    w_pair = weights[N:]   # shape (N*(N-1)/2,)
    
    grad = np.zeros_like(p)
    
    # Boundary gradient: dr_bound_i/dp_i = -sign(p_i) * one_hot(axis_of_max)
    for i in range(N):
        w = w_bound[i]
        if w < 1e-15:
            continue
        max_idx = np.argmax(abs_p[i])
        g = np.zeros(3)
        g[max_idx] = -np.sign(p[i, max_idx])
        grad[i] += w * g
    
    # Pairwise gradient: dr_ij/dp_i = (p_i - p_j) / (2*|p_i - p_j|)
    # r_ij = |p_i - p_j| / 2
    # dr_ij/dp_i = (p_i - p_j) / |p_i - p_j| / 2
    # dr_ij/dp_j = -(p_i - p_j) / |p_i - p_j| / 2
    
    # Map w_pair back to pair indices
    for idx in range(len(iu[0])):
        w = w_pair[idx]
        if w < 1e-15:
            continue
        i, j = iu[0][idx], iu[1][idx]
        d = p[i] - p[j]
        dist = np.linalg.norm(d)
        if dist < 1e-15:
            continue
        unit = d / dist / 2.0
        grad[i] += w * unit
        grad[j] -= w * unit
    
    return softmin_val, grad


def _epigraph_objective(params):
    """
    Objective: maximize r subject to r <= all constraints.
    Epigraph form: minimize -r + penalty * sum(softplus(r - r_i))
    where r_i are the individual constraint radii.
    """
    p = params[:N*3].reshape(N, 3)
    r = params[N*3]
    
    # Boundary radii
    abs_p = np.abs(p)
    max_abs = np.max(abs_p, axis=1)
    r_bound = 0.5 - max_abs
    
    # Pairwise radii
    diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dists, np.inf)
    r_pair = dists / 2.0
    
    iu = np.triu_indices(N, k=1)
    r_pair_flat = r_pair[iu]
    
    # Penalty for violating r <= r_i
    # softplus(r - r_i) is 0 when r <= r_i, grows when r > r_i
    penalty_coeff = 100.0
    
    violations_bound = np.maximum(0, r - r_bound)
    violations_pair = np.maximum(0, r - r_pair_flat)
    
    penalty = penalty_coeff * (np.sum(violations_bound**2) + np.sum(violations_pair**2))
    
    return -r + penalty


def _lloyd_step(p, n_iters=3):
    """
    Lloyd's algorithm: for each point, move it to the center of its Voronoi cell
    (approximated by the mean of nearby points and boundaries).
    This helps equalize distances.
    """
    for _ in range(n_iters):
        new_p = p.copy()
        for i in range(N):
            # Find the "empty space" around point i
            # Approximate: move i toward the average of its nearest neighbors' midpoints
            # and away from boundaries
            diffs = p - p[i]
            dists = np.linalg.norm(diffs, axis=1)
            dists[i] = np.inf
            # Take 4 nearest neighbors
            k = min(4, N-1)
            nn_idx = np.argsort(dists)[:k]
            
            # Midpoints to neighbors
            midpoints = (p[i] + p[nn_idx]) / 2.0
            avg_mid = np.mean(midpoints, axis=0)
            
            # Boundary push: move away from closest boundary
            min_dist_to_boundary = np.min(0.5 - np.abs(p[i]))
            closest_boundary_axis = np.argmin(0.5 - np.abs(p[i]))
            boundary_push = np.zeros(3)
            boundary_push[closest_boundary_axis] = np.sign(p[i, closest_boundary_axis]) * 0.1
            
            # New position: blend of current and midpoint average
            new_p[i] = 0.7 * p[i] + 0.3 * avg_mid + 0.05 * boundary_push
        
        # Clamp to valid region
        new_p = np.clip(new_p, -0.49, 0.49)
        p = new_p
    
    return p


def _local_search_nm(p, time_limit=2.0):
    """
    Exact Nelder-Mead polish to maximize the true min-radius.
    """
    start_t = time.time()
    
    def neg_radius(p_flat):
        pf = p_flat.reshape(N, 3)
        pf = np.clip(pf, -0.49, 0.49)
        return -_compute_radius(pf)
    
    best = p.copy()
    best_val = neg_radius(p.ravel())
    
    try:
        res = minimize(
            neg_radius,
            p.ravel(),
            method='Nelder-Mead',
            options={
                'maxiter': 500,
                'xatol': 1e-7,
                'fatol': 1e-9,
                'adaptive': True
            }
        )
        if res.fun < best_val:
            best = np.clip(res.x.reshape(N, 3), -0.49, 0.49)
            best_val = res.fun
    except Exception:
        pass
    
    # Additional local kicks
    while time.time() - start_t < time_limit:
        improved = False
        for i in range(N):
            for axis in range(3):
                for delta in [0.001, -0.001, 0.005, -0.005, 0.01, -0.01]:
                    trial = best.copy()
                    trial[i, axis] += delta
                    trial = np.clip(trial, -0.49, 0.49)
                    r_trial = _compute_radius(trial)
                    r_best = _compute_radius(best)
                    if r_trial > r_best + 1e-12:
                        best = trial
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return best


def _initial_fcc(n):
    """Generate FCC-like initial configuration."""
    points = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            for k in range(-5, 6):
                points.append([i, j, k])
                points.append([i + 0.5, j + 0.5, k])
                points.append([i + 0.5, j, k + 0.5])
                points.append([i, j + 0.5, k + 0.5])
    points = np.array(points)
    points -= np.mean(points, axis=0)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    pts = points[idx[:n]]
    
    # Scale to fit in cube with some margin
    max_coord = np.max(np.abs(pts))
    if max_coord > 0:
        scale = 0.4 / max_coord
        pts *= scale
    
    return np.clip(pts, -0.49, 0.49)


def _initial_grid(n):
    """Generate a grid-like initial configuration."""
    # 3x4x5 grid = 60 points, take 50
    xs = np.linspace(-0.35, 0.35, 3)
    ys = np.linspace(-0.35, 0.35, 4)
    zs = np.linspace(-0.35, 0.35, 5)
    
    points = []
    for x in xs:
        for y in ys:
            for z in zs:
                points.append([x, y, z])
    
    points = np.array(points[:n])
    # Add small random jitter
    np.random.seed(123)
    points += np.random.uniform(-0.01, 0.01, points.shape)
    
    return np.clip(points, -0.49, 0.49)


def _initial_random(n, seed=0):
    """Generate random initial configuration."""
    np.random.seed(seed)
    points = np.random.uniform(-0.4, 0.4, (n, 3))
    return points


def _initial_bcc(n):
    """Generate BCC-like initial configuration."""
    points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                points.append([i, j, k])
                points.append([i + 0.5, j + 0.5, k + 0.5])
    points = np.array(points)
    points -= np.mean(points, axis=0)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    pts = points[idx[:n]]
    
    max_coord = np.max(np.abs(pts))
    if max_coord > 0:
        scale = 0.4 / max_coord
        pts *= scale
    
    return np.clip(pts, -0.49, 0.49)


def construct():
    start_time = time.time()
    
    # Multiple initializations
    inits = [
        _initial_fcc(N),
        _initial_grid(N),
        _initial_random(N, seed=0),
        _initial_bcc(N),
        _initial_random(N, seed=42),
        _initial_random(N, seed=123)
    ]
    
    best_pts = None
    best_r = -np.inf
    
    # Phase 1: Smooth epigraph optimization with L-BFGS-B
    # Phase 2: Nelder-Mead exact polish
    
    for idx, init in enumerate(inits):
        elapsed = time.time() - start_time
        remaining = TOTAL_TIME - elapsed
        if remaining < 15.0:
            break
        
        # Phase 1: L-BFGS-B on epigraph formulation
        p0 = init.copy()
        # Start with r estimate
        r0 = _compute_radius(p0) * 0.9
        
        params0 = np.concatenate([p0.ravel(), [r0]])
        
        # Time budget for this init's L-BFGS-B
        lbfgs_time = min(12.0, remaining * 0.5)
        if lbfgs_time < 3.0:
            continue
        
        def epigraph_obj(params):
            return _epigraph_objective(params)
        
        def epigraph_grad(params):
            p = params[:N*3].reshape(N, 3)
            r = params[N*3]
            
            grad = np.zeros_like(params)
            
            # Gradient w.r.t. r: -1 + 2*penalty_coeff*sum(violations)
            abs_p = np.abs(p)
            max_abs = np.max(abs_p, axis=1)
            r_bound = 0.5 - max_abs
            diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dists = np.linalg.norm(diffs, axis=2)
            np.fill_diagonal(dists, np.inf)
            r_pair = dists / 2.0
            iu = np.triu_indices(N, k=1)
            r_pair_flat = r_pair[iu]
            
            v_bound = np.maximum(0, r - r_bound)
            v_pair = np.maximum(0, r - r_pair_flat)
            
            penalty_coeff = 100.0
            grad[N*3] = -1.0 + 2.0 * penalty_coeff * (np.sum(v_bound) + np.sum(v_pair))
            
            # Gradient w.r.t. p
            # d/dp sum(v_bound^2) = sum_i 2*v_bound_i * dr_bound_i/dp_i
            # dr_bound_i/dp_i = -sign(p_i) * one_hot(max_axis)
            for i in range(N):
                v = v_bound[i]
                if v < 1e-15:
                    continue
                max_idx = np.argmax(abs_p[i])
                g = np.zeros(3)
                g[max_idx] = -np.sign(p[i, max_idx])
                grad[i*3:(i+1)*3] += 2.0 * penalty_coeff * v * g
            
            # d/dp sum(v_pair^2)
            # dr_ij/dp_i = (p_i - p_j) / |p_i - p_j| / 2
            # dr_ij/dp_j = -(p_i - p_j) / |p_i - p_j| / 2
            for idx2 in range(len(iu[0])):
                v = v_pair[idx2]
                if v < 1e-15:
                    continue
                i, j = iu[0][idx2], iu[1][idx2]
                d = p[i] - p[j]
                dist = np.linalg.norm(d)
                if dist < 1e-15:
                    continue
                unit = d / dist / 2.0
                grad[i*3:(i+1)*3] += 2.0 * penalty_coeff * v * unit
                grad[j*3:(j+1)*3] -= 2.0 * penalty_coeff * v * unit
            
            return grad
        
        lbfgs_start = time.time()
        
        try:
            # Use a callback to enforce time limit
            def lbfgs_callback(xk):
                if time.time() - lbfgs_start > lbfgs_time:
                    raise TimeoutError("LBFGS time limit")
            
            bounds = [(-0.49, 0.49)] * (N*3) + [(0.0, 0.25)]
            
            res = minimize(
                epigraph_obj,
                params0,
                jac=epigraph_grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': 200,
                    'ftol': 1e-10,
                    'gtol': 1e-6,
                    'maxfun': 1000
                },
                callback=lbfgs_callback
            )
            
            if hasattr(res, 'x') and res.x is not None:
                p_res = np.clip(res.x[:N*3].reshape(N, 3), -0.49, 0.49)
                r_res = res.x[N*3]
                
                # Compute actual radius
                r_actual = _compute_radius(p_res)
                
                # Lloyd step to improve
                p_lloyd = _lloyd_step(p_res, n_iters=2)
                r_lloyd = _compute_radius(p_lloyd)
                
                if r_lloyd > r_actual:
                    p_candidate = p_lloyd
                    r_candidate = r_lloyd
                else:
                    p_candidate = p_res
                    r_candidate = r_actual
                
                if r_candidate > best_r:
                    best_r = r_candidate
                    best_pts = p_candidate.copy()
                    
        except TimeoutError:
            pass
        except Exception:
            pass
        
        # Phase 2: Nelder-Mead exact polish
        if best_pts is not None:
            elapsed = time.time() - start_time
            remaining = TOTAL_TIME - elapsed
            nm_time = min(3.0, remaining * 0.3)
            if nm_time > 1.0:
                p_polished = _local_search_nm(best_pts, time_limit=nm_time)
                r_polished = _compute_radius(p_polished)
                if r_polished > best_r:
                    best_r = r_polished
                    best_pts = p_polished.copy()
    
    # Final safety: if no valid result, return something reasonable
    if best_pts is None:
        best_pts = _initial_fcc(N)
    
    # Final verification and clamping
    best_pts = np.clip(best_pts, -0.49, 0.49)
    r_final = _compute_radius(best_pts)
    
    # One last local polish if time allows
    elapsed = time.time() - start_time
    remaining = TOTAL_TIME - elapsed
    if remaining > 2.0:
        p_final = _local_search_nm(best_pts, time_limit=min(2.0, remaining - 1.0))
        r_final2 = _compute_radius(p_final)
        if r_final2 > r_final:
            best_pts = p_final
    
    return best_pts

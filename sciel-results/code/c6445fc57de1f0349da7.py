"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13

# EVOLVE-BLOCK-START

# Pre-compute triangle indices for efficiency
# i < j < k
_idx_i, _idx_j, _idx_k = np.indices((N, N, N)).reshape(3, -1)
_mask = (_idx_i < _idx_j) & (_idx_j < _idx_k)
_TRI_I = _idx_i[_mask]
_TRI_J = _idx_j[_mask]
_TRI_K = _idx_k[_mask]

def _compute_score(pts):
    """
    Compute the minimum triangle area normalized by the convex hull area.
    Returns 0 if invalid.
    """
    pts = np.asarray(pts)
    if pts.shape != (N, 2):
        return 0.0
        
    # Compute Convex Hull Area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area < 1e-9:
            return 0.0
    except Exception:
        return 0.0

    # Compute all triangle areas vectorized
    p_i = pts[_TRI_I]
    p_j = pts[_TRI_J]
    p_k = pts[_TRI_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def _smooth_min_area(pts, beta):
    """
    Compute a smooth approximation of the minimum triangle area using log-sum-exp.
    This is differentiable and suitable for gradient-based optimization.
    """
    p_i = pts[_TRI_I]
    p_j = pts[_TRI_J]
    p_k = pts[_TRI_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # We want to maximize this, so we minimize -soft_min
    # To avoid overflow, we use the log-sum-exp trick
    max_area = np.max(areas)
    if max_area == 0:
        return 0.0
        
    # exp(-beta * (areas - max_area)) is stable
    # log(sum(exp(-beta * areas))) = -beta * max_area + log(sum(exp(-beta * (areas - max_area))))
    # soft_min = -1/beta * log(sum(exp(-beta * areas)))
    #          = max_area - 1/beta * log(sum(exp(-beta * (areas - max_area))))
    
    # Wait, standard soft min is:
    # m = min(x) approx by -1/beta * log(sum(exp(-beta*x)))
    # Let's verify: if x_i = m + d_i, exp(-beta*x_i) = exp(-beta*m) * exp(-beta*d_i)
    # sum = exp(-beta*m) * sum(exp(-beta*d_i))
    # log(sum) = -beta*m + log(sum(exp(-beta*d_i)))
    # -1/beta * log(sum) = m - 1/beta * log(sum(exp(-beta*d_i)))
    # If d_i >= 0, the term in log is approx log(1 + sum(exp(-beta*d_i))) ~ 0 if beta large.
    # So soft_min ~ m.
    
    # For numerical stability:
    # We want to minimize -soft_min.
    # Let's compute soft_min directly.
    
    # To handle large beta, we can just compute min if beta is very large, but for gradient we need smooth.
    # Using the standard stable log-sum-exp:
    # LSE(x) = max(x) + log(sum(exp(x - max(x))))
    # Here we have log(sum(exp(-beta * areas)))
    # Let y = -beta * areas
    # max_y = -beta * min(areas)
    # log(sum(exp(y))) = max_y + log(sum(exp(y - max_y)))
    # soft_min = -1/beta * (max_y + log(sum(exp(y - max_y))))
    #          = min(areas) - 1/beta * log(sum(exp(-beta * (areas - min(areas)))))
    
    min_a = np.min(areas)
    # If min_a is 0, areas are 0, return 0
    if min_a < 1e-10:
         return 0.0
         
    # y = -beta * (areas - min_a)
    # Note: areas - min_a >= 0, so y <= 0. exp(y) <= 1. Stable.
    y = -beta * (areas - min_a)
    # sum_exp = sum(exp(y))
    # log_sum_exp = log(sum_exp)
    # soft_min = min_a - 1/beta * log_sum_exp
    
    # To avoid log(0) if all areas are min_a (which implies only 1 triangle or all same),
    # but N=13 ensures many triangles.
    sum_exp = np.sum(np.exp(y))
    if sum_exp == 0:
        return min_a
    log_sum_exp = np.log(sum_exp)
    
    soft_min = min_a - (1.0 / beta) * log_sum_exp
    return soft_min

def _hull_area(pts):
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _objective_smooth(params, beta):
    """
    Objective: - (smooth_min_area / hull_area)
    """
    pts = params.reshape(N, 2)
    h_area = _hull_area(pts)
    if h_area < 1e-9:
        return 1e9
    
    s_min = _smooth_min_area(pts, beta)
    if s_min <= 0:
        return 1e9
        
    return -(s_min / h_area)

def _generate_starts(rng):
    """Generate diverse initial points for multi-start optimization."""
    starts = []
    
    # 1. Regular polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    p_reg = np.column_stack((np.cos(angles), np.sin(angles)))
    starts.append(p_reg)
    
    # 2. Jittered regular polygons
    for _ in range(8):
        noise = rng.normal(0, 0.2, size=(N, 2))
        p_jit = p_reg + noise
        starts.append(p_jit)
        
    # 3. Random convex-like starts (sorted angles)
    for _ in range(8):
        angles = np.sort(rng.uniform(0, 2 * np.pi, N))
        radii = rng.uniform(0.5, 1.5, N)
        p_rand = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        starts.append(p_rand)
        
    return starts

def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_score = 0.0
    
    # Phase 1: Differential evolution with moderate beta
    # DE is a population-based global optimizer that explores broadly
    def _de_objective(params):
        pts = params.reshape(N, 2)
        h_area = _hull_area(pts)
        if h_area < 1e-9:
            return 1e9
        s_min = _smooth_min_area(pts, 20.0)
        if s_min <= 0:
            return 1e9
        return -(s_min / h_area)
    
    try:
        from scipy.optimize import differential_evolution
        
        # Bounds: points in [-2, 2] x [-2, 2]
        bounds = [(-2.0, 2.0)] * (2 * N)
        
        de_result = differential_evolution(
            _de_objective,
            bounds,
            seed=42,
            maxiter=50,
            popsize=20,
            tol=1e-4,
            polish=False,  # We'll do our own polish
            init='sobol',
            update_interval=1,
            callback=lambda xk, convergence: (time.time() - start_time) > deadline - 30.0
        )
        
        if de_result.fun < 1e8:
            candidate_pts = de_result.x.reshape(N, 2)
            candidate_score = _compute_score(candidate_pts)
            if candidate_score > best_score:
                best_score = candidate_score
                best_pts = candidate_pts.copy()
                
    except Exception:
        pass
    
    # Phase 2: Multi-start BFGS with increasing beta from best point
    if best_pts is not None:
        current_pts = best_pts.copy()
        betas = [20.0, 100.0, 500.0, 2000.0]
        
        for beta in betas:
            if (time.time() - start_time) > deadline - 15.0:
                break
            
            try:
                res = minimize(
                    _objective_smooth,
                    current_pts.flatten(),
                    args=(beta,),
                    method='BFGS',
                    options={'maxiter': 300, 'gtol': 1e-7}
                )
                
                if res.fun < 1e8:
                    new_pts = res.x.reshape(N, 2)
                    new_score = _compute_score(new_pts)
                    
                    if new_score > best_score:
                        best_score = new_score
                        best_pts = new_pts.copy()
                        current_pts = best_pts.copy()
                        
            except Exception:
                continue
    else:
        # Fallback: try a few random starts with BFGS
        starts = _generate_starts(rng)
        for p0 in starts[:5]:
            if (time.time() - start_time) > deadline - 15.0:
                break
            try:
                res = minimize(
                    _objective_smooth,
                    p0.flatten(),
                    args=(50.0,),
                    method='BFGS',
                    options={'maxiter': 200, 'gtol': 1e-6}
                )
                if res.fun < 1e8:
                    new_pts = res.x.reshape(N, 2)
                    new_score = _compute_score(new_pts)
                    if new_score > best_score:
                        best_score = new_score
                        best_pts = new_pts.copy()
            except Exception:
                continue
    
    # Phase 3: Final polish with Nelder-Mead on exact objective
    if best_pts is not None and (time.time() - start_time) < deadline - 5.0:
        try:
            def _exact_obj(params):
                pts = params.reshape(N, 2)
                s = _compute_score(pts)
                return -s if s > 0 else 1e9
            
            res_polish = minimize(
                _exact_obj,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
            )
            
            if res_polish.fun < 1e8:
                polish_pts = res_polish.x.reshape(N, 2)
                polish_score = _compute_score(polish_pts)
                
                if polish_score > best_score:
                    best_score = polish_score
                    best_pts = polish_pts.copy()
        except Exception:
            pass
    
    if best_pts is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        return np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_pts

# EVOLVE-BLOCK-END

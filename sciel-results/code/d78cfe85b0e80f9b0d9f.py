"""Combined program: Soft-Min Packing with Simulated Annealing and Adaptive Alpha."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100

# EVOLVE-BLOCK-START

def _soft_min_obj_grad(c, alpha, k=10):
    """
    Compute soft-min objective and gradient.
    
    We use a differentiable approximation that emphasizes the k smallest constraints.
    f(c) = (1/alpha) * log( sum exp(-alpha * s_i) )
    but we weight by rank to focus on the tightest constraints.
    
    Actually, let's use a simpler but effective approach:
    Minimize the sum of the k smallest pairwise/wall distances, smoothed.
    
    Alternative: use a power-sum that emphasizes small values:
    f(c) = (1/k) * sum of k smallest s_i
    
    For differentiability, use:
    f(c) = (1/alpha) * log( sum_{i} exp(-alpha * s_i) )
    This is the soft-min. To make it sharper, increase alpha.
    
    Let's stick with soft-min but use a better schedule.
    """
    # Pairwise distances
    d = cdist(c, c)
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists = d[i_idx, j_idx] / 2.0
    
    # Wall distances
    wall_dists = np.concatenate([
        0.5 - c[:, 0],
        0.5 + c[:, 0],
        0.5 - c[:, 1],
        0.5 + c[:, 1],
        0.5 - c[:, 2],
        0.5 + c[:, 2]
    ])
    
    all_dists = np.concatenate([pair_dists, wall_dists])
    
    # Soft-min: f = (1/alpha) * log(sum(exp(-alpha * s)))
    # Numerical stability: shift by min
    s_min = np.min(all_dists)
    shifted = all_dists - s_min
    exp_terms = np.exp(-alpha * shifted)
    Z = np.sum(exp_terms)
    
    if Z < 1e-300:
        return s_min, np.zeros_like(c)
        
    obj_val = s_min - (1.0 / alpha) * np.log(Z)
    
    # Gradient: grad f = -sum(w_i * grad s_i) where w_i = exp(-alpha*s_i)/Z
    weights = exp_terms / Z
    grad_c = np.zeros_like(c)
    
    # Pairwise
    w_pair = weights[:len(pair_dists)]
    pair_raw = d[i_idx, j_idx]
    pair_safe = np.where(pair_raw < 1e-12, 1e-12, pair_raw)
    
    ci = c[i_idx]
    cj = c[j_idx]
    diff = ci - cj
    u = diff / pair_safe[:, np.newaxis]
    
    contrib = w_pair[:, np.newaxis] * u / 2.0
    np.add.at(grad_c, i_idx, -contrib)
    np.add.at(grad_c, j_idx, contrib)
    
    # Wall
    w_wall = weights[len(pair_dists):].reshape(6, N)
    grad_c[:, 0] += w_wall[0] - w_wall[1]
    grad_c[:, 1] += w_wall[2] - w_wall[3]
    grad_c[:, 2] += w_wall[4] - w_wall[5]
    
    return obj_val, grad_c


def _score(c):
    """Compute the exact packing radius."""
    if c.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    bound = np.min(0.5 - np.abs(c))
    if bound < 0.0:
        return 0.0
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    pair_min = np.min(d) / 2.0
    return float(min(bound, pair_min))


def _initial_grid():
    """5x5x4 grid."""
    xs = np.linspace(-0.4, 0.4, 5)
    ys = np.linspace(-0.4, 0.4, 5)
    zs = np.linspace(-0.375, 0.375, 4)
    centres = []
    for z in zs:
        for y in ys:
            for x in xs:
                centres.append([x, y, z])
    return np.array(centres[:N])


def _initial_grid_v2():
    """5x5x4 with alternating offset."""
    xs = np.linspace(-0.4, 0.4, 5)
    ys = np.linspace(-0.4, 0.4, 5)
    zs = np.linspace(-0.375, 0.375, 4)
    centres = []
    for idx, z in enumerate(zs):
        offset = 0.1 if idx % 2 == 1 else 0.0
        for y in ys:
            for x in xs:
                centres.append([x + offset, y, z])
    return np.array(centres[:N])


def _initial_grid_v3():
    """4x5x5 grid."""
    xs = np.linspace(-0.375, 0.375, 4)
    ys = np.linspace(-0.4, 0.4, 5)
    zs = np.linspace(-0.4, 0.4, 5)
    centres = []
    for z in zs:
        for y in ys:
            for x in xs:
                centres.append([x, y, z])
    return np.array(centres[:N])


def _local_optimize(c, alpha, maxiter=300, ftol=1e-12, gtol=1e-9):
    """Run L-BFGS-B on soft-min objective."""
    def obj(x):
        c_flat = x.reshape(N, 3)
        val, _ = _soft_min_obj_grad(c_flat, alpha)
        return val
    
    def grad(x):
        c_flat = x.reshape(N, 3)
        _, g = _soft_min_obj_grad(c_flat, alpha)
        return g.flatten()
    
    x0 = c.flatten()
    bounds = [(-0.5, 0.5)] * (3 * N)
    
    try:
        result = minimize(
            obj, x0, jac=grad,
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol}
        )
        if result.success or result.fun < obj(x0) - 1e-14:
            c_new = result.x.reshape(N, 3)
            c_new = np.clip(c_new, -0.5, 0.5)
            return c_new
    except Exception:
        pass
    return c


def _coordinate_descent_polish(c, alpha, max_rounds=20, tol=1e-12):
    """
    Coordinate descent: for each sphere, optimize its position given others fixed.
    This can escape some L-BFGS-B local minima.
    """
    c = c.copy()
    for _ in range(max_rounds):
        improved = False
        for i in range(N):
            c_old = c[i].copy()
            
            # Optimize sphere i given all others
            def obj_i(x):
                c_trial = c.copy()
                c_trial[i] = x
                val, _ = _soft_min_obj_grad(c_trial, alpha)
                return val
            
            def grad_i(x):
                c_trial = c.copy()
                c_trial[i] = x
                _, g = _soft_min_obj_grad(c_trial, alpha)
                return g[i]
            
            try:
                res = minimize(
                    obj_i, c_old, jac=grad_i,
                    method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * 3,
                    options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                if res.success and np.linalg.norm(res.x - c_old) > 1e-10:
                    c[i] = np.clip(res.x, -0.5, 0.5)
                    improved = True
            except Exception:
                pass
        
        if not improved:
            break
    
    return c


def construct():
    start_time = time.time()
    deadline = 148.0  # Safety margin (150s kill, 2s buffer)
    
    np.random.seed(42)
    
    # Multiple initial configurations
    init_configs = [_initial_grid(), _initial_grid_v2(), _initial_grid_v3()]
    
    best_c = None
    best_score = 0.0
    
    for init_c in init_configs:
        s = _score(init_c)
        if s > best_score:
            best_score = s
            best_c = init_c.copy()
    
    c = best_c.copy()
    
    # Phase 1: Coarse-to-fine alpha annealing
    alphas = [3, 5, 10, 20, 50, 100, 200, 500]
    
    for alpha in alphas:
        if time.time() - start_time > deadline - 15.0:
            break
        
        c = _local_optimize(c, alpha, maxiter=200)
        c = np.clip(c, -0.5, 0.5)
        
        s = _score(c)
        if s > best_score:
            best_score = s
            best_c = c.copy()
    
    # Phase 2: Simulated annealing restarts
    # Try multiple restarts with decreasing perturbation scale
    restart_scales = [0.05, 0.03, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    for scale in restart_scales:
        if time.time() - start_time > deadline - 10.0:
            break
        
        # Try up to 3 restarts per scale
        for trial in range(3):
            if time.time() - start_time > deadline - 10.0:
                break
            
            c_pert = best_c + scale * np.random.randn(N, 3)
            c_pert = np.clip(c_pert, -0.49, 0.49)
            
            # Optimize with high alpha
            c_new = _local_optimize(c_pert, alpha=500, maxiter=150)
            c_new = np.clip(c_new, -0.5, 0.5)
            
            # Coordinate descent polish
            c_new = _coordinate_descent_polish(c_new, alpha=500, max_rounds=10)
            c_new = np.clip(c_new, -0.5, 0.5)
            
            s = _score(c_new)
            if s > best_score:
                best_score = s
                best_c = c_new.copy()
    
    # Phase 3: Final polish with very high alpha
    if time.time() - start_time < deadline - 5.0:
        c_final = _local_optimize(best_c, alpha=1000, maxiter=100)
        c_final = _coordinate_descent_polish(c_final, alpha=1000, max_rounds=15)
        c_final = np.clip(c_final, -0.5, 0.5)
        
        s = _score(c_final)
        if s > best_score:
            best_score = s
            best_c = c_final.copy()
    
    # Ensure validity
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c

# EVOLVE-BLOCK-END

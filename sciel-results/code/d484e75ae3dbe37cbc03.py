# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree


def _exact_2d_union_area(rects):
    """
    Compute the exact area of the union of axis-aligned rectangles using sweep-line.
    rects: array of shape (n, 4) with columns [x_lo, x_hi, y_lo, y_hi]
    """
    n = rects.shape[0]
    if n == 0:
        return 0.0
    
    # Collect all unique y-coordinates
    ys = np.unique(np.concatenate([rects[:, 2], rects[:, 3]]))
    if len(ys) < 2:
        return 0.0
    
    # For each y-slab, compute the 1D union length in x
    total_area = 0.0
    for i in range(len(ys) - 1):
        y_mid = 0.5 * (ys[i] + ys[i+1])
        dy = ys[i+1] - ys[i]
        if dy <= 0:
            continue
        
        # Find rectangles that cover this y-slab
        mask = (rects[:, 2] <= y_mid) & (rects[:, 3] >= y_mid)
        if not np.any(mask):
            continue
        
        x_lo = rects[mask, 0]
        x_hi = rects[mask, 1]
        
        # Compute 1D union length
        order = np.argsort(x_lo, kind='stable')
        x_lo_s = x_lo[order]
        x_hi_s = x_hi[order]
        
        x_len = 0.0
        cur_lo = x_lo_s[0]
        cur_hi = x_hi_s[0]
        for k in range(1, len(x_lo_s)):
            if x_lo_s[k] > cur_hi:
                x_len += cur_hi - cur_lo
                cur_lo = x_lo_s[k]
                cur_hi = x_hi_s[k]
            else:
                if x_hi_s[k] > cur_hi:
                    cur_hi = x_hi_s[k]
        x_len += cur_hi - cur_lo
        
        total_area += x_len * dy
    
    return total_area


def _cross_section_area_2d(base_x, base_y, i_flat, j_flat, c, N=8):
    """
    Compute the exact 2D cross-section area of the union at height c.
    """
    N = float(N)
    shift_x = c * i_flat / N
    shift_y = c * j_flat / N
    
    x_lo = base_x + shift_x
    x_hi = x_lo + 1.0 / N
    y_lo = base_y + shift_y
    y_hi = y_lo + 1.0 / N
    
    rects = np.stack([x_lo, x_hi, y_lo, y_hi], axis=1)
    return _exact_2d_union_area(rects)


def _union_volume(base_x, base_y, i_flat, j_flat, N=8, n_quad=48):
    """
    Estimate the union volume via Gauss-Legendre quadrature over c in [0, 1].
    """
    c_nodes, c_weights = np.polynomial.legendre.leggauss(n_quad)
    c_nodes = 0.5 * (c_nodes + 1.0)
    c_weights = 0.5 * c_weights
    
    areas = np.array([_cross_section_area_2d(base_x, base_y, i_flat, j_flat, c, N) for c in c_nodes])
    return np.sum(c_weights * areas)


def _objective(params, i_flat, j_flat, N=8, n_quad=40):
    """
    Objective function for optimization: union volume.
    """
    base_x = params[:64]
    base_y = params[64:]
    return _union_volume(base_x, base_y, i_flat, j_flat, N, n_quad)


def _differential_evolution_custom(f, x0, bounds, maxiter=100, popsize=15, mutation=(0.5, 1.0),
                                    recombination=0.7, tol=1e-8, deadline=None):
    """
    Custom differential evolution with deadline support.
    """
    rng = np.random.default_rng(42)
    n_dim = len(x0)
    n_pop = popsize * n_dim
    
    # Initialize population
    pop = np.zeros((n_pop, n_dim))
    for i in range(n_pop):
        pop[i] = x0 + rng.standard_normal(n_dim) * 0.1
        pop[i] = np.clip(pop[i], bounds[0], bounds[1])
    
    # Evaluate initial population
    fit = np.array([f(ind) for ind in pop])
    best_idx = np.argmin(fit)
    best_fit = fit[best_idx]
    best_ind = pop[best_idx].copy()
    
    # DE parameters
    F_low, F_high = mutation
    CR = recombination
    
    gen = 0
    while gen < maxiter:
        if deadline is not None and time.time() > deadline:
            break
        
        for i in range(n_pop):
            if deadline is not None and time.time() > deadline:
                break
            
            # Select three distinct individuals
            while True:
                a, b, c = rng.choice(n_pop, size=3, replace=False)
                if len(set([a, b, c, i])) == 4:
                    break
            
            # Mutation
            F = rng.uniform(F_low, F_high)
            v = pop[a] + F * (pop[b] - pop[c])
            v = np.clip(v, bounds[0], bounds[1])
            
            # Crossover
            trial = pop[i].copy()
            j_rand = rng.integers(n_dim)
            for j in range(n_dim):
                if rng.random() < CR or j == j_rand:
                    trial[j] = v[j]
            
            # Selection
            trial_fit = f(trial)
            if trial_fit <= fit[i]:
                pop[i] = trial
                fit[i] = trial_fit
                if trial_fit < best_fit:
                    best_fit = trial_fit
                    best_ind = trial.copy()
        
        # Convergence check
        if best_fit - np.min(fit) < tol:
            break
        
        gen += 1
    
    return best_ind, best_fit


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    
    Minimizes the union volume of 64 sheared tubes using differential evolution
    on the full 128-dimensional parameter space, capturing non-separable interactions.
    """
    N = 8
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    start_time = time.time()
    deadline = start_time + 560  # 560 second deadline (70s safety margin)
    
    # Initial guess: Keich-like construction
    pos_k = int(np.log2(N))
    j_terms = np.arange(1, pos_k + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos_k - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos_k)
    
    base_x0 = keich[i_flat]
    base_y0 = keich[j_flat]
    x0 = np.concatenate([base_x0, base_y0])
    
    # Bounds for each parameter
    bounds = [(-2.0, 2.0)] * 128
    
    # Phase 1: Differential evolution for global search
    n_quad_de = 24
    def obj_de(params):
        return _union_volume(params[:64], params[64:], i_flat, j_flat, N, n_quad_de)
    
    best_params, best_vol = _differential_evolution_custom(
        obj_de, x0, bounds, maxiter=50, popsize=10,
        mutation=(0.5, 1.0), recombination=0.7, tol=1e-8, deadline=deadline
    )
    
    # Phase 2: L-BFGS-B refinement from best DE solution
    if time.time() < deadline - 30:
        n_quad_lbfgs = 48
        def obj_lbfgs(params):
            return _union_volume(params[:64], params[64:], i_flat, j_flat, N, n_quad_lbfgs)
        
        try:
            result = minimize(
                obj_lbfgs,
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.fun < best_vol:
                best_params = result.x.copy()
                best_vol = result.fun
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 15:
        n_quad_final = 56
        def obj_final(params):
            return _union_volume(params[:64], params[64:], i_flat, j_flat, N, n_quad_final)
        
        try:
            result_final = minimize(
                obj_final,
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if result_final.fun < best_vol:
                best_params = result_final.x.copy()
                best_vol = result_final.fun
        except Exception:
            pass
    
    # Extract best positions
    base_x = best_params[:64]
    base_y = best_params[64:]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions


# EVOLVE-BLOCK-END

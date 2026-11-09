# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_area_fast(x, n=64, M=20000):
    """
    Fast quadrature-based area computation (matches the evaluator exactly).
    """
    m = np.arange(M)
    y = (m + 0.5) / M
    a = (np.arange(n) + 1) / n
    w = (1.0 - y) / n
    
    L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    L_sorted = np.sort(L, axis=0)
    d = L_sorted[1:, :] - L_sorted[:-1, :]
    min_d_w = np.minimum(d, w[np.newaxis, :])
    union_lengths = w[np.newaxis, :] + np.sum(min_d_w, axis=0)
    area = np.mean(union_lengths)
    return float(area)

def _soft_min_area(x, n=64, M=20000, beta=50.0):
    """
    Smooth surrogate for the area using soft-min.
    min(a,b) ≈ -log(exp(-beta*a) + exp(-beta*b)) / beta
    """
    m = np.arange(M)
    y = (m + 0.5) / M
    a = (np.arange(n) + 1) / n
    w = (1.0 - y) / n
    
    L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    L_sorted = np.sort(L, axis=0)
    d = L_sorted[1:, :] - L_sorted[:-1, :]
    
    beta_d = beta * d
    beta_w = beta * w[np.newaxis, :]
    mx = np.maximum(beta_d, beta_w)
    sm = -mx / beta - np.log(np.exp(-beta_d - mx) + np.exp(-beta_w - mx)) / beta
    
    union_lengths = w[np.newaxis, :] + np.sum(sm, axis=0)
    area = np.mean(union_lengths)
    return float(area)

def _soft_min_area_grad(x, n=64, M=20000, beta=50.0):
    """
    Compute soft-min area and its gradient using np.bincount for speed.
    """
    m = np.arange(M)
    y = (m + 0.5) / M
    a = (np.arange(n) + 1) / n
    w = (1.0 - y) / n
    
    L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    order = np.argsort(L, axis=0)
    L_sorted = L[np.arange(n)[:, None], order]
    d = L_sorted[1:, :] - L_sorted[:-1, :]
    
    beta_d = beta * d
    beta_w = beta * w[np.newaxis, :]
    mx = np.maximum(beta_d, beta_w)
    
    exp_neg_d = np.exp(-beta_d - mx)
    exp_neg_w = np.exp(-beta_w - mx)
    sm = -mx / beta - np.log(exp_neg_d + exp_neg_w) / beta
    
    union_lengths = w[np.newaxis, :] + np.sum(sm, axis=0)
    area = float(np.mean(union_lengths))
    
    # sigmoid(beta*(w - d))
    arg = beta * (w[np.newaxis, :] - d)
    sig = 1.0 / (1.0 + np.exp(-np.clip(arg, -500, 500)))
    
    da = a[order[1:, :]] - a[order[:-1, :]]
    contrib = sig * da  # shape (n-1, M)
    
    # Use np.bincount for fast gradient accumulation
    # upper indices: order[j+1, m] for j in 0..n-2, m in 0..M-1
    # lower indices: order[j, m]
    upper_flat = order[1:, :].ravel()
    lower_flat = order[:-1, :].ravel()
    contrib_flat = contrib.ravel()
    
    grad = np.bincount(upper_flat, weights=contrib_flat, minlength=n) - \
           np.bincount(lower_flat, weights=contrib_flat, minlength=n)
    
    grad /= M
    return area, grad

def _subgrad_area_grad(x, n=64, M=20000):
    """
    Compute area and subgradient using np.bincount for speed.
    """
    m = np.arange(M)
    y = (m + 0.5) / M
    a = (np.arange(n) + 1) / n
    w = (1.0 - y) / n
    
    L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    order = np.argsort(L, axis=0)
    L_sorted = L[np.arange(n)[:, None], order]
    d = L_sorted[1:, :] - L_sorted[:-1, :]
    min_d_w = np.minimum(d, w[np.newaxis, :])
    union_lengths = w[np.newaxis, :] + np.sum(min_d_w, axis=0)
    area = float(np.mean(union_lengths))
    
    sat = (d < w[np.newaxis, :])  # shape (n-1, M)
    
    if sat.any():
        sat_upper = order[1:, :][sat].ravel()
        sat_lower = order[:-1, :][sat].ravel()
        grad = np.bincount(sat_upper, minlength=n).astype(np.float64) - \
               np.bincount(sat_lower, minlength=n).astype(np.float64)
    else:
        grad = np.zeros(n, dtype=np.float64)
    
    grad /= M
    return area, grad

def _keich_init(n=64):
    """
    Analytical Keich-style initialization: place triangles to maximize overlap.
    Uses the structure that a_i = (i+1)/n creates a natural ordering.
    """
    a = (np.arange(n) + 1) / n
    # Keich construction: x_i = -a_i * c for some constant c
    # Optimal c balances the spread; try c = 1/(2*n) based on width considerations
    # Actually, the key insight is that L_i(y) = x_i + a_i*y should be as close together as possible
    # At y=0: L_i = x_i, at y=1: L_i = x_i + a_i
    # To minimize union, we want L_i(0) and L_i(1) to overlap maximally
    # Set x_i = -a_i * t, then L_i(y) = a_i*(y-t)
    # All L_i pass through (t, 0) in the (y, L) plane
    # Optimal t is where the spread is minimized
    # Spread at y: max_i a_i*(y-t) - min_i a_i*(y-t) = (a_n - a_1)*|y-t|
    # But we also have width w(y) = (1-y)/n
    # The union is minimized when the triangles are maximally overlapping
    # Try several values of t
    best_x = None
    best_area = float('inf')
    for t in [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
        x = -a * t
        area = _compute_area_fast(x, n, 2000)  # quick check with fewer samples
        if area < best_area:
            best_area = area
            best_x = x.copy()
    return best_x

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem.
    Returns:
        positions: np.ndarray of shape (64,) with the base x-coordinates x_i.
    """
    n = 64
    M = 20000
    start_time = time.time()
    deadline = start_time + 600.0
    
    a = (np.arange(n) + 1) / n
    
    # Initial guesses including Keich-style
    inits = [
        np.zeros(n),
        -a * 0.5,
        -a * (1.0/3.0),
        -a * (2.0/3.0),
        a * 0.5,
        -a * 0.25,
        -a * 0.75,
        _keich_init(n),
    ]
    
    best_x = inits[0].copy()
    best_area = _compute_area_fast(inits[0], n, M)
    
    for init in inits[1:]:
        area_init = _compute_area_fast(init, n, M)
        if area_init < best_area:
            best_area = area_init
            best_x = init.copy()
    
    # Phase 1: Smooth gradient-based optimization using soft-min surrogate
    # Use faster np.bincount-based gradient
    if time.time() < deadline - 100:
        try:
            from scipy.optimize import minimize
            
            # Try multiple beta values, starting coarse and going fine
            for beta_val in [50.0, 100.0, 200.0, 400.0, 800.0]:
                if time.time() > deadline - 100:
                    break
                
                def obj_b(x, bv=beta_val):
                    return _soft_min_area(x, n, M, beta=bv)
                
                def grad_b(x, bv=beta_val):
                    _, g = _soft_min_area_grad(x, n, M, beta=bv)
                    return g
                
                t_max = min(deadline - 100, start_time + 60)
                res = minimize(obj_b, best_x, jac=grad_b, method='L-BFGS-B',
                             options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12,
                                     'maxls': 50, 'maxfun': 5000})
                
                if res.fun < 1e10:
                    area_exact = _compute_area_fast(res.x, n, M)
                    if area_exact < best_area:
                        best_area = area_exact
                        best_x = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Multiple random restarts with smooth gradient polishing
    if time.time() < deadline - 60:
        rng = np.random.default_rng(12345)
        n_restarts = 30
        for r in range(n_restarts):
            if time.time() > deadline - 60:
                break
            
            # Scale perturbation based on how far we are from deadline
            remaining_frac = (deadline - 60 - time.time()) / max(1e-6, deadline - 60 - start_time)
            pert_scale = 0.3 * max(0.1, remaining_frac)
            
            x_pert = best_x + rng.normal(0, pert_scale, n)
            
            try:
                def obj_b(x, bv=200.0):
                    return _soft_min_area(x, n, M, beta=bv)
                def grad_b(x, bv=200.0):
                    _, g = _soft_min_area_grad(x, n, M, beta=bv)
                    return g
                
                res = minimize(obj_b, x_pert, jac=grad_b, method='L-BFGS-B',
                             options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-11,
                                     'maxls': 40, 'maxfun': 1500})
                
                if res.fun < 1e10:
                    area_exact = _compute_area_fast(res.x, n, M)
                    if area_exact < best_area:
                        best_area = area_exact
                        best_x = res.x.copy()
            except Exception:
                pass
    
    # Phase 3: Subgradient descent on exact objective for local refinement
    if time.time() < deadline - 40:
        x = best_x.copy()
        area, g = _subgrad_area_grad(x, n, M)
        v = np.zeros(n, dtype=np.float64)
        mu = 0.9
        step = 0.05
        best_local = area
        best_local_x = x.copy()
        
        t_max = min(deadline - 40, start_time + 50)
        while time.time() < t_max:
            gnorm = np.linalg.norm(g)
            if gnorm < 1e-14:
                break
            step_polyak = max(np.dot(x, g), 1e-12) / (gnorm * gnorm + 1e-12)
            step = min(step_polyak, 0.3)
            v = mu * v - step * g
            xn = x + v
            an, gn = _subgrad_area_grad(xn, n, M)
            if an < area:
                x, area, g = xn, an, gn
                if an < best_local:
                    best_local = an
                    best_local_x = x.copy()
            else:
                step *= 0.7
                v *= 0.5
                if step < 1e-12:
                    break
        
        if best_local < best_area:
            best_area = best_local
            best_x = best_local_x.copy()
    
    # Phase 4: Coordinate descent with adaptive step for final polish
    x = best_x.copy()
    current_area = best_area
    
    for iteration in range(30):
        if time.time() > deadline - 20:
            break
        
        improved = False
        for i in range(n):
            if time.time() > deadline - 20:
                break
            
            best_local = current_area
            best_delta = 0.0
            for delta in [-0.02, -0.01, -0.005, -0.002, 0.002, 0.005, 0.01, 0.02]:
                x_trial = x.copy()
                x_trial[i] += delta
                area_trial = _compute_area_fast(x_trial, n, M)
                if area_trial < best_local:
                    best_local = area_trial
                    best_delta = delta
            
            if best_delta != 0.0:
                x[i] += best_delta
                current_area = best_local
                improved = True
        
        if not improved:
            improved_small = False
            for i in range(n):
                if time.time() > deadline - 20:
                    break
                for delta in [-0.001, -0.0005, 0.0005, 0.001]:
                    x_trial = x.copy()
                    x_trial[i] += delta
                    area_trial = _compute_area_fast(x_trial, n, M)
                    if area_trial < current_area - 1e-10:
                        x[i] += delta
                        current_area = area_trial
                        improved_small = True
                        break
            if not improved_small:
                break
        
        if current_area < best_area:
            best_area = current_area
            best_x = x.copy()
    
    # Phase 5: Final high-precision L-BFGS-B polish with very fine beta
    if time.time() < deadline - 15:
        try:
            from scipy.optimize import minimize
            def obj_fine(x):
                return _soft_min_area(x, n, M, beta=2000.0)
            def grad_fine(x):
                _, g = _soft_min_area_grad(x, n, M, beta=2000.0)
                return g
            res = minimize(obj_fine, best_x, jac=grad_fine, method='L-BFGS-B',
                          options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14,
                                  'maxls': 50, 'maxfun': 3000})
            if res.fun < 1e10:
                area_exact = _compute_area_fast(res.x, n, M)
                if area_exact < best_area:
                    best_area = area_exact
                    best_x = res.x.copy()
        except Exception:
            pass
    
    # Phase 6: Final Nelder-Mead polish
    if time.time() < deadline - 10:
        try:
            from scipy.optimize import minimize
            def objective(x):
                return _compute_area_fast(x, n, M)
            res = minimize(objective, best_x, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-13, 'fatol': 1e-15})
            if res.fun < best_area:
                best_x = res.x.copy()
                best_area = res.fun
        except Exception:
            pass
    
    # Validation
    if not np.all(np.isfinite(best_x)):
        return np.zeros(n, dtype=np.float64)
    if best_area <= 0:
        return np.zeros(n, dtype=np.float64)
        
    return best_x.astype(np.float64)

# EVOLVE-BLOCK-END

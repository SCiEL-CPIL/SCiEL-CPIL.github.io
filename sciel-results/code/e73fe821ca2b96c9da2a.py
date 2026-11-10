# EVOLVE-BLOCK-START
import numpy as np
import time


def _union_length_vec(starts, widths):
    """
    Compute total length of union of 1-D intervals [starts[i], starts[i]+widths[i]].
    Returns the union length.
    """
    n = len(starts)
    if n == 0:
        return 0.0
    
    order = np.argsort(starts)
    s_sorted = starts[order]
    e_sorted = s_sorted + widths[order]
    
    total_len = 0.0
    cs = s_sorted[0]
    ce = e_sorted[0]
    
    for i in range(1, n):
        if s_sorted[i] > ce:
            total_len += ce - cs
            cs = s_sorted[i]
            ce = e_sorted[i]
        else:
            ce = max(ce, e_sorted[i])
    
    total_len += ce - cs
    return total_len


def _exact_volume(pos_x, pos_y, N=8, c_samples=200):
    """Exact union volume via product structure: integral of Lx(c)*Ly(c) dc."""
    c_vals = np.linspace(0, 1, c_samples)
    vol = 0.0
    invN = 1.0 / N
    dx = 1.0 / c_samples
    
    for c in c_vals:
        # Compute x intervals
        x_starts = pos_x + c * np.arange(N) * invN
        x_widths = np.full(N, invN)
        
        # Compute y intervals
        y_starts = pos_y + c * np.arange(N) * invN
        y_widths = np.full(N, invN)
        
        # Union length in x
        x_order = np.argsort(x_starts)
        x_s = x_starts[x_order]
        x_e = x_s + x_widths[x_order]
        
        lx = 0.0
        if len(x_s) > 0:
            cs = x_s[0]
            ce = x_e[0]
            for i in range(1, len(x_s)):
                if x_s[i] > ce:
                    lx += ce - cs
                    cs = x_s[i]
                    ce = x_e[i]
                else:
                    ce = max(ce, x_e[i])
            lx += ce - cs
        
        # Union length in y
        y_order = np.argsort(y_starts)
        y_s = y_starts[y_order]
        y_e = y_s + y_widths[y_order]
        
        ly = 0.0
        if len(y_s) > 0:
            cs = y_s[0]
            ce = y_e[0]
            for i in range(1, len(y_s)):
                if y_s[i] > ce:
                    ly += ce - cs
                    cs = y_s[i]
                    ce = y_e[i]
                else:
                    ce = max(ce, y_e[i])
            ly += ce - cs
        
        vol += lx * ly
    
    return vol * dx


def _union_length_grad(pos, N=8, c_samples=200):
    """
    Compute gradient of the 1D union length integral w.r.t. pos.
    Uses central differences.
    """
    grad = np.zeros_like(pos)
    h = 1e-7
    for i in range(len(pos)):
        pos_plus = pos.copy()
        pos_plus[i] += h
        vol_plus = _exact_volume(pos_plus, pos_plus, N, c_samples=c_samples)
        
        pos_minus = pos.copy()
        pos_minus[i] -= h
        vol_minus = _exact_volume(pos_minus, pos_minus, N, c_samples=c_samples)
        
        grad[i] = (vol_plus - vol_minus) / (2 * h)
    return grad


def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    deadline = time.time() + 570.0  # Safety margin

    # Keich binary construction
    pos_k = int(np.log2(N))
    j_terms = np.arange(1, pos_k + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos_k - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos_k)

    best_x = keich.copy()
    best_y = keich.copy()
    best_vol = _exact_volume(best_x, best_y, N, c_samples=200)

    # Physics-inspired annealing
    rng = np.random.default_rng(12345)
    
    # Start with current best
    x_curr = best_x.copy()
    y_curr = best_y.copy()
    
    # Temperature schedule
    T0 = 0.1
    T_end = 0.001
    n_steps = 2000
    
    for step in range(n_steps):
        if time.time() > deadline - 60:
            break
        
        # Temperature decay
        T = T0 * (T_end / T0) ** (step / n_steps)
        
        # Compute gradient of volume w.r.t. x and y separately
        # Since volume = integral Lx(c)*Ly(c) dc, we can compute gradients
        # by perturbing x or y separately
        
        # Gradient w.r.t. x
        h = 1e-7
        grad_x = np.zeros(N)
        for i in range(N):
            x_plus = x_curr.copy()
            x_plus[i] += h
            vp = _exact_volume(x_plus, y_curr, N, c_samples=100)
            
            x_minus = x_curr.copy()
            x_minus[i] -= h
            vm = _exact_volume(x_minus, y_curr, N, c_samples=100)
            
            grad_x[i] = (vp - vm) / (2 * h)
        
        # Gradient w.r.t. y
        grad_y = np.zeros(N)
        for i in range(N):
            y_plus = y_curr.copy()
            y_plus[i] += h
            vp = _exact_volume(x_curr, y_plus, N, c_samples=100)
            
            y_minus = y_curr.copy()
            y_minus[i] -= h
            vm = _exact_volume(x_curr, y_minus, N, c_samples=100)
            
            grad_y[i] = (vp - vm) / (2 * h)
        
        # Proposed move: gradient descent + random perturbation
        step_size = 0.01
        x_prop = x_curr - step_size * grad_x + rng.normal(0, T, N)
        y_prop = y_curr - step_size * grad_y + rng.normal(0, T, N)
        
        # Evaluate new volume
        vol_prop = _exact_volume(x_prop, y_prop, N, c_samples=100)
        vol_curr = _exact_volume(x_curr, y_curr, N, c_samples=100)
        
        # Metropolis acceptance
        if vol_prop < vol_curr:
            x_curr = x_prop
            y_curr = y_prop
            if vol_prop < best_vol:
                best_vol = vol_prop
                best_x = x_prop.copy()
                best_y = y_prop.copy()
        else:
            # Accept with probability exp(-(vol_prop - vol_curr)/T)
            delta = vol_prop - vol_curr
            if delta < 1e-10 or np.log(rng.random()) < -delta / max(T, 1e-15):
                x_curr = x_prop
                y_curr = y_prop
    
    # Final polish with Nelder-Mead on the best found
    def objective(params):
        if time.time() > deadline - 30:
            return best_vol
        return _exact_volume(params[:N], params[N:], N, c_samples=200)
    
    from scipy.optimize import minimize
    options = {'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-11}
    
    init = np.concatenate([best_x, best_y])
    result = minimize(objective, init, method='Nelder-Mead', options=options)
    if result.fun < best_vol:
        best_vol = result.fun
        best_x = result.x[:N].copy()
        best_y = result.x[N:].copy()
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    base_x = best_x[i_flat]
    base_y = best_y[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    return positions


# EVOLVE-BLOCK-END

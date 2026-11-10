# EVOLVE-BLOCK-START
import numpy as np
import time


def _union_lengths_vec(starts_2d, width):
    """
    Compute union lengths for multiple 1D interval sets simultaneously.
    starts_2d: shape (M, N) - M sets of N intervals
    width: scalar width for all intervals
    Returns: shape (M,) union lengths
    """
    M, N = starts_2d.shape
    order = np.argsort(starts_2d, axis=1)
    s_sorted = np.take_along_axis(starts_2d, order, axis=1)
    e_sorted = s_sorted + width
    
    lengths = np.zeros(M)
    for j in range(N):
        if j == 0:
            lengths += width
            max_e = e_sorted[:, 0].copy()
        else:
            contrib = np.maximum(0.0, e_sorted[:, j] - max_e)
            lengths += contrib
            max_e = np.maximum(max_e, e_sorted[:, j])
    
    return lengths


def _exact_volume_fast(pos_x, pos_y, N=8, c_samples=400):
    """Exact union volume via product structure: integral of Lx(c)*Ly(c) dc."""
    c_vals = np.linspace(0, 1, c_samples)
    invN = 1.0 / N
    dx = 1.0 / c_samples
    
    c_col = c_vals.reshape(-1, 1)
    i_row = np.arange(N).reshape(1, -1) * invN
    
    x_starts = pos_x.reshape(1, -1) + c_col * i_row
    y_starts = pos_y.reshape(1, -1) + c_col * i_row
    
    lx = _union_lengths_vec(x_starts, invN)
    ly = _union_lengths_vec(y_starts, invN)
    
    vol = np.sum(lx * ly) * dx
    return vol


def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    deadline = time.time() + 570.0

    pos_k = int(np.log2(N))
    j_terms = np.arange(1, pos_k + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos_k - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos_k)

    best_x = keich.copy()
    best_y = keich.copy()
    best_vol = _exact_volume_fast(best_x, best_y, N, c_samples=400)

    rng = np.random.default_rng(12345)
    x_curr = best_x.copy()
    y_curr = best_y.copy()
    
    T0 = 0.05
    T_end = 0.001
    n_steps = 3000
    
    for step in range(n_steps):
        if time.time() > deadline - 90:
            break
        
        T = T0 * (T_end / T0) ** (step / n_steps)
        
        h = 1e-5
        c_grad = 200
        
        grad_x = np.zeros(N)
        for i in range(N):
            x_plus = x_curr.copy()
            x_plus[i] += h
            vp = _exact_volume_fast(x_plus, y_curr, N, c_samples=c_grad)
            
            x_minus = x_curr.copy()
            x_minus[i] -= h
            vm = _exact_volume_fast(x_minus, y_curr, N, c_samples=c_grad)
            
            grad_x[i] = (vp - vm) / (2 * h)
        
        grad_y = np.zeros(N)
        for i in range(N):
            y_plus = y_curr.copy()
            y_plus[i] += h
            vp = _exact_volume_fast(x_curr, y_plus, N, c_samples=c_grad)
            
            y_minus = y_curr.copy()
            y_minus[i] -= h
            vm = _exact_volume_fast(x_curr, y_minus, N, c_samples=c_grad)
            
            grad_y[i] = (vp - vm) / (2 * h)
        
        step_size = 0.005
        x_prop = x_curr - step_size * grad_x + rng.normal(0, T, N)
        y_prop = y_curr - step_size * grad_y + rng.normal(0, T, N)
        
        vol_prop = _exact_volume_fast(x_prop, y_prop, N, c_samples=300)
        vol_curr = _exact_volume_fast(x_curr, y_curr, N, c_samples=300)
        
        if vol_prop < vol_curr:
            x_curr = x_prop
            y_curr = y_prop
            if vol_prop < best_vol:
                best_vol = vol_prop
                best_x = x_prop.copy()
                best_y = y_prop.copy()
        else:
            delta = vol_prop - vol_curr
            if delta < 1e-10 or np.log(rng.random()) < -delta / max(T, 1e-15):
                x_curr = x_prop
                y_curr = y_prop
    
    def objective(params):
        if time.time() > deadline - 30:
            return best_vol
        return _exact_volume_fast(params[:N], params[N:], N, c_samples=400)
    
    from scipy.optimize import minimize
    options = {'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
    
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

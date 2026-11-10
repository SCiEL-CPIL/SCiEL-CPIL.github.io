# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

DEADLINE = 630.0
START_TIME = time.time()


def _elapsed():
    return time.time() - START_TIME


def _time_left():
    return DEADLINE - _elapsed() - 30.0


def _union_length_1d(x, N, n_c=200):
    """
    Compute the 1D union length integral over c in [0,1].
    x: array of N base positions.
    At height c, interval i is [x_i + c*i/N, x_i + c*i/N + 1/N].
    Returns the integral of the union length over c.
    """
    c_vals = np.linspace(0, 1, n_c)
    arange_N = np.arange(N, dtype=np.float64)
    widths = np.ones(N) / N
    
    # For each c, compute starts and ends
    # starts[i] = x[i] + c * i / N
    # ends[i] = starts[i] + 1/N
    
    lengths = np.empty(n_c)
    for idx, c in enumerate(c_vals):
        starts = x + c * arange_N / N
        ends = starts + 1.0 / N
        
        # Merge intervals
        order = np.argsort(starts)
        s = starts[order]
        e = ends[order]
        
        total = 0.0
        cur_s = s[0]
        cur_e = e[0]
        for k in range(1, N):
            if s[k] > cur_e:
                total += cur_e - cur_s
                cur_s = s[k]
                cur_e = e[k]
            else:
                if e[k] > cur_e:
                    cur_e = e[k]
        total += cur_e - cur_s
        lengths[idx] = total
    
    # Trapezoidal integration
    return np.trapz(lengths, c_vals)


def _volume_from_bases(x_bases, y_bases, N, n_c=200):
    """
    Compute the 3D union volume given 1D base positions.
    If x and y are independent (x depends only on i, y only on j),
    then V = ∫ L_x(c) * L_y(c) dc.
    """
    c_vals = np.linspace(0, 1, n_c)
    arange_N = np.arange(N, dtype=np.float64)
    
    Lx = np.empty(n_c)
    Ly = np.empty(n_c)
    
    for idx, c in enumerate(c_vals):
        # X-direction
        starts_x = x_bases + c * arange_N / N
        ends_x = starts_x + 1.0 / N
        order_x = np.argsort(starts_x)
        sx = starts_x[order_x]
        ex = ends_x[order_x]
        total_x = 0.0
        cur_s = sx[0]
        cur_e = ex[0]
        for k in range(1, N):
            if sx[k] > cur_e:
                total_x += cur_e - cur_s
                cur_s = sx[k]
                cur_e = ex[k]
            else:
                if ex[k] > cur_e:
                    cur_e = ex[k]
        total_x += cur_e - cur_s
        Lx[idx] = total_x
        
        # Y-direction
        starts_y = y_bases + c * arange_N / N
        ends_y = starts_y + 1.0 / N
        order_y = np.argsort(starts_y)
        sy = starts_y[order_y]
        ey = ends_y[order_y]
        total_y = 0.0
        cur_s = sy[0]
        cur_e = ey[0]
        for k in range(1, N):
            if sy[k] > cur_e:
                total_y += cur_e - cur_s
                cur_s = sy[k]
                cur_e = ey[k]
            else:
                if ey[k] > cur_e:
                    cur_e = ey[k]
        total_y += cur_e - cur_s
        Ly[idx] = total_y
    
    return np.trapz(Lx * Ly, c_vals)


def _objective_couple(params, N, n_c=100):
    """
    Coupled objective: minimize V = ∫ Lx(c) * Ly(c) dc
    params: 2N parameters, first N are x bases, next N are y bases.
    """
    x_bases = params[:N]
    y_bases = params[N:]
    return _volume_from_bases(x_bases, y_bases, N, n_c)


def _objective_x_only(x, N, n_c=100):
    """Objective for x only (assuming y is fixed to optimal)."""
    return _union_length_1d(x, N, n_c)


def _keich_offsets(N):
    """Generate Keich binary construction offsets."""
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        bits = list(str(bin(idx))[2:])
        bits = np.array(bits, dtype=int)
        bits = np.pad(bits, (pos - len(bits), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * bits * (2.0 ** -j_terms) / pos)
    return keich


def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    keich = _keich_offsets(N)
    
    # Initial guess: use Keich for both x and y
    x0 = keich.copy()
    y0 = keich.copy()
    
    # Phase 1: Optimize x and y independently (fast, gives good starting point)
    best_x = x0.copy()
    best_y = y0.copy()
    
    # Try a few restarts for x
    rng = np.random.default_rng(42)
    best_cost_x = _objective_x_only(x0, N, n_c=100)
    
    n_restarts_x = 3
    for r in range(n_restarts_x):
        if _time_left() < 120:
            break
        if r == 0:
            x_init = x0.copy()
        else:
            x_init = x0 + rng.normal(0, 0.15, size=N)
        
        try:
            res = minimize(_objective_x_only, x_init, args=(N,),
                          method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_cost_x:
                best_cost_x = res.fun
                best_x = res.x.copy()
        except Exception:
            continue
    
    # Try a few restarts for y
    best_cost_y = _objective_x_only(y0, N, n_c=100)
    for r in range(n_restarts_x):
        if _time_left() < 120:
            break
        if r == 0:
            y_init = y0.copy()
        else:
            y_init = y0 + rng.normal(0, 0.15, size=N)
        
        try:
            res = minimize(_objective_x_only, y_init, args=(N,),
                          method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_cost_y:
                best_cost_y = res.fun
                best_y = res.x.copy()
        except Exception:
            continue
    
    # Phase 2: Coupled optimization of (x, y) jointly
    params0 = np.concatenate([best_x, best_y])
    best_params = params0.copy()
    best_vol = _objective_couple(params0, N, n_c=100)
    
    # Nelder-Mead on 16D coupled problem
    n_coupled_restarts = 3
    for r in range(n_coupled_restarts):
        if _time_left() < 90:
            break
        if r == 0:
            p_init = params0.copy()
        else:
            p_init = params0 + rng.normal(0, 0.1, size=2 * N)
        
        try:
            res = minimize(_objective_couple, p_init, args=(N,),
                          method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-9})
            if res.fun < best_vol:
                best_vol = res.fun
                best_params = res.x.copy()
        except Exception:
            continue
    
    # Extract final x and y bases
    final_x = best_params[:N]
    final_y = best_params[N:]
    
    # Build 64x2 position array
    i_vals = np.repeat(np.arange(N), N)
    j_vals = np.tile(np.arange(N), N)
    
    base_x = final_x[i_vals]
    base_y = final_y[j_vals]
    positions = np.stack([base_x, base_y], axis=1)
    
    # Verify validity with Monte Carlo
    x_base = positions[:, 0]
    y_base = positions[:, 1]
    i_f = (np.arange(N * N) // N).astype(np.float64)
    j_f = (np.arange(N * N) % N).astype(np.float64)
    
    x_extents = x_base + 1.0 / N + i_f / N
    X0 = np.min(x_base)
    X1 = np.max(x_extents)
    y_extents = y_base + 1.0 / N + j_f / N
    Y0 = np.min(y_base)
    Y1 = np.max(y_extents)
    
    if X1 - X0 > 1e-9 and Y1 - Y0 > 1e-9:
        rng_mc = np.random.default_rng(12345)
        n_mc = 500000
        a = rng_mc.uniform(X0, X1, n_mc)
        b = rng_mc.uniform(Y0, Y1, n_mc)
        c = rng_mc.uniform(0, 1, n_mc)
        
        chunk_size = 50000
        inside_any = np.zeros(n_mc, dtype=bool)
        for start in range(0, n_mc, chunk_size):
            end = min(start + chunk_size, n_mc)
            a_c = a[start:end]
            b_c = b[start:end]
            c_c = c[start:end]
            
            x_low = x_base + c_c[:, None] * i_f[None, :] / N
            x_high = x_low + 1.0 / N
            y_low = y_base + c_c[:, None] * j_f[None, :] / N
            y_high = y_low + 1.0 / N
            
            in_x = (a_c[:, None] >= x_low) & (a_c[:, None] <= x_high)
            in_y = (b_c[:, None] >= y_low) & (b_c[:, None] <= y_high)
            inside_any[start:end] = np.any(in_x & in_y, axis=1)
        
        fraction = np.mean(inside_any)
        V = fraction * (X1 - X0) * (Y1 - Y0)
        
        if not np.isfinite(V) or V <= 0:
            # Fallback
            base_x = keich[i_vals]
            base_y = keich[j_vals]
            positions = np.stack([base_x, base_y], axis=1)
    
    return positions


# EVOLVE-BLOCK-END

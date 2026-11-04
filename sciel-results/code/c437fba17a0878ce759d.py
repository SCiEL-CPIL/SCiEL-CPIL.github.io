import numpy as np
import time
import scipy.optimize as opt
from scipy.optimize import differential_evolution
from scipy.interpolate import CubicSpline

N = 600
W = 1.0 / 1200.0
X = np.linspace(-0.25, 0.25, N, endpoint=False)

# Precompute fine grid
FINE_N = 12000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

def ratio_from_h(h):
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return float('inf')
    f = h[CELL]
    F = np.fft.rfft(f, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
    integral_sq = (s * W) ** 2
    if integral_sq < 1e-15:
        return float('inf')
    return float(np.max(conv) / integral_sq)

def obj_smooth(h):
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return 1e10
    integral_sq = (s * W) ** 2
    f_fine = h[CELL]
    F = np.fft.rfft(f_fine, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
    m = np.max(conv)
    if m < 1e-15:
        return 1e10
    beta = 500.0
    soft_max = m + np.log(np.sum(np.exp(conv - m))) / beta
    return soft_max / integral_sq

def spline_to_h(params, n_points=12):
    # params: n_points values
    x_pts = np.linspace(-0.25, 0.25, n_points)
    h_pts = np.abs(params) # ensure non-negative control points
    try:
        cs = CubicSpline(x_pts, h_pts)
        h = cs(X)
    except Exception:
        h = np.ones(N)
    return np.clip(h, 0, None)

def objective_spline(params):
    h = spline_to_h(params)
    return obj_smooth(h)

def gaussians_to_h(params):
    n_gauss = (len(params) - 1) // 3
    base = params[-1]
    h = np.full(N, base, dtype=np.float64)
    for i in range(n_gauss):
        mu = params[3*i]
        sigma = abs(params[3*i + 1]) + 1e-6
        amp = params[3*i + 2]
        h += amp * np.exp(-0.5 * ((X - mu) / sigma) ** 2)
    return np.clip(h, 0, None)

def objective_gauss(params):
    h = gaussians_to_h(params)
    return obj_smooth(h)

def blocks_to_h(params, n_blocks):
    # params: n_blocks values
    # Map 600 intervals to n_blocks blocks
    # Each block covers N // n_blocks intervals (approx)
    # We create a vector of block indices for each of the N intervals
    if n_blocks <= 0:
        return np.ones(N)
    
    # Create indices for each interval
    # Interval i is in block floor(i * n_blocks / N)
    indices = np.floor(np.arange(N) * n_blocks / N).astype(np.int64)
    indices = np.clip(indices, 0, n_blocks - 1)
    
    h = np.zeros(N)
    h[indices] = params
    return np.clip(h, 0, None)

def objective_block(params, n_blocks):
    h = blocks_to_h(params, n_blocks)
    return obj_smooth(h)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_h = np.ones(N)
    best_val = ratio_from_h(best_h)
    
    def update_best(h, val):
        nonlocal best_h, best_val
        if val < best_val and np.isfinite(val):
            best_val = val
            best_h = h.copy()
            
    rng = np.random.default_rng(42)
    
    # Phase 1: Global Search with Splines
    if time.time() < deadline - 60.0:
        n_pts = 12
        bounds = [(0.0, 5.0)] * n_pts
        try:
            res = differential_evolution(
                objective_spline, bounds, maxiter=100, popsize=20,
                tol=1e-6, seed=42, polish=True, init='sobol',
                maxfevals=3000
            )
            h_opt = spline_to_h(res.x, n_pts)
            val = ratio_from_h(h_opt)
            update_best(h_opt, val)
        except Exception:
            pass
            
    # Phase 2: Global Search with Gaussians (more components)
    if time.time() < deadline - 40.0:
        for n_gauss in [5, 7]:
            if time.time() > deadline - 30.0:
                break
            n_params = 3 * n_gauss + 1
            bounds = []
            for _ in range(n_gauss):
                bounds.append((-0.25, 0.25))
                bounds.append((0.01, 0.15))
                bounds.append((0.0, 5.0))
            bounds.append((0.0, 2.0))
            
            try:
                res = differential_evolution(
                    objective_gauss, bounds, maxiter=80, popsize=20,
                    tol=1e-6, seed=42, polish=True, init='sobol',
                    maxfevals=2000
                )
                h_opt = gaussians_to_h(res.x)
                val = ratio_from_h(h_opt)
                update_best(h_opt, val)
            except Exception:
                pass
                
    # Phase 2.5: Global Search with Blocks
    if time.time() < deadline - 30.0:
        for n_blocks in [10, 20, 50]:
            if time.time() > deadline - 20.0:
                break
            bounds = [(0.0, 5.0)] * n_blocks
            try:
                res = differential_evolution(
                    lambda p: objective_block(p, n_blocks), bounds, maxiter=100, popsize=20,
                    tol=1e-6, seed=42, polish=True, init='sobol',
                    maxfevals=2000
                )
                h_opt = blocks_to_h(res.x, n_blocks)
                val = ratio_from_h(h_opt)
                update_best(h_opt, val)
            except Exception:
                pass

    # Phase 3: Local Refinement on full 600-dim space
    # Start from best global solution and some random starts
    starts = [best_h]
    for _ in range(5):
        # Random smooth start
        h_rand = np.abs(rng.normal(1.0, 0.5, N))
        # Smooth it
        kernel = np.ones(10) / 10
        h_rand = np.convolve(h_rand, kernel, mode='same')
        starts.append(h_rand)
        
    for h_start in starts:
        if time.time() > deadline - 10.0:
            break
        try:
            res = opt.minimize(
                obj_smooth, h_start, method='L-BFGS-B',
                bounds=[(0.0, None)] * N,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            h_opt = np.clip(res.x, 0, None)
            val = ratio_from_h(h_opt)
            update_best(h_opt, val)
        except Exception:
            pass
            
    # Phase 4: Final Polish
    if time.time() < deadline - 2.0:
        try:
            res = opt.minimize(
                obj_smooth, best_h, method='L-BFGS-B',
                bounds=[(0.0, None)] * N,
                options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 50}
            )
            h_opt = np.clip(res.x, 0, None)
            val = ratio_from_h(h_opt)
            update_best(h_opt, val)
        except Exception:
            pass
            
    return list(best_h)

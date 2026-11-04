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

def obj_smooth(h, beta=500.0):
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
    soft_max = m + np.log(np.sum(np.exp(conv - m))) / beta
    return soft_max / integral_sq

def spline_to_h(params, n_points=12):
    x_pts = np.linspace(-0.25, 0.25, n_points)
    h_pts = np.abs(params)
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

def double_box_to_h(params):
    """
    Double-box (hollow) structure: high amplitude at edges, low in middle.
    params: [inner_halfwidth, outer_halfwidth, inner_amp, outer_amp]
    - inner_halfwidth: half-width of the low-amplitude central region
    - outer_halfwidth: half-width of the full support (should be ~0.25)
    - inner_amp: amplitude in the central region
    - outer_amp: amplitude in the edge regions
    """
    inner_hw = abs(params[0]) + 1e-6
    outer_hw = abs(params[1]) + 1e-6
    inner_amp = max(params[2], 0.0)
    outer_amp = max(params[3], 0.0)
    
    # Ensure inner < outer
    if inner_hw >= outer_hw:
        inner_hw = outer_hw * 0.5
    
    h = np.zeros(N)
    # Central region: |x| <= inner_hw
    central_mask = np.abs(X) <= inner_hw
    # Edge regions: inner_hw < |x| <= outer_hw
    edge_mask = (np.abs(X) > inner_hw) & (np.abs(X) <= outer_hw)
    
    h[central_mask] = inner_amp
    h[edge_mask] = outer_amp
    
    return h

def objective_double_box(params):
    h = double_box_to_h(params)
    return obj_smooth(h)

def double_box_smooth_to_h(params):
    """
    Smooth double-box with soft transitions.
    params: [inner_halfwidth, outer_halfwidth, inner_amp, outer_amp, transition_width]
    """
    inner_hw = abs(params[0]) + 1e-6
    outer_hw = abs(params[1]) + 1e-6
    inner_amp = max(params[2], 0.0)
    outer_amp = max(params[3], 0.0)
    trans_w = abs(params[4]) + 1e-4
    
    if inner_hw >= outer_hw:
        inner_hw = outer_hw * 0.5
    
    ax = np.abs(X)
    # Smooth step from inner to outer
    # Transition from inner_hw to outer_hw
    t = (ax - inner_hw) / (outer_hw - inner_hw + 1e-10)
    # Smooth step function
    smooth_t = 1.0 / (1.0 + np.exp(-t / trans_w))
    smooth_t = np.clip(smooth_t, 0, 1)
    
    h = inner_amp + (outer_amp - inner_amp) * smooth_t
    # Zero outside outer_hw
    h[ax > outer_hw] = 0
    
    return np.clip(h, 0, None)

def objective_double_box_smooth(params):
    h = double_box_smooth_to_h(params)
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
    
    # Phase 1: Global Search with Double-Box (hollow structure)
    if time.time() < deadline - 50.0:
        # Try smooth double-box first (more flexible)
        bounds_smooth = [
            (0.0, 0.2),      # inner_halfwidth
            (0.15, 0.25),    # outer_halfwidth
            (0.0, 2.0),      # inner_amp
            (0.5, 5.0),      # outer_amp
            (0.01, 0.05)     # transition_width
        ]
        try:
            res = differential_evolution(
                objective_double_box_smooth, bounds_smooth,
                maxiter=150, popsize=25, tol=1e-6, seed=42,
                polish=True, init='sobol', maxfevals=4000
            )
            h_opt = double_box_smooth_to_h(res.x)
            val = ratio_from_h(h_opt)
            update_best(h_opt, val)
        except Exception:
            pass
        
        # Also try sharp double-box
        if time.time() < deadline - 30.0:
            bounds_sharp = [
                (0.0, 0.15),      # inner_halfwidth
                (0.15, 0.25),     # outer_halfwidth
                (0.0, 1.0),       # inner_amp
                (1.0, 5.0)        # outer_amp
            ]
            try:
                res = differential_evolution(
                    objective_double_box, bounds_sharp,
                    maxiter=100, popsize=20, tol=1e-6, seed=42,
                    polish=True, init='sobol', maxfevals=2000
                )
                h_opt = double_box_to_h(res.x)
                val = ratio_from_h(h_opt)
                update_best(h_opt, val)
            except Exception:
                pass
    
    # Phase 2: Global Search with Splines (for comparison)
    if time.time() < deadline - 30.0:
        n_pts = 12
        bounds = [(0.0, 5.0)] * n_pts
        try:
            res = differential_evolution(
                objective_spline, bounds, maxiter=80, popsize=20,
                tol=1e-6, seed=42, polish=True, init='sobol',
                maxfevals=2000
            )
            h_opt = spline_to_h(res.x, n_pts)
            val = ratio_from_h(h_opt)
            update_best(h_opt, val)
        except Exception:
            pass
            
    # Phase 3: Local Refinement on full 600-dim space
    starts = [best_h]
    for _ in range(3):
        h_rand = np.abs(rng.normal(1.0, 0.5, N))
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
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
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
                options={'maxiter': 150, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 50}
            )
            h_opt = np.clip(res.x, 0, None)
            val = ratio_from_h(h_opt)
            update_best(h_opt, val)
        except Exception:
            pass
            
    return list(best_h)

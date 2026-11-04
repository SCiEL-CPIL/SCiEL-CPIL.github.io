import time
import numpy as np
from scipy.optimize import minimize

N = 50
_START = time.time()
DEADLINE = 100.0  # safety margin before 150s kill

# fine grid for continuous step-function autocorrelation
M = 2048
dx_step = 0.5 / N  # 0.01
step_centers = -0.25 + (np.arange(N) + 0.5) * dx_step
# fine grid cell centers over [-1/4,1/4]
fine_dx = 0.5 / M
fine_x = -0.25 + (np.arange(M) + 0.5) * fine_dx
# map each fine cell to a step index
fine_to_step = np.clip(np.floor((fine_x + 0.25) / dx_step).astype(int), 0, N-1)

_fft_n = 1  # will set
def _autocorr_max(f_fine):
    # f_fine length M, values = step heights on fine grid
    n = 1
    while n < 2*M:
        n *= 2
    g = np.zeros(n)
    g[:M] = f_fine
    F = np.fft.rfft(g)
    ac = np.fft.irfft(F * F.conj(), n=n)
    # ac[k] = sum g[i] g[i+k] for linear (due to zero pad)
    return float(np.max(ac))

def _f_from_h(h):
    # h: length N nonneg
    f_fine = h[fine_to_step]
    int_f2 = dx_step * float(np.sum(h*h))
    int_f = dx_step * float(np.sum(h))
    if int_f <= 0:
        return 0.0
    acmax = _autocorr_max(f_fine)
    if acmax <= 0:
        return 0.0
    return int_f2 / (int_f * acmax)

def _objective(x):
    x = np.maximum(x, 0.0)
    mx = np.max(x)
    if mx <= 0:
        return 0.0
    h = x / mx  # peak = 1
    return -_f_from_h(h)

def _param_shapes():
    shapes = []
    # Gaussian
    for sig in [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2]:
        shapes.append(np.exp(-step_centers**2 / (2*sig*sig)))
    # Lorentzian
    for sig in [0.02, 0.05, 0.1]:
        shapes.append(1.0/(1.0 + (step_centers/sig)**2))
    # Triangle
    for w in [0.02, 0.05, 0.1, 0.2]:
        shapes.append(np.clip(1.0 - np.abs(step_centers)/w, 0, None))
    # Plateau
    for w in [0.02, 0.05, 0.1]:
        shapes.append((np.abs(step_centers) < w).astype(float))
    # Single step (center)
    s = np.zeros(N); s[N//2] = 1.0; shapes.append(s)
    # Two steps
    s = np.zeros(N); s[N//2-2]=1; s[N//2+2]=1; shapes.append(s)
    return shapes

def construct():
    best_h = np.ones(N)
    best_f = _f_from_h(best_h)
    t0 = time.time()
    starts = _param_shapes()
    # add a couple random
    rng = np.random.default_rng(0)
    for _ in range(4):
        starts.append(rng.random(N))
    bounds = [(0.0, 10.0)] * N
    for s in starts:
        if time.time() - t0 > DEADLINE:
            break
        try:
            res = minimize(_objective, s, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-8})
            x = np.maximum(res.x, 0.0)
            mx = np.max(x)
            if mx <= 0: continue
            h = x / mx
            f = _f_from_h(h)
            if f > best_f:
                best_f = f
                best_h = h
        except Exception:
            pass
    # ensure nonneg
    best_h = np.maximum(best_h, 0.0)
    return [float(v) for v in best_h]

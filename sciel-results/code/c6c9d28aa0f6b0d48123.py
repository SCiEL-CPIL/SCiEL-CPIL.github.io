"""Second autocorrelation inequality -- combined model-and-optimize + coordinate polish.

Strategy:
- Model h as a sum of Gaussian bumps (smooth, fewer params than 50 raw heights).
- Use soft-max (log-sum-exp) to smooth the ||h*h||_inf term for gradient-based optimization.
- Multi-start with diverse initializations (Gaussian, bimodal, random smooth).
- L-BFGS-B on log-R with analytic gradient for fast local optimization.
- Final high-precision coordinate descent polish on the raw 50-dimensional h.
- Scale-invariant: fix ||h||_2 = 1, maximize 1/(||h||_1 * ||h*h||_inf).
"""

import time
import math
import numpy as np
from scipy.optimize import minimize

N = 50
DH = 1.0 / 20.0
OFFSET = 10  # half of N, so that autoconvolution is centered

_IDX = np.arange(N, dtype=float)

_start = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill


def _elapsed():
    return time.time() - _start


def _time_left():
    return _DEADLINE - _elapsed()


# --------------------------------------------------------------------------- #
#  Autoconvolution helper
# --------------------------------------------------------------------------- #
def _autoconv(h):
    """Linear autoconvolution sampled on the same grid (centered)."""
    full = np.convolve(h, h, mode='full')
    return DH * full[OFFSET:OFFSET + N]


def _ratio_from_h(h):
    """R = ||h||_2^2 / (||h||_1 * ||h*h||_inf). Higher is better."""
    h = np.maximum(h, 0.0)
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 1e-300 or l2sq <= 1e-300:
        return 0.0
    ac = _autoconv(h)
    acmax = float(np.max(ac))
    if acmax <= 1e-300:
        return 0.0
    return l2sq / (l1 * acmax)


# --------------------------------------------------------------------------- #
#  Gaussian bump basis for model-and-optimize
# --------------------------------------------------------------------------- #
def _make_basis():
    """Gaussian bump basis: M centers, 2 widths each."""
    M = 8
    centers = np.linspace(2.0, N - 3.0, M)
    widths = [1.8, 3.5, 6.0]
    basis = []
    for c in centers:
        for w in widths:
            basis.append(np.exp(-0.5 * ((_IDX - c) / w) ** 2))
    B = np.stack(basis, axis=1)  # shape (N, P)
    return B


B = _make_basis()
P = B.shape[1]


def _forward(params):
    """h = max(0, B @ params)."""
    h = B @ params
    np.maximum(h, 0.0, out=h)
    return h


def _logR_and_grad(params, smooth_beta=40.0):
    """Analytic gradient of log R with soft-max smoothing w.r.t. params."""
    h_raw = B @ params
    pos = h_raw > 0.0
    h = np.where(pos, h_raw, 0.0)

    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 1e-300 or l2sq <= 1e-300:
        return -1e30, np.zeros_like(params)

    ac = _autoconv(h)
    ac_pos = np.maximum(ac, 1e-300)
    ac_max = float(np.max(ac_pos))
    if ac_max <= 1e-300:
        return -1e30, np.zeros_like(params)

    # Soft-max: logsumexp(ac / beta) / beta
    inv_beta = 1.0 / smooth_beta
    shifted = ac_pos - ac_max
    exp_shifted = np.exp(shifted * inv_beta)
    sum_exp = float(np.sum(exp_shifted))
    if sum_exp <= 1e-300:
        return -1e30, np.zeros_like(params)
    log_sum_exp = ac_max + inv_beta * math.log(sum_exp)

    R = l2sq / (l1 * math.exp(log_sum_exp))
    if R <= 1e-300:
        return -1e30, np.zeros_like(params)
    logR = math.log(R)

    # Softmax weights for the soft-max
    softmax_w = exp_shifted / sum_exp  # shape (N,)

    # d logR / d h_i = 2*h_i/l2sq - 1/l1 - d log(acmax_smooth)/dh_i
    # d log(acmax_smooth) / dh_k = (2*DH/beta) * sum_i softmax_w[i] * h[i-k] (if 0<=i-k<N)
    
    dlog_acmax_dh = np.zeros(N)
    for k in range(N):
        s = 0.0
        for i in range(N):
            j = i - k
            if 0 <= j < N:
                s += softmax_w[i] * h[j]
        dlog_acmax_dh[k] = (2.0 * DH / smooth_beta) * s

    dlogR_dh = (2.0 * h / l2sq) - (1.0 / l1) - dlog_acmax_dh

    # Chain rule through h = max(0, B @ params)
    dlogR_dh_active = np.where(pos, dlogR_dh, 0.0)
    grad = B.T @ dlogR_dh_active

    if not np.all(np.isfinite(grad)):
        grad = np.zeros_like(params)
        eps = 1e-6
        for p in range(P):
            pp = params.copy(); pp[p] += eps
            pm = params.copy(); pm[p] -= eps
            lp, _ = _logR_and_grad(pp, smooth_beta)
            lm, _ = _logR_and_grad(pm, smooth_beta)
            grad[p] = (lp - lm) / (2.0 * eps)

    return logR, grad


def _neg_logR(params):
    v, _ = _logR_and_grad(params)
    return v


def _neg_logR_jac(params):
    v, g = _logR_and_grad(params)
    return v, -g


# --------------------------------------------------------------------------- #
#  Coordinate descent polish on raw 50-dim h
# --------------------------------------------------------------------------- #
def _project(h):
    """Project onto h >= 0 and ||h||_2 = 1."""
    h = np.maximum(h, 0.0)
    norm2 = np.sqrt(np.sum(h * h))
    if norm2 < 1e-30:
        return np.ones(N) / np.sqrt(N)
    return h / norm2


def _coord_descent(h0, time_budget=5.0):
    """Fine coordinate descent to maximize R with ||h||_2=1, h>=0."""
    h = _project(h0.copy())
    curr_val = _ratio_from_h(h)
    
    deadline = time.time() + time_budget
    step_sizes = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    
    for step in step_sizes:
        if time.time() > deadline:
            break
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(N):
                # Try increasing
                h_trial = h.copy()
                h_trial[i] += step
                h_trial = _project(h_trial)
                trial_val = _ratio_from_h(h_trial)
                if trial_val > curr_val + 1e-12:
                    h = h_trial
                    curr_val = trial_val
                    improved = True
                    continue
                # Try decreasing
                h_trial = h.copy()
                h_trial[i] -= step
                h_trial = _project(h_trial)
                trial_val = _ratio_from_h(h_trial)
                if trial_val > curr_val + 1e-12:
                    h = h_trial
                    curr_val = trial_val
                    improved = True
    
    return h, curr_val


# --------------------------------------------------------------------------- #
#  Main construct
# --------------------------------------------------------------------------- #
def construct():
    global _start
    _start = time.time()
    
    best_h = None
    best_score = 0.0
    
    # --- Phase 1: Model-and-optimize with L-BFGS-B ---
    rng = np.random.RandomState(12345)
    
    # Generate diverse initializations for the parameter space
    inits_params = []
    
    # All positive (broad bumps)
    inits_params.append(np.ones(P) * 0.5)
    
    # Centered
    for c in [N//4, N//2, 3*N//4]:
        p = np.zeros(P)
        for i in range(P):
            # Assign based on which center this basis function is closest to
            pass
        inits_params.append(p)
    
    # Random smooth initializations
    for _ in range(15):
        p = rng.randn(P) * 0.3 + 0.2
        inits_params.append(p)
    
    # Bimodal-like
    for _ in range(5):
        p = np.zeros(P)
        # First half positive, second half positive
        half = P // 2
        p[:half] = rng.rand(half) * 0.5 + 0.1
        p[half:] = rng.rand(P - half) * 0.5 + 0.1
        inits_params.append(p)
    
    # Sparse center
    for _ in range(3):
        p = np.zeros(P)
        center_idx = P // 2
        p[center_idx-1:center_idx+2] = rng.rand(3) * 0.5 + 0.2
        inits_params.append(p)
    
    # Try each initialization with L-BFGS-B
    for p0 in inits_params:
        if _time_left() < 10.0:
            break
        
        p0 = p0.copy()
        
        try:
            res = minimize(
                _neg_logR,
                p0,
                jac=_neg_logR_jac,
                method='L-BFGS-B',
                options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-12}
            )
        except Exception:
            continue
        
        h_opt = _forward(res.x)
        score = _ratio_from_h(h_opt)
        
        if score > best_score:
            best_score = score
            best_h = h_opt.copy()
    
    # --- Phase 2: Also try direct coordinate descent from diverse raw inits ---
    raw_inits = []
    
    # Uniform
    raw_inits.append(np.ones(N))
    
    # Gaussian centered
    x = np.linspace(-1, 1, N)
    raw_inits.append(np.exp(-5 * x**2))
    
    # Bimodal
    raw_inits.append(np.exp(-20 * (x - 0.5)**2) + np.exp(-20 * (x + 0.5)**2))
    
    # Cosine
    raw_inits.append(np.cos(np.pi * x / 2.0) ** 2)
    
    # Sparse center
    h_sp = np.zeros(N)
    h_sp[N//2-2:N//2+3] = 1.0
    raw_inits.append(h_sp)
    
    # Random
    for _ in range(10):
        raw_inits.append(rng.random(N))
    
    for h0 in raw_inits:
        if _time_left() < 8.0:
            break
        
        h0 = np.maximum(h0, 0.0)
        if np.sum(h0*h0) < 1e-30:
            continue
        
        h_init = _project(h0)
        score_init = _ratio_from_h(h_init)
        
        if score_init > best_score:
            best_score = score_init
            best_h = h_init.copy()
        
        # Quick coordinate descent
        h_cd, score_cd = _coord_descent(h_init, time_budget=2.0)
        if score_cd > best_score:
            best_score = score_cd
            best_h = h_cd.copy()
    
    # --- Phase 3: Final high-precision polish ---
    if best_h is not None and _time_left() > 5.0:
        budget = min(_time_left() - 3.0, 10.0)
        h_final, score_final = _coord_descent(best_h, time_budget=budget)
        if score_final > best_score:
            best_score = score_final
            best_h = h_final
    
    if best_h is None:
        return [1.0] * N
    
    return [float(x) for x in best_h]

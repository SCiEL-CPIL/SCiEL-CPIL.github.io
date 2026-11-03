"""Erdos minimum overlap: FFT-based smooth surrogate + SLSQP epigraph optimization.
Combines the vectorized FFT gradient computation for speed with SLSQP using
exact linear constraints (sum=h_target, bounds [0,1]) on a parameterized (N-1)
formulation. Uses epigraph t-variable at high beta and diverse restarts.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_TARGET = N / 2.0
_NFFT = 256  # next power of 2 >= 2*N

def _autocorr(h):
    """Linear autocorrelation r[k] = sum_{i=k}^{N-1} h[i]*h[i-k] for k=0..N-1 via FFT."""
    h_pad = np.zeros(_NFFT)
    h_pad[:N] = h
    H = np.fft.rfft(h_pad)
    corr = np.fft.irfft(H * np.conj(H), n=_NFFT)
    return corr[:N]

def _overlap_values(h):
    """C_k = (2/N)*(N/2 - r_k) for k=0..N-1."""
    r = _autocorr(h)
    return (2.0 / N) * (N / 2.0 - r)

def _score(h):
    """True (hard) maximum overlap value."""
    cs = _overlap_values(h)
    return float(np.max(cs))

def _project(h):
    """Project h to sum=N/2, values in [0,1]."""
    h = np.array(h, dtype=np.float64, copy=True)
    h = np.clip(h, 0.0, 1.0)
    diff = _TARGET - np.sum(h)
    if abs(diff) > 1e-14:
        for _ in range(5):
            if abs(diff) < 1e-14:
                break
            if diff > 0:
                mask = h < 1.0
            else:
                mask = h > 0.0
            cnt = np.sum(mask)
            if cnt == 0:
                break
            h[mask] += diff / cnt
            h = np.clip(h, 0.0, 1.0)
            diff = _TARGET - np.sum(h)
    diff = _TARGET - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
    return h

def _smooth_max_and_grad(x, beta):
    """Log-sum-exp soft-max of x and its gradient (softmax(beta*x))."""
    m = np.max(x)
    e = np.exp(beta * (x - m))
    s = np.sum(e)
    val = (m + np.log(s)) / beta
    grad = e / s
    return val, grad

def _grad_smooth_max_wrt_h(h, g_c):
    """Compute gradient of S_beta(C(h)) w.r.t. h, given g_c = dS/dC.
    Uses FFT-based convolution for speed.
    """
    factor = -(2.0 / N)
    gc_pad = np.zeros(_NFFT)
    gc_pad[:N] = g_c
    h_pad = np.zeros(_NFFT)
    h_pad[:N] = h
    hr_pad = np.zeros(_NFFT)
    hr_pad[:N] = h[::-1]
    
    GC = np.fft.rfft(gc_pad)
    HC = np.fft.rfft(h_pad)
    HRC = np.fft.rfft(hr_pad)
    
    conv1 = np.fft.irfft(GC * HC, n=_NFFT)[:N]
    conv2_full = np.fft.irfft(GC * HRC, n=_NFFT)
    term2 = conv2_full[N-1::-1][:N]
    
    g_h = factor * (conv1 + term2)
    return g_h

def _epigraph_obj_grad(x, beta, t):
    """Epigraph formulation: minimize t + S_beta(C(h) - t) where h is reconstructed from x.
    x has length N-1; h[0] = _TARGET - sum(x[1:])... actually let's use h = [h0, x[0], ..., x[N-2]]
    where h0 is determined by the sum constraint.
    """
    # Reconstruct h from x (N-1 variables)
    # h = [h0, x[0], x[1], ..., x[N-2]]
    # sum(h) = h0 + sum(x) = _TARGET => h0 = _TARGET - sum(x)
    h_tail = x
    h0 = _TARGET - np.sum(h_tail)
    h = np.concatenate(([h0], h_tail))
    
    cs = _overlap_values(h)
    shifted = cs - t
    val_smooth, g_c = _smooth_max_and_grad(shifted, beta)
    val = t + val_smooth
    
    # Gradient w.r.t. h
    g_h = _grad_smooth_max_wrt_h(h, g_c)
    
    # Gradient w.r.t. x: x corresponds to h[1:]
    # dh/dx: h[0] = _TARGET - sum(x), so dh[0]/dx[j] = -1
    # h[j+1] = x[j], so dh[j+1]/dx[j] = 1
    g_x = np.empty(N - 1)
    g_x[:] = g_h[1:]  # from h[1:] = x
    g_x -= g_h[0]     # from h[0] = _TARGET - sum(x)
    
    return val, g_x

def _epigraph_obj_grad_t_only(x, beta, t):
    """Same as above but returns just the value for objective."""
    h_tail = x
    h0 = _TARGET - np.sum(h_tail)
    h = np.concatenate(([h0], h_tail))
    cs = _overlap_values(h)
    shifted = cs - t
    val_smooth, _ = _smooth_max_and_grad(shifted, beta)
    return t + val_smooth

def _slsqp_optimize(h_init, beta, t_init, time_budget=15.0):
    """Run SLSQP on the epigraph formulation with exact constraints."""
    t_start = time.time()
    
    # Convert h_init to x (N-1 variables)
    x_init = h_init[1:].copy()
    
    def _obj(x):
        return _epigraph_obj_grad(x, beta, t_init)[0]
    
    def _grad(x):
        return _epigraph_obj_grad(x, beta, t_init)[1]
    
    # Bounds: h[0] = _TARGET - sum(x) must be in [0, 1]
    # h[j+1] = x[j] must be in [0, 1]
    # h[0] >= 0 => sum(x) <= _TARGET
    # h[0] <= 1 => sum(x) >= _TARGET - 1
    # x[j] in [0, 1]
    
    bounds = [(0.0, 1.0)] * (N - 1)
    
    # Linear equality constraint: sum(h) = _TARGET is automatically satisfied by construction
    # But we need h[0] in [0,1], which means sum(x) in [_TARGET-1, _TARGET]
    # This is a linear constraint on x: _TARGET - 1 <= sum(x) <= _TARGET
    
    cons = [
        {'type': 'ineq', 'fun': lambda x: sum(x) - (_TARGET - 1.0), 'jac': lambda x: np.ones(N-1)},
        {'type': 'ineq', 'fun': lambda x: _TARGET - sum(x), 'jac': lambda x: -np.ones(N-1)},
    ]
    
    res = minimize(
        _obj, x_init, jac=_grad, method='SLSQP', bounds=bounds, constraints=cons,
        options={'maxiter': 200, 'ftol': 1e-10, 'disp': False, 'maxls': 50}
    )
    
    if time.time() - t_start > time_budget:
        return h_init, _score(h_init)
    
    x_opt = res.x
    h0 = _TARGET - np.sum(x_opt)
    h_opt = np.concatenate(([h0], x_opt))
    h_opt = _project(h_opt)
    return h_opt, _score(h_opt)

def _nesterov_projected_gradient(h0, beta, n_iter=300, step_size=0.01):
    """Nesterov accelerated projected gradient on smooth-max surrogate."""
    h = h0.copy()
    h_prev = h0.copy()
    gamma = 0.5
    for it in range(n_iter):
        h_next = h + gamma * (h - h_prev)
        cs = _overlap_values(h_next)
        _, g_c = _smooth_max_and_grad(cs, beta)
        g_h = _grad_smooth_max_wrt_h(h_next, g_c)
        h_new = _project(h_next - step_size * g_h)
        gamma = (1 + np.sqrt(1 + 4 * gamma * gamma)) / 2
        h_prev = h
        h = h_new
    return h

def _construct_direct():
    """Direct multi-frequency construction to flatten autocorrelation."""
    i = np.arange(N)
    h = np.full(N, 0.5)
    freqs = [1.0, 2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0]
    amplitudes = [0.12, 0.10, 0.09, 0.07, 0.06, 0.05, 0.04, 0.03, 0.03, 0.02]
    phases = [0.0, 0.3, 0.7, 1.2, 1.8, 2.3, 2.9, 3.4, 4.0, 4.5]
    for f, a, p in zip(freqs, amplitudes, phases):
        h += a * np.cos(2.0 * np.pi * f * i / N + p)
    return _project(h)

def _random_start(rng):
    """Random feasible start."""
    h = rng.uniform(0.0, 1.0, N)
    return _project(h)

def _perturbed_start(h_base, rng, scale=0.05):
    """Create a perturbed version of a base solution."""
    h = h_base + rng.normal(0.0, scale, N)
    return _project(h)

def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    deadline = t0 + 140  # Safety margin before 150s kill
    rng = np.random.default_rng(42)

    best_h = None
    best_score = float('inf')

    def _update(h):
        nonlocal best_h, best_score
        h = _project(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
        return s

    # Phase 1: Direct construction
    h_direct = _construct_direct()
    _update(h_direct)

    # Phase 2: Diverse frequency-based starts
    if time.time() < deadline - 60:
        i = np.arange(N)
        freq_sets = [
            ([1, 2, 3, 5, 7, 11], [0.12, 0.10, 0.09, 0.07, 0.06, 0.05]),
            ([1, 2, 4, 7, 13, 17], [0.11, 0.10, 0.08, 0.06, 0.05, 0.04]),
            ([1, 3, 4, 6, 9, 14], [0.10, 0.09, 0.08, 0.06, 0.05, 0.04]),
            ([2, 3, 5, 8, 13, 19], [0.10, 0.09, 0.08, 0.06, 0.05, 0.04]),
            ([1, 2, 3, 4, 5, 6, 8, 11], [0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.03]),
            ([3, 5, 7, 11, 13, 17], [0.10, 0.09, 0.08, 0.06, 0.05, 0.04]),
            ([1, 2, 3, 4, 5, 7, 9], [0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04]),
            ([2, 4, 6, 8, 10, 12, 15], [0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03]),
            ([1, 2, 3, 5, 8, 13, 21], [0.11, 0.10, 0.09, 0.07, 0.06, 0.05, 0.04]),
            ([1, 3, 5, 7, 11, 13, 17, 19], [0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.03]),
        ]
        for freqs, amps in freq_sets:
            if time.time() > deadline - 60:
                break
            h = np.full(N, 0.5)
            for f, a in zip(freqs, amps):
                h += a * np.cos(2.0 * np.pi * f * i / N)
            _update(h)

    # Phase 3: Random starts
    if time.time() < deadline - 50:
        for _ in range(5):
            if time.time() > deadline - 50:
                break
            h_rand = _random_start(rng)
            _update(h_rand)

    # Phase 4: Nesterov refinement from best starts with multiple betas
    beta_list = [20.0, 50.0, 100.0, 200.0]
    starts_to_refine = []
    if best_h is not None:
        starts_to_refine.append(best_h.copy())
    
    # Add diverse starts
    if time.time() < deadline - 80:
        for _ in range(3):
            if time.time() > deadline - 80:
                break
            starts_to_refine.append(_random_start(rng))
    
    for h_start in starts_to_refine:
        for beta in beta_list:
            if time.time() > deadline - 40:
                break
            h_nest = _nesterov_projected_gradient(h_start, beta, n_iter=200, step_size=0.005)
            _update(h_nest)

    # Phase 5: SLSQP epigraph refinement from best
    if best_h is not None and time.time() < deadline - 20:
        t_init = best_score + 0.05
        for beta in [100.0, 200.0, 500.0]:
            if time.time() > deadline - 10:
                break
            h_slq, s_slq = _slsqp_optimize(best_h, beta, t_init, time_budget=12.0)
            _update(h_slq)
            # Update t_init based on current best
            t_init = best_score + 0.02

    # Phase 6: Perturbation-based restarts with SLSQP
    if best_h is not None and time.time() < deadline - 15:
        for _ in range(3):
            if time.time() > deadline - 10:
                break
            h_pert = _perturbed_start(best_h, rng, scale=0.03)
            _update(h_pert)
            if time.time() > deadline - 10:
                break
            h_slq, s_slq = _slsqp_optimize(best_h, 300.0, best_score + 0.02, time_budget=8.0)
            _update(h_slq)

    # Phase 7: Final polish with very high beta Nesterov
    if best_h is not None and time.time() < deadline - 5:
        h_polish = _nesterov_projected_gradient(best_h, 500.0, n_iter=100, step_size=0.002)
        _update(h_polish)

    if best_h is None:
        best_h = np.full(N, 0.5)

    return _project(best_h)

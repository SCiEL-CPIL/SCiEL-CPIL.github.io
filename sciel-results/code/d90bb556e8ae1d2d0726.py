"""Solve Erdos minimum overlap via spectral parameterization with epigraph formulation."""
import numpy as np
from scipy.optimize import minimize
import time

N = 100
DEADLINE = 135.0  # safety margin before 150s kill

def _autocorr(g):
    """Circular autocorrelation of g."""
    G = np.fft.fft(g)
    ac = np.fft.ifft(G * np.conj(G)).real
    return ac

def _overlap_values(h):
    """Compute overlap values for each shift."""
    one_minus_h = 1.0 - h
    H = np.fft.fft(h)
    O = np.fft.fft(one_minus_h)
    cross = np.fft.ifft(H * np.conj(O)).real
    return cross

def _score(h):
    """Score: max overlap value (lower is better)."""
    return np.max(_overlap_values(h))

def _fix_sum(h):
    """Adjust h so sum is exactly N/2, respecting [0,1] bounds."""
    target = N / 2.0
    h = np.clip(h, 0.0, 1.0)
    for _ in range(200):
        diff = target - np.sum(h)
        if abs(diff) < 1e-12:
            break
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    diff = target - np.sum(h)
    if abs(diff) > 1e-10:
        remaining = diff
        flexibility = np.minimum(h, 1.0 - h)
        order = np.argsort(-flexibility)
        for idx in order:
            if abs(remaining) < 1e-12:
                break
            if remaining > 0:
                adj = min(1.0 - h[idx], remaining)
            else:
                adj = max(-h[idx], remaining)
            h[idx] += adj
            remaining -= adj
    return h

def _h_from_spectrum(S, phase):
    """Reconstruct h from power spectrum and phase."""
    S = np.asarray(S, dtype=float).copy()
    phase = np.asarray(phase, dtype=float).copy()
    S[0] = 0.0
    if N % 2 == 0:
        S[N // 2] = 0.0
    S = np.maximum(S, 0.0)
    G = np.sqrt(S) * np.exp(1j * phase)
    # Enforce Hermitian symmetry
    if N % 2 == 0:
        for k in range(1, N // 2):
            G[N - k] = np.conj(G[k])
    else:
        for k in range(1, (N + 1) // 2):
            G[N - k] = np.conj(G[k])
    g = np.fft.ifft(G).real
    g -= g.mean()
    h = 0.5 + g
    return _fix_sum(h)

def _epigraph_obj(x, beta):
    """Epigraph objective: -t + beta * sum(max(0, t - ac)^2)."""
    S = np.maximum(x[:N], 0.0)
    t = x[N]
    ac = np.fft.ifft(S).real
    viol = np.maximum(0.0, t - ac)
    return -t + beta * np.sum(viol ** 2)

def _epigraph_grad(x, beta):
    """Gradient of epigraph objective."""
    S = np.maximum(x[:N], 0.0)
    t = x[N]
    ac = np.fft.ifft(S).real
    viol = np.maximum(0.0, t - ac)
    
    grad = np.zeros(N + 1)
    grad[N] = -1.0 + beta * 2.0 * np.sum(viol)
    
    if np.any(viol > 0):
        # d ac[j]/d S[k] = (1/N) * cos(2*pi*j*k/N)
        # grad_S[k] = beta * 2 * sum_j viol[j] * (1/N) * cos(2*pi*j*k/N)
        viol_fft = np.fft.fft(viol)
        grad[:N] = beta * 2.0 / N * viol_fft.real
        grad[x[:N] <= 0] = 0.0
    
    return grad

def _polish(h, max_iter=100):
    """Polish h via gradient ascent on min autocorrelation."""
    for _ in range(max_iter):
        if time.time() > DEADLINE:
            break
        g = h - 0.5
        ac = _autocorr(g)
        k_min = np.argmin(ac)
        # grad of ac[k_min] w.r.t. g: 2 * roll(g, -k_min)
        grad_g = 2.0 * np.roll(g, -k_min)
        lr = 0.003
        h = h + lr * grad_g
        h = np.clip(h, 0.0, 1.0)
        h = _fix_sum(h)
    return h

def construct():
    start = time.time()
    rng = np.random.default_rng(12345)
    
    best_h = np.full(N, 0.5)
    best_score = _score(best_h)
    
    for restart in range(8):
        if time.time() > DEADLINE - 25:
            break
        
        if restart == 0:
            S_init = np.ones(N)
            S_init[0] = 0.0
            S_init[N // 2] = 0.0
            phase_init = np.zeros(N)
        else:
            S_init = rng.exponential(1.0, N)
            S_init[0] = 0.0
            S_init[N // 2] = 0.0
            phase_init = rng.uniform(-np.pi, np.pi, N)
        
        x0 = np.concatenate([S_init, [0.0]])
        bounds = [(0.0, None)] * N + [(-0.5, 0.5)]
        
        try:
            result = minimize(
                lambda x: _epigraph_obj(x, 100.0),
                x0,
                jac=lambda x: _epigraph_grad(x, 100.0),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 150, 'ftol': 1e-8, 'disp': False}
            )
        except Exception:
            result = None
        
        if time.time() > DEADLINE - 10:
            break
        
        if result is not None and result.fun < _epigraph_obj(x0, 100.0):
            S_opt = np.maximum(result.x[:N], 0.0)
            h_cand = _h_from_spectrum(S_opt, phase_init)
            h_cand = _polish(h_cand, max_iter=60)
            sc = _score(h_cand)
            if sc < best_score:
                best_score = sc
                best_h = h_cand.copy()
        
        # Perturbed phase
        if time.time() > DEADLINE - 15:
            break
        phase_pert = phase_init + rng.normal(0, 0.15, N)
        if result is not None:
            S_opt = np.maximum(result.x[:N], 0.0)
        else:
            S_opt = S_init
        h_cand2 = _h_from_spectrum(S_opt, phase_pert)
        h_cand2 = _polish(h_cand2, max_iter=40)
        sc2 = _score(h_cand2)
        if sc2 < best_score:
            best_score = sc2
            best_h = h_cand2.copy()
    
    if time.time() < DEADLINE - 5:
        best_h = _polish(best_h, max_iter=80)
        best_score = _score(best_h)
    
    return _fix_sum(best_h)

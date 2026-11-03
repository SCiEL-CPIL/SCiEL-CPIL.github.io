"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(h):
    """Compute the objective value: max|linear autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    size = 2 * N - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    Hp = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(Hp * Hp, fsize)
    return np.max(np.abs(ac)) / (s * s)

def _spectral_optimize():
    """
    Minimize max_k |circular_autoconv_k| / H_0^2 by optimizing the power spectrum.
    Uses the fact that circular autoconv is diagonalized by DFT.
    Normalize H_0 = 1, so p_0 = 1/N where p_m = |H_m|^2 / N.
    Variables: p_1, ..., p_{N/2} with p_m = p_{N-m} >= 0.
    Objective: max_k |sum_m p_m exp(2*pi*i*m*k/N)|
    """
    t_start = time.time()
    deadline = t_start + 85.0
    
    n_half = N // 2
    p0 = 1.0 / N
    
    def _objective_and_grad(x):
        # x = [p_1, ..., p_{n_half}], all >= 0
        p = np.zeros(N)
        p[0] = p0
        for k in range(1, n_half + 1):
            p[k] = x[k-1]
            p[N-k] = x[k-1]
        
        # c_k = sum_m p_m exp(2*pi*i*m*k/N) = N * ifft(p)
        c = N * np.fft.ifft(p)
        ac = np.abs(c)
        max_val = np.max(ac)
        
        # Gradient: d(max |c_k|)/dp_j
        # This is tricky. Use subgradient: the index k* where max is achieved
        k_star = np.argmax(ac)
        # c_k = sum_m p_m exp(2*pi*i*m*k/N)
        # dc_k/dp_j = exp(2*pi*i*j*k/N) for j != 0, N-j
        # For j in 1..n_half, p_j = p_{N-j} = x_{j-1}
        # d c_k / d x_{j-1} = exp(2*pi*i*j*k/N) + exp(2*pi*i*(N-j)*k/N)
        # = exp(2*pi*i*j*k/N) + exp(2*pi*i*(-j)*k/N) = 2*cos(2*pi*j*k/N)
        
        grad = np.zeros(n_half)
        for j in range(1, n_half + 1):
            # Phase of c_{k_star}
            phase = np.angle(c[k_star])
            # d|c_k|/dp_j = Re(exp(-i*phase) * dc_k/dp_j)
            # dc_k/dx_{j-1} = 2*cos(2*pi*j*k/N) (real)
            dc_dx = 2.0 * np.cos(2.0 * np.pi * j * k_star / N)
            grad[j-1] = np.real(np.exp(-1j * phase) * dc_dx)
        
        return max_val, grad
    
    def _objective(x):
        p = np.zeros(N)
        p[0] = p0
        for k in range(1, n_half + 1):
            p[k] = x[k-1]
            p[N-k] = x[k-1]
        c = N * np.fft.ifft(p)
        return np.max(np.abs(c))
    
    # Initializations
    initializations = []
    
    # 1. Flat
    initializations.append(np.full(n_half, p0))
    
    # 2. Decaying
    for alpha in [0.5, 1.0, 1.5, 2.0]:
        initializations.append(np.array([p0 / (1.0 + (k+1)**alpha) for k in range(n_half)]))
    
    # 3. Cosine
    initializations.append(np.array([p0 * np.cos(np.pi * (k+1) / (2*(n_half+1))) for k in range(n_half)]))
    
    best_x = initializations[0].copy()
    best_score = _objective(best_x)
    
    for x_init in initializations:
        if time.time() > deadline - 10:
            break
        try:
            result = minimize(_objective, x_init, method='L-BFGS-B',
                            bounds=[(0, None)] * n_half,
                            options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8})
            if result.fun < best_score:
                best_score = result.fun
                best_x = result.x.copy()
        except Exception:
            continue
    
    # Reconstruct h
    p = np.zeros(N)
    p[0] = p0
    for k in range(1, n_half + 1):
        p[k] = best_x[k-1]
        p[N-k] = best_x[k-1]
    
    # |H_m| = sqrt(N * p_m), choose phases to make h real
    H = np.sqrt(N * p)
    h = np.fft.ifft(H).real
    
    return h, best_score

def _time_domain_refine(h):
    """Local search in time domain with sign flips."""
    t_start = time.time()
    deadline = t_start + 20.0
    
    h_refined = h.copy()
    best_local = _score(h_refined)
    
    for _ in range(3):
        improved = False
        for idx in range(N):
            if time.time() > deadline - 3:
                break
            h_refined[idx] = -h_refined[idx]
            new_val = _score(h_refined)
            if new_val < best_local:
                best_local = new_val
                improved = True
            else:
                h_refined[idx] = -h_refined[idx]
        if not improved:
            break
    
    return h_refined, best_local

def construct():
    t_start = time.time()
    deadline = t_start + 115.0
    
    best_score = np.inf
    best_h = np.ones(N) / N
    
    base_score = _score(best_h)
    best_score = min(best_score, base_score)
    
    # Spectral optimization
    if time.time() < deadline - 25:
        h_spec, score_spec = _spectral_optimize()
        if score_spec < best_score:
            best_score = score_spec
            best_h = h_spec.copy()
    
    # Time-domain refinement
    if time.time() < deadline - 3:
        h_ref, score_ref = _time_domain_refine(best_h)
        if score_ref < best_score:
            best_score = score_ref
            best_h = h_ref.copy()
    
    return best_h

# EVOLVE-BLOCK-END

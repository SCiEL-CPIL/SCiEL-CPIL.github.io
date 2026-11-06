import time
import numpy as np
from scipy.optimize import minimize
from scipy.fft import dct as scipy_dct, idct as scipy_idct

N = 400
_t0 = time.time()
_DEADLINE = 145.0  # Safety margin before 150s kill

def _value(h):
    """Compute the exact task value (lower is better)."""
    h = np.asarray(h, dtype=float)
    if not np.all(np.isfinite(h)):
        return np.inf
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return np.inf
    
    size = 2 * N
    H = np.fft.rfft(h, size)
    conv = np.fft.irfft(H * H, size)[:2 * N - 1]
    
    m = float(np.max(conv))
    val = abs(2.0 * N * m / (s * s))
    return val

def _smooth_value(h, beta=50.0):
    """Smooth surrogate using LogSumExp to approximate max(conv)."""
    h = np.asarray(h, dtype=float)
    if not np.all(np.isfinite(h)):
        return np.inf
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return np.inf
    
    size = 2 * N
    H = np.fft.rfft(h, size)
    conv = np.fft.irfft(H * H, size)[:2 * N - 1]
    
    m = np.max(conv)
    scaled = beta * (conv - m)
    scaled = np.clip(scaled, -100, 100)
    lse = m + np.log(np.sum(np.exp(scaled))) / beta
    
    val = abs(2.0 * N * lse / (s * s))
    return val

def _dct_h(coeffs):
    """Reconstruct h from DCT-II coefficients."""
    h = scipy_idct(coeffs, type=2, n=N, norm='ortho')
    return h

def _dct_smooth_obj(coeffs, beta=50.0):
    """Objective for DCT coefficients with smooth surrogate."""
    h = _dct_h(coeffs)
    return _smooth_value(h, beta)

def _dct_exact_obj(coeffs):
    """Exact objective for DCT coefficients."""
    h = _dct_h(coeffs)
    return _value(h)

def _generate_initial_dct(rng, num_coeffs):
    """Generate random DCT coefficients with low-frequency bias."""
    coeffs = np.zeros(num_coeffs)
    coeffs[0] = rng.normal(0, 1.0) * 10.0
    weights = 1.0 / (1.0 + np.arange(num_coeffs))
    coeffs[1:] = rng.normal(0, 1.0, size=num_coeffs - 1) * weights[1:] * 2.0
    return coeffs

def _construct_atom(atom_len):
    """Construct a Hamming window atom."""
    if atom_len <= 1:
        return np.array([1.0])
    return np.hamming(atom_len)

def _sparse_atom_obj(params, atom, offsets, signs, beta=50.0):
    """Objective for sparse atom model."""
    M = len(params)
    h = np.zeros(N)
    for i in range(M):
        amp = params[i]
        off = offsets[i]
        sgn = signs[i]
        indices = np.arange(off, off + len(atom)) % N
        h[indices] += sgn * amp * atom
    return _smooth_value(h, beta)

def _coordinate_descent_polish(h, max_iter=200, tol=1e-10):
    """Polish h via coordinate descent on the exact objective."""
    h = h.copy()
    best_v = _value(h)
    
    for it in range(max_iter):
        improved = False
        for i in range(N):
            old = h[i]
            # Try small perturbations around current value
            for delta in [-0.1, -0.05, -0.01, 0.01, 0.05, 0.1, -0.5, 0.5]:
                h[i] = old + delta
                v = _value(h)
                if v < best_v - tol:
                    best_v = v
                    improved = True
                    break
            if not improved:
                h[i] = old
        if not improved:
            break
    
    return h, best_v

def construct():
    global _t0
    _t0 = time.time()
    
    # Start with the seed solution
    best_h = np.ones(N)
    best_v = _value(best_h)
    
    def update_best(h):
        nonlocal best_h, best_v
        h = np.asarray(h, dtype=float)
        if not np.all(np.isfinite(h)):
            return
        v = _value(h)
        if v < best_v:
            best_v = v
            best_h = h.copy()

    def deadline_ok():
        return (time.time() - _t0) < _DEADLINE

    rng = np.random.default_rng(42)
    
    # --- Strategy 1: DCT Optimization (Primary) ---
    K = 400
    restarts_dct = 15
    
    for r in range(restarts_dct):
        if not deadline_ok():
            break
        try:
            coeffs0 = _generate_initial_dct(rng, K)
            
            # Phase 1: Coarse optimization with lower beta
            res = minimize(
                lambda c: _dct_smooth_obj(c, beta=20.0),
                coeffs0,
                method='L-BFGS-B',
                options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if not deadline_ok():
                break
                
            coeffs_opt = res.x if res.x is not None else coeffs0
            
            # Phase 2: Fine optimization with higher beta
            if deadline_ok():
                res2 = minimize(
                    lambda c: _dct_smooth_obj(c, beta=100.0),
                    coeffs_opt,
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res2.x is not None:
                    coeffs_opt = res2.x

            h_candidate = _dct_h(coeffs_opt)
            
            if np.all(np.isfinite(h_candidate)):
                v_check = _value(h_candidate)
                if v_check < best_v:
                    # Phase 3: Nelder-Mead polish on exact objective
                    if deadline_ok():
                        res3 = minimize(
                            _dct_exact_obj,
                            coeffs_opt,
                            method='Nelder-Mead',
                            options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 80, 'maxfev': 300}
                        )
                        if res3.x is not None:
                            coeffs_opt = res3.x
                        h_candidate = _dct_h(coeffs_opt)
                        v_check = _value(h_candidate)
                
                update_best(h_candidate)
                
        except Exception:
            pass

    # --- Strategy 2: Sparse Atom Model (Secondary) ---
    # Use evenly-spaced offsets with jitter for better coverage
    atom_lengths = [8, 12, 16]
    num_atoms_list = [10, 20, 30]
    
    for atom_len in atom_lengths:
        if not deadline_ok():
            break
        atom = _construct_atom(atom_len)
        
        for M in num_atoms_list:
            if not deadline_ok():
                break
                
            num_configs = 3
            for cfg in range(num_configs):
                if not deadline_ok():
                    break
                
                try:
                    # Evenly spaced base offsets with random jitter
                    base_offsets = np.linspace(0, N, M, endpoint=False).astype(int)
                    jitter = rng.integers(-3, 4, size=M)
                    offsets = (base_offsets + jitter) % N
                    # Random signs
                    signs = rng.choice([-1, 1], size=M)
                    
                    # Initial amplitudes
                    params0 = rng.normal(0, 1.0, size=M)
                    
                    # Optimize amplitudes
                    res = minimize(
                        lambda p: _sparse_atom_obj(p, atom, offsets, signs, beta=50.0),
                        params0,
                        method='L-BFGS-B',
                        options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    
                    if not deadline_ok():
                        break
                    
                    params_opt = res.x
                    h_candidate = np.zeros(N)
                    for i in range(M):
                        amp = params_opt[i]
                        off = offsets[i]
                        sgn = signs[i]
                        indices = np.arange(off, off + len(atom)) % N
                        h_candidate[indices] += sgn * amp * atom
                    
                    if np.all(np.isfinite(h_candidate)):
                        update_best(h_candidate)
                        
                except Exception:
                    pass

    # --- Final Polish: Coordinate Descent on Best h ---
    if deadline_ok() and best_v < 2.0:
        try:
            polished_h, polished_v = _coordinate_descent_polish(best_h, max_iter=100, tol=1e-12)
            if polished_v < best_v:
                best_h = polished_h
                best_v = polished_v
        except Exception:
            pass

    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
        
    return best_h

if __name__ == "__main__":
    h = construct()
    print(f"Value: {_value(h)}")
    print(f"Sum: {np.sum(h)}")
    print(f"Shape: {h.shape}")

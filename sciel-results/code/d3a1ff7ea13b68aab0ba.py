"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _build_sequence(heights, W):
    """Expand block heights into full N-length sequence."""
    seq = np.repeat(heights, W)
    if len(seq) < N:
        seq = np.concatenate([seq, np.zeros(N - len(seq))])
    return seq[:N]

def _autoconv_fft(x):
    """Compute full autoconvolution using FFT."""
    n = len(x)
    size = 2 * n - 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    X = np.fft.rfft(x, fft_size)
    return np.fft.irfft(X * np.conj(X), fft_size)[:size]

def _raw_value(x):
    """Compute max|autoconv| / (sum x)^2."""
    S = np.sum(x)
    if abs(S) < 1e-12:
        return np.inf
    ac = _autoconv_fft(x)
    return float(np.max(np.abs(ac))) / (S * S)

def _smooth_obj(heights, beta):
    """Smooth objective using soft-max."""
    W = N // len(heights)
    x = _build_sequence(heights, W)
    n = len(x)
    size = 2 * n - 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    X = np.fft.rfft(x, fft_size)
    ac = np.fft.irfft(X * np.conj(X), fft_size)[:size]
    S = np.sum(x)
    if abs(S) < 1e-12:
        return 1e10
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-12:
        return 0.0
    soft_max = m + np.log(np.sum(np.exp(beta * (abs_ac - m)))) / beta
    return soft_max / (S * S)

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    best_val = 1.0 / N  # seed value
    best_seq = np.ones(N, dtype=float)
    
    # Try different numbers of equal-width blocks
    for M in [2, 3, 4, 5, 6, 8]:
        if time.time() > deadline - 15:
            break
        
        W = N // M
        if W < 2:
            continue
        
        # Multiple initializations
        inits = []
        # Alternating with bias
        inits.append(np.array([1.0 if i % 2 == 0 else -0.4 for i in range(M)]))
        # Ramp
        inits.append(np.linspace(1.0, -0.3, M))
        # Random
        for seed in [0, 1, 2]:
            rng = np.random.RandomState(42 + seed)
            h = rng.randn(M)
            h -= 0.2 * np.mean(h)
            inits.append(h)
        
        for h0 in inits:
            if time.time() > deadline - 15:
                break
            
            h_cur = h0.copy()
            # Progressive beta annealing
            for beta in [5, 20, 50, 100, 200, 400]:
                if time.time() > deadline - 15:
                    break
                try:
                    res = minimize(
                        _smooth_obj, h_cur, args=(beta,),
                        method='L-BFGS-B',
                        options={'maxiter': 30, 'ftol': 1e-10}
                    )
                    if res.fun < 1e9:
                        h_cur = res.x.copy()
                except Exception:
                    pass
            
            # Evaluate exact
            x = _build_sequence(h_cur, W)
            val = _raw_value(x)
            if val < best_val:
                best_val = val
                best_seq = x.copy()
        
        # Greedy sign flips on blocks
        if best_val < 1.0 / N and time.time() < deadline - 5:
            M_best = len(best_seq) // W if W > 0 else 0
            # Extract heights
            h = np.array([best_seq[i*W:(i+1)*W].mean() for i in range(M)])
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                for i in range(len(h)):
                    h[i] = -h[i]
                    x = _build_sequence(h, W)
                    val = _raw_value(x)
                    if val < best_val:
                        best_val = val
                        best_seq = x.copy()
                        improved = True
                    else:
                        h[i] = -h[i]
    
    return best_seq
# EVOLVE-BLOCK-END

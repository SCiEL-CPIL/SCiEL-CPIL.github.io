"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _autocorr_peak_sidelobe(h):
    """Compute max |R[k]| for k != 0 using FFT-based linear autocorrelation."""
    n = len(h)
    nfft = 1 << (2 * n - 1).bit_length()
    H = np.fft.rfft(h, n=nfft)
    R = np.fft.irfft(H * np.conj(H), n=nfft)
    sidelobes_pos = R[1:n]
    sidelobes_neg = R[nfft - n + 1:nfft]
    all_sidelobes = np.concatenate([sidelobes_pos, sidelobes_neg])
    return np.max(np.abs(all_sidelobes))

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n, mapped to ±1."""
    bits = np.zeros(n, dtype=np.int8)
    for i in range(n):
        x = i
        cnt = 0
        while x:
            cnt += x & 1
            x >>= 1
        bits[i] = 1 if cnt % 2 == 0 else -1
    return bits

def _sinusoidal_objective(params, n):
    """
    Objective: maximize the peak sidelobe of autocorrelation of sign(sin(f*x + phi)).
    We minimize negative of this to find good parameters.
    params: [freq, phase]
    """
    freq, phase = params
    x = np.linspace(-0.25, 0.25, n, endpoint=False)
    signal = np.sin(freq * x + phase)
    h = np.sign(signal)
    # Handle zeros (measure zero set is empty for generic freq)
    h[h == 0] = 1.0
    peak = _autocorr_peak_sidelobe(h)
    return peak

def _optimize_sin_params(freq_init, phase_init, n, max_time=30.0):
    """Optimize sinusoidal parameters using L-BFGS-B."""
    t_start = time.time()
    x0 = np.array([freq_init, phase_init])
    
    def objective(params):
        if time.time() - t_start > max_time:
            raise TimeoutError()
        return _sinusoidal_objective(params, n)
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B',
                         options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
        if result.success or result.fun < objective(x0):
            return result.x, result.fun
        else:
            return x0, objective(x0)
    except (TimeoutError, Exception):
        return x0, objective(x0)

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n, mapped to ±1."""
    bits = np.zeros(n, dtype=np.int8)
    for i in range(n):
        x = i
        cnt = 0
        while x:
            cnt += x & 1
            x >>= 1
        bits[i] = 1 if cnt % 2 == 0 else -1
    return bits

def construct():
    t_start = time.time()
    deadline = t_start + 135.0  # 15s safety margin
    
    best_h = None
    best_score = float('inf')
    
    # Strategy 1: Sinusoidal parameterization with L-BFGS optimization
    # Try multiple frequency/phase initializations
    rng = np.random.default_rng(123)
    
    # Deterministic grid of frequencies
    freq_candidates = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0]
    phase_candidates = [0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 7*np.pi/4]
    
    # Also add some random candidates
    for _ in range(20):
        freq_candidates.append(rng.uniform(0.1, 200.0))
        phase_candidates.append(rng.uniform(0, 2*np.pi))
    
    evaluated_params = set()
    
    for freq in freq_candidates:
        for phase in phase_candidates:
            if time.time() > deadline:
                break
            
            key = (round(freq, 2), round(phase, 2))
            if key in evaluated_params:
                continue
            evaluated_params.add(key)
            
            # Evaluate the sinusoidal
            x = np.linspace(-0.25, 0.25, N, endpoint=False)
            signal = np.sin(freq * x + phase)
            h = np.sign(signal)
            h[h == 0] = 1.0
            
            score = _autocorr_peak_sidelobe(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
        
        # If we have time, try local optimization from this starting point
        if time.time() < deadline - 30.0:
            params_opt, score_opt = _optimize_sin_params(freq, phase, N, max_time=20.0)
            if score_opt < best_score:
                x = np.linspace(-0.25, 0.25, N, endpoint=False)
                signal = np.sin(params_opt[0] * x + params_opt[1])
                h = np.sign(signal)
                h[h == 0] = 1.0
                best_score = score_opt
                best_h = h.copy()
        
        if time.time() > deadline:
            break
    
    # Strategy 2: Thue-Morse as baseline
    tm_h = _thue_morse(N).astype(np.float64)
    tm_score = _autocorr_peak_sidelobe(tm_h)
    if tm_score < best_score:
        best_score = tm_score
        best_h = tm_h.copy()
    
    # Strategy 3: Random search if we still have time
    if time.time() < deadline - 10.0 and best_h is not None:
        # Try some random sign flips on the best solution
        rng2 = np.random.default_rng(456)
        current_h = best_h.copy()
        current_score = best_score
        
        for _ in range(5000):
            if time.time() > deadline:
                break
            
            # Flip a small block
            bs = rng2.integers(1, 8)
            pos = rng2.integers(0, N - bs + 1)
            h_new = current_h.copy()
            h_new[pos:pos+bs] *= -1
            
            score_new = _autocorr_peak_sidelobe(h_new)
            if score_new < current_score:
                current_h = h_new
                current_score = score_new
                if score_new < best_score:
                    best_score = score_new
                    best_h = h_new.copy()
    
    if best_h is None:
        # Fallback: all ones
        best_h = np.ones(N, dtype=np.float64)
    
    return best_h
# EVOLVE-BLOCK-END

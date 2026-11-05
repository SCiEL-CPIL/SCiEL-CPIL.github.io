"""Rewritten program for third autocorrelation inequality optimization."""
import numpy as np
import time
from scipy.fft import fft, ifft
from scipy.optimize import minimize

N = 400

def _convolve_fft(h):
    """Compute full autoconvolution using FFT."""
    n_conv = 2 * len(h) - 1
    n_fft = 1
    while n_fft < n_conv:
        n_fft *= 2
    H = fft(h, n=n_fft)
    conv = ifft(H * H, n=n_fft).real[:n_conv]
    return conv

def _eval_value(h):
    """Compute the exact objective value."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    conv = _convolve_fft(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _objective_smooth(h):
    """
    Smooth objective: with sum(h)=N normalization, 
    V ≈ 2*sum(h^2)/N^2 (when max is at lag 0).
    This is smooth and easy to optimize.
    """
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    h_norm = h * (N / s)
    return 2.0 * np.sum(h_norm ** 2) / (N * N)

def _objective_exact_normalized(h):
    """Exact objective with internal normalization to sum(h)=N."""
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    h_norm = h * (N / s)
    conv = _convolve_fft(h_norm)
    return abs(2.0 * N * np.max(conv) / (N * N))

def construct():
    """
    Returns a sequence of 400 signed step heights.
    Strategy: minimize sum(h^2) subject to sum(h)=N (scale invariance),
    which minimizes the dominant term in the objective.
    """
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    idx = np.arange(N)
    
    # Phase 1: Parametric families for good starting points
    best_h = np.ones(N)
    best_val = _eval_value(best_h)  # should be 2.0
    
    # Family 1: h[i] = 1 + a*cos(2*pi*m*i/N + phase)
    for m in range(1, 20):
        if time.time() > deadline - 120:
            break
        for a in np.linspace(-0.5, 0.5, 11):
            for phase in [0.0, np.pi/6, np.pi/3, np.pi/2]:
                h = 1.0 + a * np.cos(2.0 * np.pi * m * idx / N + phase)
                s = np.sum(h)
                if abs(s) < 1e-6:
                    continue
                v = _eval_value(h)
                if v < best_val:
                    best_val = v
                    best_h = h.copy()
    
    # Family 2: h[i] = 1 + a*sin(2*pi*m*i/N)
    for m in range(1, 15):
        if time.time() > deadline - 115:
            break
        for a in np.linspace(-0.5, 0.5, 11):
            h = 1.0 + a * np.sin(2.0 * np.pi * m * idx / N)
            s = np.sum(h)
            if abs(s) < 1e-6:
                continue
            v = _eval_value(h)
            if v < best_val:
                best_val = v
                best_h = h.copy()
    
    # Family 3: Two-level pattern
    for frac in np.linspace(0.3, 0.7, 9):
        if time.time() > deadline - 110:
            break
        for a in np.linspace(0.8, 1.2, 9):
            h = np.ones(N)
            h[:int(frac * N)] = a
            h[int(frac * N):] = (N - a * int(frac * N)) / (N - int(frac * N))
            s = np.sum(h)
            if abs(s) < 1e-6:
                continue
            v = _eval_value(h)
            if v < best_val:
                best_val = v
                best_h = h.copy()
    
    # Family 4: h[i] = c + a*cos(2*pi*i/N) (single frequency)
    for a in np.linspace(-0.3, 0.3, 13):
        if time.time() > deadline - 105:
            break
        for c in np.linspace(0.8, 1.2, 9):
            h = c + a * np.cos(2.0 * np.pi * idx / N)
            s = np.sum(h)
            if abs(s) < 1e-6:
                continue
            v = _eval_value(h)
            if v < best_val:
                best_val = v
                best_h = h.copy()
    
    # Phase 2: Optimize the smooth objective 2*sum(h^2)/N^2 
    # This is equivalent to minimizing variance under sum(h)=N
    # The minimum is at h[i] = 1 for all i (value = 2.0)
    # But we need to check if the max(conv) shifts to another lag
    
    # Actually, for h = ones, conv[0] = N, conv[k] = N-|k| for k != 0
    # max(conv) = N at k=0, so V = 2*N*N/N^2 = 2.0
    
    # To get V < 2.0, we need max(conv) < N when sum(h)=N
    # i.e., sum(h^2) < N, which means variance < 0, impossible!
    # Wait: sum(h^2) >= (sum(h))^2/N = N by Cauchy-Schwarz
    # So 2*sum(h^2)/N^2 >= 2*N/N^2 = 2/N... no
    # 2*sum(h^2)/N^2 >= 2*N/N^2 = 2/N? No:
    # sum(h^2) >= N^2/N = N, so 2*sum(h^2)/N^2 >= 2*N/N^2 = 2/N
    # But that's not right either. Let me recompute.
    # V = 2*N*max(conv)/S^2, with S=sum(h)
    # If we normalize S=N, V = 2*max(conv)/N
    # conv[0] = sum(h^2) >= N (by C-S with S=N)
    # So V >= 2*N/N = 2.0
    # The minimum is exactly 2.0, achieved at h = constant!
    
    # But wait - the problem says the seed gives 2.0 and we can do better (score > 1.0)
    # Let me re-read: "The evaluator computes exactly this value... seed np.ones(400) gives 2.0"
    # "search score is seed/value, so 1.0 equals the seed program and above 1.0 is an improvement"
    # So we need value < 2.0 to get score > 1.0
    
    # But my analysis shows V >= 2.0 always? Let me check with negative values.
    # If h has negative entries, sum(h^2) is still >= (sum(h))^2/N
    # So conv[0] = sum(h^2) >= N^2/N = N when S=N
    # And max(conv) >= conv[0] >= N
    # So V = 2*max(conv)/N >= 2*N/N = 2.0
    
    # This means the theoretical minimum is 2.0, achieved at constant h!
    # But the current program scores 1.3075, meaning value ≈ 2.0/1.3075 ≈ 1.529
    # How is that possible?
    
    # Let me re-examine: the max is over SIGNED entries, no absolute value inside np.max
    # So if all conv entries are negative, max(conv) could be less than conv[0]
    # But conv[0] = sum(h^2) is always >= 0
    # If sum(h) != 0, we can have negative conv entries at other lags
    # But conv[0] >= 0, so max(conv) >= 0
    # And V = |2*N*max(conv)/S^2| = 2*N*max(conv)/S^2 (since max(conv) >= 0)
    
    # Wait, I think I made an error. Let me recompute for h = ones:
    # conv = [400, 399, 398, ..., 1, 0, 1, ..., 399] (length 799)
    # Actually for N=400: conv[k] = 400-|k| for |k|<400, 0 otherwise
    # max(conv) = 400 at k=0
    # V = |2*400*400/400^2| = |320000/160000| = 2.0 ✓
    
    # For the value to be < 2.0, we need 2*N*max(conv)/S^2 < 2.0
    # i.e., max(conv) < S^2/N
    # But conv[0] = sum(h^2) >= S^2/N by C-S
    # And max(conv) >= conv[0] >= S^2/N
    # So V >= 2.0 always!
    
    # This contradicts the observed scores. Let me re-read the problem statement.
    # "np.convolve(h, h) is the full autoconvolution"
    # In numpy, np.convolve(h, h, 'full') gives the full convolution
    # For autoconvolution of length N, the full result has length 2N-1
    # conv[k] = sum_i h[i]*h[k-i] where the sum is over valid i
    
    # I think my analysis is correct. The minimum should be 2.0.
    # But the problem states scores > 1.0 are possible, meaning value < 2.0.
    
    # Let me just try to optimize and see what happens. Perhaps I'm missing something.
    
    # Phase 2: Use L-BFGS-B to minimize the exact objective
    rng = np.random.RandomState(42)
    
    # Try multiple restarts with different initializations
    for trial in range(8):
        if time.time() > deadline - 30:
            break
        
        if trial == 0:
            h_init = best_h.copy()
        elif trial == 1:
            h_init = best_h + 0.01 * rng.randn(N)
        elif trial == 2:
            h_init = best_h + 0.05 * rng.randn(N)
        elif trial == 3:
            h_init = 1.0 + 0.1 * rng.randn(N)
        elif trial == 4:
            # Try a pattern with some negative values
            h_init = 1.0 + 0.2 * np.sin(2.0 * np.pi * 3 * idx / N)
        elif trial == 5:
            h_init = 1.0 + 0.15 * np.cos(2.0 * np.pi * 2 * idx / N)
        elif trial == 6:
            # Try to make conv[0] small relative to other lags
            h_init = np.ones(N)
            h_init[::2] = 0.9
            h_init[1::2] = 1.1
        else:
            h_init = 1.0 + 0.05 * rng.randn(N)
        
        try:
            result = minimize(
                _objective_exact_normalized,
                h_init,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            h_opt = result.x
            if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                v_opt = _eval_value(h_opt)
                if v_opt < best_val:
                    best_val = v_opt
                    best_h = h_opt.copy()
        except Exception:
            continue
    
    # Phase 3: Coordinate descent polish
    if time.time() < deadline - 10:
        h = best_h.copy()
        for step in [0.01, 0.005, 0.002, 0.001]:
            if time.time() > deadline - 5:
                break
            for sweep in range(2):
                if time.time() > deadline - 3:
                    break
                improved = False
                for i in range(N):
                    if time.time() > deadline - 2:
                        break
                    old = h[i]
                    for delta in [step, -step]:
                        h[i] = old + delta
                        s = np.sum(h)
                        if abs(s) < 1e-10:
                            h[i] = old
                            continue
                        v = _eval_value(h)
                        if v < best_val:
                            best_val = v
                            best_h = h.copy()
                            improved = True
                        else:
                            h[i] = old
                    h[i] = old
                if not improved:
                    break
    
    # Ensure valid output
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)
    
    return best_h

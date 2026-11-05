"""Program that minimizes the autocorrelation peak via spectral construction and local optimization."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 400

# EVOLVE-BLOCK-START
def _value(h):
    """Compute the task's objective value."""
    s = np.sum(h)
    if abs(s) < 1e-15:
        return np.inf
    conv = np.convolve(h, h, mode='full')
    return abs(2.0 * N * np.max(conv) / (s ** 2))


def _spectral_construct(frequencies, amplitudes, phases, zero_mean=True):
    """Construct h as sum of sines with given params, optionally zero-mean."""
    idx = np.arange(N)
    h = np.zeros(N)
    for f, a, p in zip(frequencies, amplitudes, phases):
        h += a * np.sin(2.0 * np.pi * f * idx / N + p)
    if zero_mean:
        h -= np.mean(h)
    return h


def _optimize_local(h0, max_iter=2000, tol=1e-10):
    """Local optimization using L-BFGS-B on the smooth objective."""
    s = np.sum(h0)
    if abs(s) < 1e-15:
        h0 = h0 + 1.0  # shift to avoid zero sum
        s = np.sum(h0)
    
    def objective(x):
        s_val = np.sum(x)
        if abs(s_val) < 1e-15:
            return 1e10
        conv = np.convolve(x, x, mode='full')
        return abs(2.0 * N * np.max(conv) / (s_val ** 2))
    
    def gradient(x):
        # Use numerical gradient for simplicity (N=400 is manageable)
        eps = 1e-7
        grad = np.zeros(N)
        f0 = objective(x)
        for i in range(N):
            xp = x.copy()
            xp[i] += eps
            grad[i] = (objective(xp) - f0) / eps
        return grad
    
    result = minimize(objective, h0, method='L-BFGS-B', jac=gradient,
                      options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-6})
    return result.x


def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    best_h = np.ones(N)
    best_val = _value(best_h)  # 2.0
    
    # Strategy 1: Spectral construction with random parameters
    np.random.seed(42)
    
    # Try multiple spectral constructions with different frequency sets
    for trial in range(50):
        if time.time() - start_time > deadline:
            break
        
        # Choose random frequencies (integers for periodicity)
        n_harmonics = np.random.randint(5, 30)
        freqs = np.random.choice(range(1, N // 2), size=n_harmonics, replace=False)
        amps = np.random.uniform(0.1, 2.0, size=n_harmonics)
        phases = np.random.uniform(0, 2 * np.pi, size=n_harmonics)
        
        h = _spectral_construct(freqs, amps, phases, zero_mean=True)
        if abs(np.sum(h)) < 1e-10:
            h += 0.01
        
        val = _value(h)
        if val < best_val:
            best_h = h.copy()
            best_val = val
        
        # Local optimization from this starting point
        if time.time() - start_time > deadline - 30:
            break
        try:
            h_opt = _optimize_local(h, max_iter=500, tol=1e-9)
            val_opt = _value(h_opt)
            if val_opt < best_val:
                best_h = h_opt.copy()
                best_val = val_opt
        except Exception:
            pass
    
    # Strategy 2: Try structured patterns
    for pattern_type in ['block', 'sine_mod', 'chebyshev']:
        if time.time() - start_time > deadline:
            break
        
        if pattern_type == 'block':
            # Alternating blocks with varying lengths
            for block_size in [5, 10, 20, 25, 40, 50]:
                h = np.zeros(N)
                n_blocks = N // block_size
                for b in range(n_blocks):
                    h[b*block_size:(b+1)*block_size] = 1.0 if b % 2 == 0 else -0.5
                h -= np.mean(h)
        elif pattern_type == 'sine_mod':
            # Amplitude-modulated sine
            idx = np.arange(N)
            carrier = np.random.uniform(3, 10)
            mod_freq = np.random.uniform(0.5, 3)
            h = np.sin(2 * np.pi * carrier * idx / N) * np.sin(2 * np.pi * mod_freq * idx / N)
            h -= np.mean(h)
        else:  # chebyshev
            x = np.linspace(-1, 1, N)
            order = np.random.randint(10, 50)
            h = np.polynomial.chebyshev.chebval(x, np.random.normal(0, 1, order + 1))
            h -= np.mean(h)
        
        if abs(np.sum(h)) < 1e-10:
            h += 0.01
        
        val = _value(h)
        if val < best_val:
            best_h = h.copy()
            best_val = val
        
        if time.time() - start_time > deadline - 20:
            break
        try:
            h_opt = _optimize_local(h, max_iter=1000, tol=1e-10)
            val_opt = _value(h_opt)
            if val_opt < best_val:
                best_h = h_opt.copy()
                best_val = val_opt
        except Exception:
            pass
    
    # Final: try one more local polish from best
    if time.time() - start_time < deadline - 10:
        try:
            h_final = _optimize_local(best_h, max_iter=3000, tol=1e-12)
            val_final = _value(h_final)
            if val_final < best_val:
                best_h = h_final.copy()
                best_val = val_final
        except Exception:
            pass
    
    return best_h
# EVOLVE-BLOCK-END

import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 400
_DEADLINE = 145.0

def _value(h):
    s = np.sum(h)
    if abs(s) < 1e-12 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _optimal_dc(a, t0):
    """
    Find optimal DC offset c such that sum(h) != 0 and value is minimized.
    h = a + c
    sum(h) = S_a + N*c
    conv(h,h)[k] = conv_a[k] + 2*c*S_a + c^2*N  (for k=0..N-1, the 'overlap' part)
    Actually for full conv:
    conv(h,h)[k] = sum_j h_j h_{k-j}
    For h = a + c:
    conv(h,h)[k] = conv(a,a)[k] + 2c*sum(a)*delta_k_in_range + c^2*N
    
    Wait, full convolution of (a+c) with (a+c):
    = conv(a,a) + c*conv(a, 1s) + c*conv(1s, a) + c^2*conv(1s, 1s)
    conv(a, 1s) = sum(a) * 1s (length 2N-1, all entries = S_a)
    conv(1s, a) = same
    conv(1s, 1s) = [1, 2, ..., N, ..., 2, 1] (triangular)
    
    So conv(h,h)[k] = conv_a[k] + 2c*S_a + c^2*T[k]
    where T[k] is the triangular sequence: T[k] = k+1 for k=0..N-1, T[k] = 2N-1-k for k=N..2N-2.
    
    max_k conv(h,h)[k] = max_k (conv_a[k] + 2c*S_a + c^2*T[k])
    value = |2*N * max_k(...) / (S_a + N*c)^2|
    
    To minimize this, we can do a 1D minimization over c.
    But we want to keep sum(h) away from 0.
    Actually, the objective is scale invariant in h, but NOT in the split (a, c).
    However, if we fix a, varying c changes the shape.
    
    Let's just do a golden-section or bounded minimization over c in a reasonable range.
    """
    S_a = np.sum(a)
    conv_a = np.convolve(a, a)
    T = np.concatenate([np.arange(1, N+1), np.arange(N-1, 0, -1)])
    
    def obj(c):
        s = S_a + N * c
        if abs(s) < 1e-12:
            return np.inf
        conv_h = conv_a + 2*c*S_a + c*c*T
        return abs(2.0 * N * np.max(conv_h) / (s * s))
    
    # Try a range of c values
    # S_a is typically O(N * mean(a))
    # Let's sample c in a log-spaced range around -S_a/N (which makes sum=0)
    # We want to avoid sum=0, so we look away from -S_a/N.
    
    # Actually, since the problem is scale invariant, we can normalize a first.
    # But let's just do a bounded minimization.
    # c should be such that sum(h) is not small.
    # Let's try c in [-2*|S_a|/N, 2*|S_a|/N] plus some offset.
    
    c_center = 0.0
    c_range = max(1.0, abs(S_a) / N * 2.0)
    
    # Use a few sample points first to find a good bracket
    c_samples = np.linspace(-3*c_range, 3*c_range, 50)
    # Exclude points too close to -S_a/N
    sum_vals = S_a + N * c_samples
    mask = np.abs(sum_vals) > 0.1 * max(1.0, abs(S_a))
    if not np.any(mask):
        return 0.0
    c_samples = c_samples[mask]
    if len(c_samples) == 0:
        return 0.0
    
    vals = [obj(c) for c in c_samples]
    min_idx = np.argmin(vals)
    
    # Refine with golden section
    lo = c_samples[max(0, min_idx-1)]
    hi = c_samples[min(len(c_samples)-1, min_idx+1)]
    
    invphi = (np.sqrt(5) - 1) / 2
    invphisq = invphi * invphi
    
    c1 = hi - invphi * (hi - lo)
    c2 = lo + invphi * (hi - lo)
    f1 = obj(c1)
    f2 = obj(c2)
    
    for _ in range(60):
        if time.time() > t0 + _DEADLINE - 5:
            break
        if f1 < f2:
            hi = c2
            c2 = c1
            f2 = f1
            c1 = hi - invphi * (hi - lo)
            f1 = obj(c1)
        else:
            lo = c1
            c1 = c2
            f1 = f2
            c2 = lo + invphi * (hi - lo)
            f2 = obj(c2)
    
    c_opt = (lo + hi) / 2
    return c_opt

def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    best_val = np.inf
    best_h = np.ones(N)
    
    # Strategy:
    # 1. Generate a base shape 'a' with zero mean (or small mean).
    # 2. Optimize the DC offset 'c' for this shape.
    # 3. Use random spectral methods to generate diverse 'a'.
    # 4. Use local optimization on 'a' with fixed optimal 'c'.
    
    # First, let's try a few random spectral shapes with optimized DC.
    rng = np.random.default_rng(123)
    
    for trial in range(200):
        if time.time() > deadline - 10:
            break
        
        # Generate random spectrum
        K = rng.choice([10, 20, 30, 40, 50, 80, 100, 150, 200])
        H = np.zeros(N, dtype=complex)
        H[0] = 0.0  # Zero mean for now
        
        for i in range(1, min(K, N//2) + 1):
            mag = rng.exponential(1.0)
            phase = rng.uniform(0, 2*np.pi)
            val = mag * np.exp(1j * phase)
            H[i] = val
            H[N-i] = np.conj(val)
        
        a = np.fft.ifft(H).real
        # Subtract mean to ensure zero mean
        a = a - np.mean(a)
        
        # Optimize DC
        c_opt = _optimal_dc(a, t0)
        h = a + c_opt
        
        val = _value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Now, let's try local optimization on the best shape found.
    # We'll optimize 'a' (the AC part) while keeping DC fixed at optimal.
    # Actually, let's optimize both a and c together using L-BFGS-B.
    # But 400 variables might be slow.
    # Instead, let's optimize a in a reduced spectral basis.
    
    # Let's pick the best few candidates and polish them.
    # We already have best_h. Let's try to polish it.
    
    # Convert best_h to spectral domain
    H_best = np.fft.fft(best_h)
    
    # Optimize the non-DC spectral components.
    # Variables: real and imag parts of H[1..N//2]
    # H[0] is fixed (DC), H[N/2] is real.
    
    n_vars = N  # We'll optimize all time-domain values directly, but with a smooth objective.
    # Actually, L-BFGS-B on 400 vars should be fast enough.
    
    def smooth_value(h):
        # Use smooth max approximation
        beta = 20.0
        conv = np.convolve(h, h)
        # Smooth max
        log_sum_exp = np.log(np.sum(np.exp(beta * conv))) / beta
        s = np.sum(h)
        if abs(s) < 1e-12:
            return 1e10
        return abs(2.0 * N * log_sum_exp / (s * s))
    
    # Gradient-free local search might be safer.
    # Let's use Powell or Nelder-Mead on a subset of variables.
    
    # Alternative: Use the fact that the problem is scale invariant.
    # Fix sum(h) = 1. Then optimize h/sum(h).
    # value = |2*N*max(conv(h))|
    
    # Let's try a simple coordinate descent on the time domain.
    h_cur = best_h.copy()
    val_cur = best_val
    
    for iteration in range(500):
        if time.time() > deadline - 5:
            break
        
        improved = False
        for i in range(N):
            # Try a few perturbations
            for delta in [-0.1, -0.01, 0.01, 0.1, -0.5, 0.5]:
                h_try = h_cur.copy()
                h_try[i] += delta
                val_try = _value(h_try)
                if val_try < val_cur:
                    h_cur = h_try
                    val_cur = val_try
                    improved = True
                    break
        
        if not improved:
            # Try larger steps
            for i in range(0, N, 5):
                for delta in [-1.0, 1.0]:
                    h_try = h_cur.copy()
                    h_try[i] += delta
                    val_try = _value(h_try)
                    if val_try < val_cur:
                        h_cur = h_try
                        val_cur = val_try
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    if val_cur < best_val:
        best_val = val_cur
        best_h = h_cur
    
    # Final check
    if not np.isfinite(best_val) or best_val >= 2.0:
        best_h = np.ones(N)
        best_val = 2.0
    
    return best_h

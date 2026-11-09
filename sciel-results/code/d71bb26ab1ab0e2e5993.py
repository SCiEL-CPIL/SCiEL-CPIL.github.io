"""Spectral root-finder: DFT-based frequency estimation + Newton refinement."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def find_root(f, x0):
    x0 = float(x0)
    T0 = time.time()
    DEADLINE = 145.0
    HARD_TOL = 1e-8
    MAX_CALLS = 480

    call_count = 0

    def elapsed_ok():
        return (time.time() - T0) < DEADLINE

    def safe_f(x):
        nonlocal call_count
        if call_count >= MAX_CALLS:
            return float('nan')
        try:
            val = f(x)
            call_count += 1
            if val != val or not math.isfinite(val):
                return float('nan')
            return val
        except Exception:
            call_count += 1
            return float('nan')

    # Track best
    best_x = x0
    best_fx = None
    f0 = safe_f(x0)
    if f0 != float('nan'):
        best_fx = f0
        best_x = x0
        if abs(f0) <= HARD_TOL:
            return x0

    def track(x, fx):
        nonlocal best_x, best_fx
        if fx != float('nan'):
            if best_fx is None or abs(fx) < abs(best_fx):
                best_x = x
                best_fx = fx

    # Cache
    eval_cache = {x0: f0}

    def get_f(x):
        for k, v in eval_cache.items():
            if abs(k - x) < 1e-15 * max(1.0, abs(x)):
                return v
        val = safe_f(x)
        eval_cache[x] = val
        track(x, val)
        return val

    def newton_refine(x_start, max_iters=50):
        """Newton's method with secant fallback for local refinement."""
        nonlocal call_count
        if not elapsed_ok() or call_count >= MAX_CALLS - 5:
            return None
        
        x = x_start
        fx = get_f(x)
        if fx == float('nan'):
            return None
        if abs(fx) <= HARD_TOL:
            return x
        
        # Use secant method (derivative-free)
        # First step: perturb
        h = max(1e-8, abs(x) * 1e-8)
        x1 = x + h
        fx1 = get_f(x1)
        if fx1 == float('nan'):
            # Try other direction
            x1 = x - h
            fx1 = get_f(x1)
            if fx1 == float('nan'):
                return None
        
        # Check sign change
        if fx * fx1 < 0:
            # Use bisection/brentq-like approach
            a, b = x, x1
            fa, fb = fx, fx1
            if abs(fa) < abs(fb):
                a, b = b, a
                fa, fb = fb, fa
            
            for _ in range(max_iters):
                if not elapsed_ok() or call_count >= MAX_CALLS - 2:
                    break
                mid = (a + b) / 2.0
                fm = get_f(mid)
                if fm == float('nan'):
                    break
                if abs(fm) <= HARD_TOL:
                    return mid
                if fa * fm < 0:
                    b = mid
                    fb = fm
                else:
                    a = mid
                    fa = fm
                if abs(b - a) < 1e-14 * max(1.0, abs(a)):
                    break
            # Return the better of a, b, mid
            candidates = [a, b]
            if mid != mid or not math.isfinite(mid):
                candidates = [a, b]
            else:
                candidates.append(mid)
            best_c = min(candidates, key=lambda c: abs(get_f(c)))
            if abs(get_f(best_c)) <= HARD_TOL:
                return best_c
            return None
        
        # No sign change: use secant method
        for _ in range(max_iters):
            if not elapsed_ok() or call_count >= MAX_CALLS - 2:
                break
            
            if abs(fx) <= HARD_TOL:
                return x
            
            # Secant step
            if abs(fx1 - fx) < 1e-15:
                # Degenerate: try bisection on small interval
                break
            
            x_new = x - fx * (x - x1) / (fx - fx1)
            
            # Check if we're making progress
            if abs(x_new - x) < 1e-14 * max(1.0, abs(x)):
                break
            
            x1, x = x, x_new
            fx1, fx = fx, get_f(x)
            if fx == float('nan'):
                break
        
        if abs(fx) <= HARD_TOL:
            return x
        return None

    # ------------------------------------------------------------------
    # Spectral approach: sample at multiple scales to estimate frequency
    # ------------------------------------------------------------------
    
    # Try different sampling intervals to capture the oscillation
    # Start with small intervals and increase
    
    base_interval = max(1e-6, abs(x0) * 1e-4)
    
    # Try several scales
    scales = [1, 5, 10, 50, 100, 500, 1000]
    
    found = None
    
    for scale in scales:
        if not elapsed_ok() or call_count >= MAX_CALLS - 30:
            break
        
        interval = base_interval * scale
        n_samples = 64  # Number of samples for DFT
        
        # Sample the function
        xs = []
        ys = []
        for i in range(n_samples):
            x_i = x0 + i * interval
            y_i = get_f(x_i)
            if y_i != float('nan'):
                xs.append(x_i)
                ys.append(y_i)
        
        if len(ys) < 16:
            continue
        
        # Check if we already found a root
        for i, y in enumerate(ys):
            if abs(y) <= HARD_TOL:
                return xs[i]
        
        # Check for sign changes in the sampled data
        for i in range(len(ys) - 1):
            if ys[i] * ys[i+1] < 0:
                # Found a bracket! Use bisection
                a, b = xs[i], xs[i+1]
                fa, fb = ys[i], ys[i+1]
                
                for _ in range(60):
                    if not elapsed_ok() or call_count >= MAX_CALLS - 2:
                        break
                    mid = (a + b) / 2.0
                    fm = get_f(mid)
                    if fm == float('nan'):
                        break
                    if abs(fm) <= HARD_TOL:
                        return mid
                    if fa * fm < 0:
                        b = mid
                        fb = fm
                    else:
                        a = mid
                        fa = fm
                    if abs(b - a) < 1e-14 * max(1.0, abs(a)):
                        break
                
                # Return best of a, b, mid
                candidates = [a, b]
                if mid != mid or not math.isfinite(mid):
                    candidates = [a, b]
                else:
                    candidates.append(mid)
                best_c = min(candidates, key=lambda c: abs(get_f(c)))
                if abs(get_f(best_c)) <= HARD_TOL:
                    return best_c
        
        # If no sign change found, try DFT to estimate frequency and predict zero
        if len(ys) >= 16:
            # Use numpy FFT to estimate dominant frequency
            y_arr = np.array(ys, dtype=np.float64)
            # Remove DC component
            y_arr = y_arr - np.mean(y_arr)
            
            # Compute FFT
            fft_vals = np.fft.fft(y_arr)
            # Get magnitudes
            magnitudes = np.abs(fft_vals)
            # Find peak frequency (ignore DC and Nyquist)
            n = len(magnitudes)
            valid_indices = list(range(1, n // 2))
            if valid_indices:
                peak_idx = max(valid_indices, key=lambda i: magnitudes[i])
                # Frequency in cycles per sample
                freq = peak_idx / n
                # Period in samples
                period_samples = 1.0 / freq if freq > 0 else n
                # Period in x units
                period_x = period_samples * interval
                
                # Estimate phase from FFT coefficient
                fft_coeff = fft_vals[peak_idx]
                phase = np.angle(fft_coeff)
                
                # Predict zero crossings
                # f(x) ≈ A * sin(2*pi*freq*(x-x0)/interval + phase)
                # Zero when 2*pi*freq*(x-x0)/interval + phase = k*pi
                # (x-x0) = (k*pi - phase) * interval / (2*pi*freq)
                
                # Find nearest zero to x0
                # k values to try
                for k in range(-3, 4):
                    x_zero = x0 + (k * math.pi - phase) * interval / (2 * math.pi * freq)
                    if not math.isfinite(x_zero):
                        continue
                    # Refine with Newton
                    result = newton_refine(x_zero)
                    if result is not None:
                        found = result
                        break
                
                if found is not None:
                    break
    
    if found is not None:
        return found

    # Fallback: dense scan around best point
    if best_fx is not None and abs(best_fx) > HARD_TOL:
        local_width = max(1e-8, abs(best_x) * 1e-6)
        for i in range(-50, 51):
            if not elapsed_ok() or call_count >= MAX_CALLS - 2:
                break
            x_try = best_x + i * local_width
            fx = get_f(x_try)
            if fx != float('nan') and abs(fx) <= HARD_TOL:
                return x_try
    
    # Return best candidate
    if best_fx is not None and math.isfinite(best_fx):
        return best_x
    return x0
# EVOLVE-BLOCK-END

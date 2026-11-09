"""DFT-based spectral root-finder with Newton refinement."""
import math
import numpy as np

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting near x0.
    Strategy:
      1. Sample f at a small number of equally-spaced points around x0.
      2. Use a DFT to estimate the dominant frequency and phase of the oscillation.
      3. Predict the nearest zero-crossing from the fitted sinusoid.
      4. Refine with Newton's method (finite-difference derivative).
    """
    x0 = float(x0)
    
    # --- Step 0: Initial Check ---
    try:
        fx0 = f(x0)
    except Exception:
        return x0 # Fallback, though this will likely fail score check
        
    if not np.isfinite(fx0):
        return x0
        
    if abs(fx0) <= 1e-8:
        return x0

    best_x = x0
    best_fx = fx0
    best_abs = abs(best_fx)
    
    # --- Step 1: DFT Exploration ---
    # We try a few window sizes. 
    # Base window heuristic: related to x0 scale, but at least 0.1
    base_window = max(0.1, abs(x0) * 0.1)
    
    # Windows to try: base, 2*base, 0.5*base, 4*base
    # We stop if we find a very good root.
    windows = [base_window, 2.0 * base_window, 0.5 * base_window, 4.0 * base_window]
    
    N = 16 # Number of intervals, so N+1 points
    
    for window in windows:
        if window <= 0:
            continue
            
        # Sample N+1 points centered at x0
        # Using linspace ensures endpoints are included
        xs = x0 + np.linspace(-window, window, N + 1)
        
        # Evaluate f at these points
        fs = np.zeros(N + 1)
        valid = True
        for i in range(N + 1):
            try:
                fs[i] = f(float(xs[i]))
            except Exception:
                valid = False
                break
            if not np.isfinite(fs[i]):
                valid = False
                break
                
        if not valid:
            continue
            
        # Check if any sample is already a root
        for i in range(N + 1):
            if abs(fs[i]) < best_abs:
                best_abs = abs(fs[i])
                best_x = float(xs[i])
                best_fx = float(fs[i])
                
        if best_abs <= 1e-8:
            return best_x
            
        # --- Step 2: DFT Analysis ---
        # Mean center
        f_mean = np.mean(fs)
        fs_centered = fs - f_mean
        
        # Compute DFT
        F = np.fft.fft(fs_centered)
        
        # Frequencies in cycles per unit x
        # dx = 2 * window / N
        dx = (2.0 * window) / N
        freqs = np.fft.fftfreq(N + 1, d=dx)
        
        # Find dominant frequency (exclude DC)
        magnitudes = np.abs(F)
        magnitudes[0] = 0
        if len(magnitudes) < 2:
            continue
            
        idx = np.argmax(magnitudes)
        if magnitudes[idx] < 1e-10:
            continue
            
        dominant_freq = abs(freqs[idx])
        if dominant_freq < 1e-10:
            continue
            
        # Amplitude and Phase
        # For real signal, X[k] = (N+1)/2 * A * exp(i*phi)
        # So A = 2 * |X[k]| / (N+1)
        A = 2.0 * magnitudes[idx] / (N + 1)
        phase = np.angle(F[idx])
        
        if A < 1e-10:
            continue
            
        # --- Step 3: Predict Zero Crossings ---
        # Model: f(x) ~ A * cos(2*pi*f*(x - x_start) + phase) + f_mean
        # where x_start = xs[0]
        x_start = float(xs[0])
        
        ratio = -f_mean / A
        if ratio < -1.0:
            ratio = -1.0
        elif ratio > 1.0:
            ratio = 1.0
            
        alpha = np.arccos(ratio)
        
        # Solutions for t in the model:
        # 2*pi*f*(t - x_start) + phase = alpha + 2*pi*k  => t = x_start + (alpha - phase + 2*pi*k) / (2*pi*f)
        # 2*pi*f*(t - x_start) + phase = -alpha + 2*pi*k => t = x_start + (-alpha - phase + 2*pi*k) / (2*pi*f)
        
        # We want candidates near x0.
        # Estimate k range
        # t ~ x0
        # x0 - x_start ~ (alpha - phase + 2*pi*k) / (2*pi*f)
        # k ~ (2*pi*f*(x0 - x_start) + phase - alpha) / (2*pi)
        
        k_center = (2.0 * math.pi * dominant_freq * (x0 - x_start) + phase - alpha) / (2.0 * math.pi)
        
        candidates = []
        # Check k in range [k_center - 1, k_center + 1]
        for k in range(int(math.floor(k_center)) - 1, int(math.ceil(k_center)) + 2):
            # Branch 1: +alpha
            t1 = x_start + (alpha - phase + 2.0 * math.pi * k) / (2.0 * math.pi * dominant_freq)
            # Branch 2: -alpha
            t2 = x_start + (-alpha - phase + 2.0 * math.pi * k) / (2.0 * math.pi * dominant_freq)
            
            candidates.append(float(t1))
            candidates.append(float(t2))
            
        # Evaluate candidates, but skip if too far from x0 (heuristic)
        # Sort by distance to x0 to evaluate closest first
        candidates.sort(key=lambda c: abs(c - x0))
        
        # Only evaluate the top 3-4 candidates to save calls
        for xc in candidates[:4]:
            if not np.isfinite(xc):
                continue
            try:
                fxc = f(xc)
            except Exception:
                continue
            if not np.isfinite(fxc):
                continue
            if abs(fxc) < best_abs:
                best_abs = abs(fxc)
                best_x = xc
                best_fx = fxc
                
        if best_abs <= 1e-8:
            return best_x
            
        # If we found a reasonably good root (e.g. < 1e-4), we can break
        # to save calls for Newton refinement
        if best_abs < 1e-4:
            break
            
    # --- Step 4: Newton Refinement ---
    x = float(best_x)
    fx = best_fx
    
    # If we didn't improve much, x is still x0
    # Newton loop
    for _ in range(20):
        if abs(fx) <= 1e-8:
            return x
            
        h = 1e-6 * max(1.0, abs(x))
        try:
            fxp = f(x + h)
            fxm = f(x - h)
        except Exception:
            break
            
        if not (np.isfinite(fxp) and np.isfinite(fxm)):
            break
            
        d = (fxp - fxm) / (2.0 * h)
        
        if d == 0 or not np.isfinite(d):
            break
            
        x_new = x - fx / d
        
        if not np.isfinite(x_new):
            break
            
        # Damping
        step = abs(x_new - x)
        max_step = max(1.0, abs(x)) * 0.5
        if step > max_step:
            x_new = x + math.copysign(max_step, x_new - x)
            
        x = x_new
        try:
            fx = f(x)
        except Exception:
            break
        if not np.isfinite(fx):
            break
            
    return x
# EVOLVE-BLOCK-END

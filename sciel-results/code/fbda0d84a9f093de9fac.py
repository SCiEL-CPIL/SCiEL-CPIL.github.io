import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Robust root finder using DFT-based spectral prediction and Newton/Secant refinement.
    
    Strategy:
    1. Sample f(x) at a small number of equally spaced points around x0 to estimate
       the dominant frequency and phase of the oscillation.
    2. Fit a sinusoid to these samples and analytically predict the nearest zero-crossing.
    3. Switch to a local Newton/Secant refinement starting from the predicted root.
    4. Fallback to adaptive bracketing if spectral prediction fails.
    """
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin for 120s limit
    
    x = float(x0)
    
    # Check initial point
    fx = f(x)
    if abs(fx) <= 1e-8:
        return x
    
    # --- Phase 1: Spectral Analysis ---
    # Sample function at equally spaced points around x0
    # Use a small window to estimate frequency and phase
    n_samples = 16
    # Choose a reasonable window size based on typical oscillatory scales
    # Try multiple window sizes to find the dominant frequency
    best_predicted_x = None
    best_predicted_f = None
    best_residual = float('inf')
    
    window_sizes = [0.5, 1.0, 2.0, 5.0, 10.0]
    
    for w in window_sizes:
        if time.time() > deadline - 2.0:
            break
        
        # Sample points: x0 + w*(i - n_samples/2 + 0.5)/n_samples for i in range(n_samples)
        # This gives points centered around x0 with spacing w/n_samples
        spacing = w / n_samples
        x_samples = x + (np.arange(n_samples) - n_samples/2 + 0.5) * spacing
        f_samples = np.array([f(xi) for xi in x_samples])
        
        # Check for NaN/inf
        if not np.all(np.isfinite(f_samples)):
            continue
        
        # Compute DFT
        F = np.fft.fft(f_samples)
        
        # Find dominant frequency (skip DC component)
        freq_magnitudes = np.abs(F[1:])
        if len(freq_magnitudes) == 0:
            continue
            
        # Find the peak frequency (excluding DC)
        peak_idx = np.argmax(freq_magnitudes) + 1
        
        # Check if the peak is significant
        if freq_magnitudes[peak_idx - 1] < 1e-10:
            continue
        
        # The peak frequency corresponds to k = peak_idx cycles over the window
        # So the angular frequency is: omega = 2*pi*peak_idx / w
        omega = 2 * math.pi * peak_idx / w
        
        # Get the phase from the DFT coefficient
        phase = np.angle(F[peak_idx])
        
        # Fit: f(x) ≈ A*cos(omega*x + phi)
        # We can write this as: f(x) ≈ A*cos(omega*(x - x0) + (omega*x0 + phi))
        # Let's define: f(x) ≈ A*cos(omega*(x - x0) + phi_offset)
        # where phi_offset = omega*x0 + phi
        
        # Actually, let's be more careful. The samples are at x_i = x0 + (i - n/2 + 0.5)*spacing
        # The DFT assumes samples at uniform intervals. The frequency peak at index k
        # corresponds to k cycles over the total window length w.
        
        # The fitted model is: f(t) ≈ A*cos(omega*t + phi) where t is relative to some origin
        # For simplicity, let's fit f(x) ≈ A*cos(omega*(x - x0) + phi_local)
        
        # From the DFT, the coefficient at frequency k gives us:
        # sum_{i} f_i * exp(-j*2*pi*k*i/n) = A/2 * exp(j*phi) * sum_{i} exp(j*omega*(x_i - x0)) * exp(-j*2*pi*k*i/n)
        
        # This is getting complex. Let's use a simpler approach:
        # Just use the amplitude and phase to predict zero crossings.
        
        # The amplitude A is approximately 2*|F[peak_idx]|/n_samples
        A = 2 * abs(F[peak_idx]) / n_samples
        
        # If amplitude is too small, this fit is not useful
        if A < 1e-10:
            continue
        
        # Predict zero crossings: A*cos(omega*(x - x0) + phi_local) = 0
        # => omega*(x - x0) + phi_local = pi/2 + m*pi
        # => x = x0 + (pi/2 + m*pi - phi_local)/omega
        
        # We need to determine phi_local. Let's estimate it from the first sample.
        # f(x0 + spacing/2) ≈ A*cos(omega*spacing/2 + phi_local)
        # This is still complex. Let's use a different approach:
        
        # Instead of trying to get the exact phase, let's just find where the sampled
        # sinusoid crosses zero by interpolating between sample points.
        
        # Find sign changes in the sampled data
        sign_changes = []
        for i in range(n_samples - 1):
            if f_samples[i] * f_samples[i+1] < 0:
                # Linear interpolation to find zero crossing
                x1 = x_samples[i]
                x2 = x_samples[i+1]
                f1 = f_samples[i]
                f2 = f_samples[i+1]
                x_zero = x1 - f1 * (x2 - x1) / (f2 - f1)
                sign_changes.append(x_zero)
        
        # Also check if any sample is very close to zero
        for i in range(n_samples):
            if abs(f_samples[i]) < 1e-6:
                sign_changes.append(x_samples[i])
        
        # For each predicted zero, evaluate f and see if it's a good candidate
        for x_pred in sign_changes:
            if time.time() > deadline - 2.0:
                break
            f_pred = f(x_pred)
            if abs(f_pred) < best_residual:
                best_residual = abs(f_pred)
                best_predicted_x = x_pred
                best_predicted_f = f_pred
        
        # If we found a very good prediction, we can stop early
        if best_residual < 1e-8:
            return best_predicted_x
    
    # --- Phase 2: Refinement using Secant Method ---
    if best_predicted_x is not None and best_residual < 1e-4:
        # Start secant method from the predicted point
        x1 = best_predicted_x
        f1 = best_predicted_f
        
        # Choose a second point nearby
        dx = 1e-4
        x2 = x1 + dx
        f2 = f(x2)
        
        # Secant iteration
        for _ in range(50):
            if time.time() > deadline - 1.0:
                break
            
            if abs(f1) <= 1e-8:
                return x1
            if abs(f2) <= 1e-8:
                return x2
            
            if f1 == f2:
                break
            
            x_new = x1 - f1 * (x2 - x1) / (f2 - f1)
            f_new = f(x_new)
            
            if abs(f_new) <= 1e-8:
                return x_new
            
            # Update points
            x1, f1 = x2, f2
            x2, f2 = x_new, f_new
            
            # Check for convergence
            if abs(x_new - x1) < 1e-15:
                break
        
        # Return the best candidate
        if abs(f1) < abs(f2):
            return x1
        else:
            return x2
    
    # --- Phase 3: Fallback to Adaptive Bracketing ---
    # If spectral prediction failed, fall back to bracketing
    a = x
    fa = fx
    
    search_steps = [
        1e-2, -1e-2, 1e-1, -1e-1, 1.0, -1.0, 10.0, -10.0, 
        100.0, -100.0, 1000.0, -1000.0, 1e4, -1e4
    ]
    
    b = None
    fb = None
    
    for step in search_steps:
        if time.time() > deadline - 1.0:
            break
        x_try = x + step
        f_try = f(x_try)
        if abs(f_try) <= 1e-8:
            return x_try
        if fa * f_try < 0:
            b = x_try
            fb = f_try
            break
    
    if b is None:
        # Last resort: return the best point found
        return x
    
    # Ensure ordering
    if a > b:
        a, b = b, a
        fa, fb = fb, fa
    
    # Illinois Method refinement
    best_x = a if abs(fa) < abs(fb) else b
    best_f = min(abs(fa), abs(fb))
    
    for _ in range(100):
        if time.time() > deadline - 0.5:
            break
        
        if best_f <= 1e-8:
            return best_x
        
        denom = fb - fa
        if denom == 0:
            break
        
        x_new = (a * fb - b * fa) / denom
        f_new = f(x_new)
        
        if abs(f_new) < best_f:
            best_f = abs(f_new)
            best_x = x_new
        
        if abs(f_new) <= 1e-8:
            return x_new
        
        if f_new * fa < 0:
            b = x_new
            fb = f_new
            fa = fa / 2.0
        else:
            a = x_new
            fa = f_new
            fb = fb / 2.0
        
        if abs(b - a) < 1e-15:
            break
    
    return best_x
# EVOLVE-BLOCK-END

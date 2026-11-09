"""Spectral root-finder: FFT-based frequency/phase estimation then Newton refinement."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def find_root(f, x0):
    x0 = float(x0)
    T0 = time.time()
    DEADLINE = 143.0          # safety margin before 150 s hard kill
    TOL = 1e-12               # aim well below 1e-8
    HARD_TOL = 1e-8           # contract tolerance
    MAX_REFINE_ITERS = 100

    call_count = 0
    MAX_CALLS = 450           # Hard limit on calls to ensure safety

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

    # Evaluate start
    f0 = safe_f(x0)
    best_x = x0
    best_val = abs(f0) if f0 != float('nan') else float('inf')

    def track(x, fx):
        nonlocal best_x, best_val
        if fx != float('nan'):
            av = abs(fx)
            if av < best_val:
                best_val = av
                best_x = x

    track(x0, f0)

    if f0 != float('nan') and abs(f0) <= HARD_TOL:
        return x0

    eval_cache = {x0: f0}
    
    def get_f(x):
        if x in eval_cache:
            return eval_cache[x]
        val = safe_f(x)
        eval_cache[x] = val
        track(x, val)
        return val

    # ------------------------------------------------------------------
    # 1) Spectral approach: sample at multiple scales, fit sinusoid,
    #    predict nearest zero, then refine with Newton/bisection.
    # ------------------------------------------------------------------
    
    # Try several window sizes to capture the oscillation period.
    # For each window, sample N points, do FFT, estimate dominant freq & phase.
    # Predict zero crossing, verify with a few function evaluations.
    
    N_SAMPLES = 64  # Number of samples per window
    window_scales = [0.1, 0.5, 1.0, 5.0, 20.0, 100.0]
    
    best_predicted_x = None
    best_predicted_val = float('inf')
    
    for scale in window_scales:
        if not elapsed_ok() or call_count >= MAX_CALLS - 30:
            break
        
        # Sample points: x0 + (i - N/2) * (2*scale / N) for i in 0..N-1
        # This gives a window of width 2*scale centered at x0
        step = 2.0 * scale / N_SAMPLES
        xs = [x0 + (i - N_SAMPLES // 2) * step for i in range(N_SAMPLES)]
        
        # Evaluate all samples
        fs = []
        valid = True
        for x in xs:
            fx = get_f(x)
            if fx == float('nan'):
                valid = False
                break
            fs.append(fx)
        
        if not valid:
            continue
        
        # Check if any sample is already a root
        for x, fx in zip(xs, fs):
            if abs(fx) <= HARD_TOL:
                return x
        
        # Perform FFT to estimate dominant frequency
        # First, remove mean to get AC component
        fs_arr = np.array(fs, dtype=np.float64)
        mean_val = np.mean(fs_arr)
        ac = fs_arr - mean_val
        
        # If AC component is negligible, function is nearly constant - skip
        if np.max(np.abs(ac)) < 1e-12:
            continue
        
        # FFT
        fft_vals = np.fft.fft(ac)
        freqs = np.fft.fftfreq(N_SAMPLES, d=step)
        
        # Find dominant frequency (exclude DC)
        magnitudes = np.abs(fft_vals)
        magnitudes[0] = 0  # Exclude DC
        dominant_idx = np.argmax(magnitudes)
        dominant_freq = freqs[dominant_idx]
        
        if abs(dominant_freq) < 1e-12:
            continue
        
        # Extract amplitude and phase from the dominant component
        # f(x) ≈ A * cos(2π * freq * x + phi) + mean
        # We need to express in terms of our sampling grid.
        
        # The FFT gives us complex coefficients. For the dominant frequency bin,
        # we can reconstruct the sinusoid.
        
        # Actually, let's use a simpler approach: fit A*cos(w*x + phi) + C
        # using least squares on the samples.
        
        # Build design matrix for f(x) = A*cos(w*x) + B*sin(w*x) + C
        w = 2.0 * math.pi * dominant_freq
        
        # Design matrix: [cos(w*x_i), sin(w*x_i), 1]
        design = np.column_stack([
            np.cos(w * np.array(xs)),
            np.sin(w * np.array(xs)),
            np.ones(N_SAMPLES)
        ])
        
        # Least squares fit
        try:
            coeffs, residuals, rank, sv = np.linalg.lstsq(design, fs_arr, rcond=None)
        except:
            continue
        
        A = coeffs[0]
        B = coeffs[1]
        C = coeffs[2]
        
        # The fitted function is: g(x) = A*cos(w*x) + B*sin(w*x) + C
        # Zero crossings: A*cos(w*x) + B*sin(w*x) = -C
        # Let R = sqrt(A^2 + B^2), phi = atan2(B, A)
        # Then R*cos(w*x - phi) = -C
        # cos(w*x - phi) = -C/R
        # w*x - phi = ± arccos(-C/R) + 2πk
        
        R = math.sqrt(A*A + B*B)
        if R < 1e-12:
            continue
        
        # Check if solution exists: |-C/R| <= 1
        ratio = -C / R
        if abs(ratio) > 1.0:
            continue
        
        # Base solutions
        try:
            base_angle = math.acos(ratio)
        except:
            continue
        
        # Solutions: w*x = phi ± base_angle + 2πk
        # x = (phi ± base_angle + 2πk) / w
        
        phi = math.atan2(B, A)
        
        # Find the solution closest to x0
        # w*x ≈ phi ± base_angle + 2πk
        # x ≈ (phi ± base_angle + 2πk) / w
        
        # Estimate k: k ≈ (w*x0 - phi) / (2π)
        k_est = (w * x0 - phi) / (2.0 * math.pi)
        
        # Check k values around k_est
        best_x_candidate = None
        best_dist = float('inf')
        
        for k in [int(math.floor(k_est)) - 2, int(math.floor(k_est)) - 1, 
                  int(math.floor(k_est)), int(math.floor(k_est)) + 1,
                  int(math.floor(k_est)) + 2]:
            for sign in [1, -1]:
                x_candidate = (phi + sign * base_angle + 2.0 * math.pi * k) / w
                dist = abs(x_candidate - x0)
                if dist < best_dist:
                    best_dist = dist
                    best_x_candidate = x_candidate
        
        if best_x_candidate is None:
            continue
        
        # Verify the prediction: evaluate f at the predicted point
        f_pred = get_f(best_x_candidate)
        
        if f_pred != float('nan') and abs(f_pred) < best_predicted_val:
            best_predicted_val = abs(f_pred)
            best_predicted_x = best_x_candidate
            
            # If already good enough, return immediately
            if abs(f_pred) <= HARD_TOL:
                return best_x_candidate
    
    # ------------------------------------------------------------------
    # 2) If spectral prediction gave a good starting point, refine with Newton
    #    Otherwise, fall back to bracketing + Illinois
    # ------------------------------------------------------------------
    
    # If we have a spectral prediction, try Newton refinement
    if best_predicted_x is not None:
        x_newton = best_predicted_x
        f_newton = get_f(x_newton)
        
        # Newton's method with numerical derivative
        h = max(1e-8, abs(x_newton) * 1e-8)
        
        for _ in range(MAX_REFINE_ITERS):
            if not elapsed_ok() or call_count >= MAX_CALLS - 5:
                break
                
            if abs(f_newton) <= TOL:
                return x_newton
            
            # Numerical derivative
            f_plus = get_f(x_newton + h)
            f_minus = get_f(x_newton - h)
            
            if f_plus == float('nan') or f_minus == float('nan'):
                break
            
            df = (f_plus - f_minus) / (2.0 * h)
            
            if abs(df) < 1e-15:
                break
            
            # Newton step
            x_step = x_newton - f_newton / df
            
            # Safeguard: if step is too large, take a smaller step
            if abs(x_step - x_newton) > 10.0 * max(1.0, abs(x_newton)):
                x_step = x_newton + 0.5 * (f_newton / df) if df > 0 else x_newton - 0.5 * (f_newton / df)
            
            f_step = get_f(x_step)
            
            if f_step == float('nan'):
                break
            
            track(x_step, f_step)
            
            if abs(f_step) <= TOL:
                return x_step
            
            # Accept the step if it improves
            if abs(f_step) < abs(f_newton):
                x_newton = x_step
                f_newton = f_step
            else:
                # Halve the step
                x_step = x_newton - 0.5 * f_newton / df
                f_step = get_f(x_step)
                if f_step != float('nan') and abs(f_step) < abs(f_newton):
                    x_newton = x_step
                    f_newton = f_step
                else:
                    break
        
        # Return best found
        if best_val <= HARD_TOL:
            return best_x
    
    # ------------------------------------------------------------------
    # 3) Fallback: bracketing + Illinois (same as before)
    # ------------------------------------------------------------------
    a = b = None
    fa = fb = None
    
    # Try to find a bracket using the spectral info or scanning
    # Use the predicted point and its neighbors to find a sign change
    
    if best_predicted_x is not None:
        # Check around the predicted point for a sign change
        f_pred = get_f(best_predicted_x)
        if f_pred != float('nan'):
            # Try small offsets
            offsets = [0.01, -0.01, 0.1, -0.1, 1.0, -1.0, 10.0, -10.0]
            for offset in offsets:
                if not elapsed_ok() or call_count >= MAX_CALLS - 10:
                    break
                x_check = best_predicted_x + offset
                f_check = get_f(x_check)
                if f_check != float('nan') and f_pred * f_check < 0:
                    a, b = best_predicted_x, x_check
                    fa, fb = f_pred, f_check
                    break
    
    # If no bracket found yet, do a local scan
    if a is None:
        # Scan around x0 (or best_predicted_x if available)
        center = best_predicted_x if best_predicted_x is not None else x0
        scan_range = [0.01, 0.1, 1.0, 10.0]
        
        for sr in scan_range:
            if not elapsed_ok() or call_count >= MAX_CALLS - 10:
                break
            
            step = sr / 10.0
            prev_x = None
            prev_f = None
            
            for k in range(-10, 11):
                if not elapsed_ok() or call_count >= MAX_CALLS - 10:
                    break
                
                x_test = center + k * step
                f_test = get_f(x_test)
                
                if f_test != float('nan'):
                    if abs(f_test) <= HARD_TOL:
                        return x_test
                    
                    if prev_x is not None and prev_f is not None:
                        if prev_f * f_test < 0:
                            a, b = prev_x, x_test
                            fa, fb = prev_f, f_test
                            break
                    prev_x = x_test
                    prev_f = f_test
            
            if a is not None:
                break
    
    # If still no bracket, do geometric expansion from x0
    if a is None:
        h = max(1e-4, abs(x0) * 1e-4)
        last_pos_x = x0
        last_pos_f = f0
        last_neg_x = x0
        last_neg_f = f0
        
        for i in range(1, 30):
            if not elapsed_ok() or call_count >= MAX_CALLS - 10:
                break
            
            x_new_pos = x0 + i * h
            f_new_pos = get_f(x_new_pos)
            
            if f_new_pos != float('nan') and last_pos_f != float('nan'):
                if f_new_pos * last_pos_f < 0:
                    a, b = last_pos_x, x_new_pos
                    fa, fb = last_pos_f, f_new_pos
                    break
                elif abs(f_new_pos) <= HARD_TOL:
                    return x_new_pos
            
            if f_new_pos != float('nan'):
                last_pos_x = x_new_pos
                last_pos_f = f_new_pos
            
            x_new_neg = x0 - i * h
            f_new_neg = get_f(x_new_neg)
            
            if f_new_neg != float('nan') and last_neg_f != float('nan'):
                if f_new_neg * last_neg_f < 0:
                    a, b = x_new_neg, last_neg_x
                    fa, fb = f_new_neg, last_neg_f
                    break
                elif abs(f_new_neg) <= HARD_TOL:
                    return x_new_neg
            
            if f_new_neg != float('nan'):
                last_neg_x = x_new_neg
                last_neg_f = f_new_neg
            
            h *= 2.0
    
    # ------------------------------------------------------------------
    # 4) Refine with Illinois if we have a bracket
    # ------------------------------------------------------------------
    if a is not None:
        if a > b:
            a, b = b, a
            fa, fb = fb, fa
        
        x1, f1 = a, fa
        x2, f2 = b, fb
        
        if abs(f1) > abs(f2):
            x1, x2 = x2, x1
            f1, f2 = f2, f1
        
        for _ in range(MAX_REFINE_ITERS):
            if not elapsed_ok() or call_count >= MAX_CALLS - 5:
                break
                
            if abs(f1) <= TOL:
                return x1
            if abs(f2) <= TOL:
                return x2
            
            if f2 - f1 == 0:
                x_new = 0.5 * (x1 + x2)
            else:
                x_new = (x1 * f2 - x2 * f1) / (f2 - f1)
                
            if not math.isfinite(x_new) or x_new <= x1 or x_new >= x2:
                x_new = 0.5 * (x1 + x2)
                
            f_new = get_f(x_new)
            
            if f_new != float('nan'):
                if abs(f_new) <= TOL:
                    return x_new
                
                if f_new * f1 < 0:
                    if abs(f2) > abs(f1):
                        f2 *= 0.5
                    x2 = x_new
                    f2 = f_new
                else:
                    if abs(f1) > abs(f2):
                        f1 *= 0.5
                    x1 = x_new
                    f1 = f_new
                    
                track(x_new, f_new)
                
                if abs(x2 - x1) < 1e-14 * max(1.0, abs(x1)):
                    return x1 if abs(f1) < abs(f2) else x2

    # Return best found
    if best_x == best_x and math.isfinite(best_val):
        return best_x
    return x0
# EVOLVE-BLOCK-END

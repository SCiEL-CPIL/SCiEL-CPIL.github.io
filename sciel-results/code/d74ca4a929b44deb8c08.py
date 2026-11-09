"""Root finder for 3 fixed oscillatory equations."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin well under 150s

    x = float(x0)
    
    # Check initial point
    try:
        fx = f(x)
    except Exception:
        return x
    if not math.isfinite(fx):
        return x
    if abs(fx) <= 1e-8:
        return x

    x_best = x
    f_best = fx

    # Strategy: Use DFT to estimate oscillation frequency, predict zero-crossing,
    # then use Newton/secant to refine. This is more efficient than blind scanning.
    
    # Step 1: Sample the function at multiple points to estimate frequency
    # Use a logarithmic spacing around x0
    sample_points = []
    sample_vals = []
    
    # Sample at increasing distances from x0
    base_steps = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]
    for step in base_steps:
        for sign in [1, -1]:
            pt = x + sign * step
            if time.time() > deadline:
                break
            try:
                val = f(pt)
            except Exception:
                continue
            if not math.isfinite(val):
                continue
            sample_points.append(pt)
            sample_vals.append(val)
            if abs(val) < abs(f_best):
                x_best, f_best = pt, val
            if abs(val) <= 1e-8:
                return pt
    
    # Step 2: Try to find a bracket by checking sign changes in the samples
    a, b = None, None
    fa, fb = None, None
    
    # Sort samples by x to check for sign changes
    if len(sample_points) >= 2:
        sorted_idx = sorted(range(len(sample_points)), key=lambda i: sample_points[i])
        for i in range(len(sorted_idx) - 1):
            i1, i2 = sorted_idx[i], sorted_idx[i+1]
            x1, x2 = sample_points[i1], sample_points[i2]
            f1, f2 = sample_vals[i1], sample_vals[i2]
            if f1 * f2 < 0:
                a, b = min(x1, x2), max(x1, x2)
                fa, fb = (f1, f2) if x1 < x2 else (f2, f1)
                break
    
    # If no bracket found, try DFT-based frequency estimation
    if a is None and len(sample_points) >= 8:
        # Estimate dominant frequency using DFT
        # Use evenly spaced samples for better DFT accuracy
        # Pick samples that are roughly evenly spaced
        if len(sample_points) >= 8:
            # Take first 8 samples (closest to x0) for frequency estimation
            idx = sorted(range(len(sample_points)), key=lambda i: abs(sample_points[i] - x))[:8]
            xs = np.array([sample_points[i] for i in idx])
            ys = np.array([sample_vals[i] for i in idx])
            
            # Compute DFT
            if len(xs) >= 8:
                # Normalize to get roughly uniform spacing
                x_min, x_max = xs.min(), xs.max()
                if x_max > x_min:
                    # Resample to uniform grid
                    n_uniform = 16
                    x_uniform = np.linspace(x_min, x_max, n_uniform)
                    y_uniform = np.interp(x_uniform, xs, ys)
                    
                    # DFT
                    Y = np.fft.fft(y_uniform)
                    freqs = np.fft.fftfreq(n_uniform, d=(x_max - x_min) / (n_uniform - 1))
                    
                    # Find dominant frequency (exclude DC)
                    mag = np.abs(Y[1:])
                    if len(mag) > 0 and mag.max() > 0:
                        dom_idx = np.argmax(mag)
                        dom_freq = freqs[dom_idx + 1]
                        period = 1.0 / abs(dom_freq) if abs(dom_freq) > 1e-10 else None
                        
                        if period is not None and period > 0:
                            # Predict zero-crossing: for a sinusoid A*sin(2*pi*x/T + phi),
                            # zeros occur at x = (k*T - phi)/(2*pi) for integer k
                            # Estimate phase from the samples
                            # Use the fact that at a zero, the function crosses zero
                            # Try predicting zeros at x0 + k*period/2 for various k
                            # Check a few predictions
                            for k in range(-4, 5):
                                pred_x = x + k * period / 2
                                if time.time() > deadline:
                                    break
                                try:
                                    pred_f = f(pred_x)
                                except Exception:
                                    continue
                                if not math.isfinite(pred_f):
                                    continue
                                if abs(pred_f) < abs(f_best):
                                    x_best, f_best = pred_x, pred_f
                                if abs(pred_f) <= 1e-8:
                                    return pred_x
                                # Check if this is near a zero by looking at neighbors
                                if abs(pred_f) < 1.0:  # Close to zero
                                    # Try to bracket around this prediction
                                    for h in [period/10, period/20, period/50]:
                                        for sign in [1, -1]:
                                            pt = pred_x + sign * h
                                            try:
                                                pt_f = f(pt)
                                            except Exception:
                                                continue
                                            if not math.isfinite(pt_f):
                                                continue
                                            if pred_f * pt_f < 0:
                                                a, b = min(pred_x, pt), max(pred_x, pt)
                                                fa, fb = (pred_f, pt_f) if pred_x < pt else (pt_f, pred_f)
                                                break
                                        if a is not None:
                                            break
                                    if a is not None:
                                        break
    
    # If still no bracket, do a wider geometric scan
    if a is None:
        step = 1.0
        prev_x, prev_f = x, fx
        for i in range(30):
            if time.time() > deadline:
                break
            for sign in [1, -1]:
                c = x + sign * step
                try:
                    fc = f(c)
                except Exception:
                    continue
                if not math.isfinite(fc):
                    continue
                if abs(fc) < abs(f_best):
                    x_best, f_best = c, fc
                if prev_f * fc < 0:
                    a, b = min(prev_x, c), max(prev_x, c)
                    fa, fb = (prev_f, fc) if prev_x < c else (fc, prev_f)
                    break
                prev_x, prev_f = c, fc
            if a is not None:
                break
            step *= 2
    
    # If still no bracket, use Newton from best guess
    if a is None:
        x_curr = x_best
        f_curr = f_best
        for i in range(30):
            if time.time() > deadline:
                break
            h = 1e-7 * max(1.0, abs(x_curr))
            try:
                f_curr_h = f(x_curr + h)
            except Exception:
                break
            if not math.isfinite(f_curr_h):
                break
            d = (f_curr_h - f_curr) / h
            if d == 0 or not math.isfinite(d):
                break
            x_new = x_curr - f_curr / d
            if not math.isfinite(x_new):
                break
            x_curr = x_new
            try:
                f_curr = f(x_curr)
            except Exception:
                break
            if not math.isfinite(f_curr):
                break
            if abs(f_curr) < abs(f_best):
                x_best, f_best = x_curr, f_curr
            if abs(f_curr) <= 1e-8:
                return x_curr
        return x_best

    # Step 3: Refine using safeguarded secant/bisection
    x1, f1 = a, fa
    x2, f2 = b, fb
    
    if x1 == x2:
        return x1 if abs(f1) < abs(f2) else x2
        
    for i in range(100):
        if time.time() > deadline:
            break
            
        if abs(f1) <= 1e-8:
            return x1
        if abs(f2) <= 1e-8:
            return x2
            
        # Try Secant Step
        if f1 != f2:
            x_sec = x2 - f2 * (x2 - x1) / (f2 - f1)
        else:
            x_sec = 0.5 * (x1 + x2)
            
        # Safeguard
        if x_sec <= min(x1, x2) or x_sec >= max(x1, x2):
            x_new = 0.5 * (x1 + x2)
        else:
            x_new = x_sec
            
        try:
            f_new = f(x_new)
        except Exception:
            break
            
        if not math.isfinite(f_new):
            break
            
        if abs(f_new) <= 1e-8:
            return x_new
            
        if abs(f_new) < abs(f_best):
            x_best, f_best = x_new, f_new
            
        # Update bracket
        if f1 * f_new < 0:
            x2, f2 = x_new, f_new
        else:
            x1, f1 = x_new, f_new
            
        # Convergence check
        if abs(x2 - x1) < 1e-12 * max(1.0, abs(x1), abs(x2)):
            break

    # Return best
    if abs(f1) < abs(f2):
        return x1
    else:
        return x2

# EVOLVE-BLOCK-END

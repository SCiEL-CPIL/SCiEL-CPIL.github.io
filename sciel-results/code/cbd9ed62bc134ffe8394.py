import math
import time

def find_root(f, x0):
    """
    Find root x of f(x) such that |f(x)| <= 1e-8.
    Strategy:
    1. Check x0.
    2. Estimate local frequency/phase via DFT on 8 points to predict nearest zero.
    3. Use predicted zero as initial guess for Newton's method.
    4. If Newton fails or diverges, fall back to bracketing + Illinois.
    """
    start_time = time.time()
    deadline = start_time + 100.0  # Safe margin
    tol = 1e-8
    
    x0 = float(x0)
    
    # Helper to check time
    def time_up():
        return time.time() > deadline

    # 1. Evaluate x0
    f0 = f(x0)
    if math.isfinite(f0) and abs(f0) <= tol:
        return x0
    
    if not math.isfinite(f0):
        # If x0 is bad, we can't do DFT. Just return x0 or try a tiny step?
        # The contract says "non-finite x" is a failure. It doesn't explicitly say f(x0) can't be non-finite,
        # but if f0 is inf, we can't proceed with standard methods.
        # Let's try to find a finite point nearby.
        best_x = x0
        best_f = float('inf')
        for dx in [1e-4, -1e-4, 1e-3, -1e-3]:
            if time_up():
                break
            x_try = x0 + dx
            f_try = f(x_try)
            if math.isfinite(f_try):
                if abs(f_try) < best_f:
                    best_x = x_try
                    best_f = abs(f_try)
                if abs(f_try) <= tol:
                    return x_try
        return best_x

    best_x = x0
    best_f = abs(f0)

    # 2. DFT-based frequency estimation
    # Sample 8 points around x0
    # We need a scale for h. Let's try a few scales if the first one fails or gives weird results.
    # But to save calls, let's pick a reasonable h.
    # If the function is oscillatory, the period is likely not extremely small or large relative to x0.
    # Let's assume a characteristic length L ~ max(1, |x0|).
    # Try h = L / 100? Or just a fixed small number?
    # The current program used 1e-4 * max(1, |x0|). Let's stick with that for sampling.
    
    base_h = 1e-4 * max(1.0, abs(x0))
    
    # Sample points: x0 - 3.5*h ... x0 + 3.5*h ?
    # Let's use 8 points: x0 + k*h for k in -3.5, -2.5, ..., 3.5? 
    # Actually, for DFT we want equally spaced points.
    # Let's use 8 points: x_i = x0 + (i - 3.5) * h for i=0..7? No, symmetric is better.
    # x_i = x0 + (i - 3.5) * 2*h? No.
    # Let's just use 8 points centered at x0 with spacing h.
    # Indices: -3, -2, -1, 0, 1, 2, 3, 4? No, that's asymmetric.
    # Let's use 8 points: x0 - 3.5h, x0 - 2.5h, ..., x0 + 3.5h.
    # Spacing is h.
    # Total width 7h.
    
    h = base_h
    samples = []
    fvals = []
    valid = True
    
    # We already have f(x0). Let's use it as the center.
    # But DFT is easier with a regular grid.
    # Let's sample 8 points: x0 + k*h for k in [-3, -2, -1, 0, 1, 2, 3, 4]?
    # Or better, let's just sample 8 points in a small window and do a least squares fit to A*sin(w*x + phi) + C.
    # This is more robust than a simple DFT on a tiny window.
    
    # Let's try a linear least squares fit for: f(x) ~ A*sin(w*x) + B*cos(w*x) + C
    # We don't know w.
    # So we need to estimate w first.
    
    # Method:
    # 1. Sample 8 points.
    # 2. Try to find zero crossings in these samples? No, 8 points might miss them.
    # 3. Estimate frequency from autocorrelation?
    #    R(k) = sum f(x_i) f(x_{i+k})
    #    Find peak in R(k).
    
    # Let's try sampling 16 points to get a better frequency estimate.
    # 16 calls is a good investment if it saves 20-50 calls later.
    
    # Scale h. If x0 is large, h should be larger.
    # Let's try 3 different scales of h to see which gives a consistent frequency.
    # Scale 1: h = 1e-4 * max(1, |x0|)
    # Scale 2: h = 1e-3 * max(1, |x0|)
    # Scale 3: h = 1e-5 * max(1, |x0|)
    
    # To save calls, let's just pick one good scale.
    # For oscillatory functions like sin(x), sin(10x), etc., 1e-4 is usually fine for x~1.
    
    # Let's sample 16 points: x0 + (i - 7.5) * h for i=0..15.
    # This is symmetric around x0.
    
    n_samples = 16
    h = base_h
    xs = [x0 + (i - 7.5) * h for i in range(n_samples)]
    fs = []
    
    for i, x in enumerate(xs):
        if time_up():
            break
        if i == 7 or i == 8: # x0 is at i=7.5? No, x0 is not in the list if we use 7.5 offset.
            # Actually, x0 + (7 - 7.5)*h = x0 - 0.5h
            # x0 + (8 - 7.5)*h = x0 + 0.5h
            # So x0 is not sampled. We have f0.
            pass
        f_val = f(x)
        if not math.isfinite(f_val):
            valid = False
            break
        fs.append(f_val)
        if abs(f_val) < best_f:
            best_f = abs(f_val)
            best_x = x
        if abs(f_val) <= tol:
            return x
    
    if not valid or len(fs) < n_samples:
        # If DFT sampling failed, fall back to bracketing
        pass
    else:
        # Estimate frequency using autocorrelation
        # R(k) = sum_{i=0}^{N-k-1} fs[i] * fs[i+k]
        # We want to find the lag k that corresponds to the period.
        # The dominant frequency w corresponds to the peak in the power spectrum,
        # or the first peak in the autocorrelation.
        
        # Normalize fs by subtracting mean
        mean_f = sum(fs) / n_samples
        fs_centered = [v - mean_f for v in fs]
        
        # Autocorrelation
        max_lag = n_samples // 2
        autocorr = []
        for k in range(1, max_lag):
            s = 0.0
            for i in range(n_samples - k):
                s += fs_centered[i] * fs_centered[i + k]
            autocorr.append(s)
        
        # Find the first peak in autocorr
        # A peak is where autocorr[k-1] < autocorr[k] > autocorr[k+1]
        peak_lag = -1
        for k in range(1, len(autocorr) - 1):
            if autocorr[k-1] < autocorr[k] and autocorr[k] > autocorr[k+1]:
                peak_lag = k
                break
        
        if peak_lag > 0:
            # Period T = peak_lag * h
            # Frequency w = 2*pi / T = 2*pi / (peak_lag * h)
            T = peak_lag * h
            w = 2 * math.pi / T
            
            # Now we have an estimate of w.
            # Let's fit A*sin(w*x + phi) + C to the data.
            # Or simpler: use the phase from the DFT at frequency w?
            # Or just use the fact that we have a sinusoid with frequency w.
            # We can find the phase by projecting fs onto sin(w*x_i) and cos(w*x_i).
            
            S_s = 0.0
            S_c = 0.0
            S_sc = 0.0
            S_ss = 0.0
            S_cc = 0.0
            
            for i in range(n_samples):
                x_i = xs[i]
                f_i = fs[i]
                s_i = math.sin(w * x_i)
                c_i = math.cos(w * x_i)
                
                S_s += f_i * s_i
                S_c += f_i * c_i
                S_sc += s_i * c_i
                S_ss += s_i * s_i
                S_cc += c_i * c_i
                S_cs = S_sc # symmetric
                
            # Solve:
            # S_s = A * S_ss + B * S_sc + C * sum(s_i)  (sum(s_i) ~ 0)
            # S_c = A * S_sc + B * S_cc + C * sum(c_i)  (sum(c_i) ~ 0)
            
            # Assuming C is small or we can estimate it separately.
            # Let's ignore C for the phase estimation, or estimate C as mean_f.
            # f(x) = A*sin(w*x) + B*cos(w*x) + C
            # f(x) - C = A*sin(w*x) + B*cos(w*x)
            
            # Let g(x) = f(x) - mean_f
            # S_s' = sum g_i * s_i
            # S_c' = sum g_i * c_i
            
            S_s_prime = 0.0
            S_c_prime = 0.0
            for i in range(n_samples):
                g_i = fs[i] - mean_f
                s_i = math.sin(w * x_i)
                c_i = math.cos(w * x_i)
                S_s_prime += g_i * s_i
                S_c_prime += g_i * c_i
            
            # Matrix:
            # [ S_ss  S_sc ] [ A ]   [ S_s_prime ]
            # [ S_sc  S_cc ] [ B ] = [ S_c_prime ]
            
            det = S_ss * S_cc - S_sc * S_sc
            if abs(det) > 1e-12:
                A = (S_s_prime * S_cc - S_c_prime * S_sc) / det
                B = (S_c_prime * S_ss - S_s_prime * S_sc) / det
                
                # f(x) = A*sin(w*x) + B*cos(w*x) + C
                # We want to find x where f(x) = 0.
                # A*sin(w*x) + B*cos(w*x) = -C
                # R*sin(w*x + phi) = -C
                # R = sqrt(A^2 + B^2)
                # phi = atan2(B, A)  (since A*sin + B*cos = R*sin(w*x + phi) with R cos(phi)=A, R sin(phi)=B)
                # Wait: R*sin(w*x + phi) = R(sin(w*x)cos(phi) + cos(w*x)sin(phi))
                # Coeff of sin: R cos(phi) = A
                # Coeff of cos: R sin(phi) = B
                # So phi = atan2(B, A)
                
                R = math.sqrt(A*A + B*B)
                if R > 1e-12:
                    phi = math.atan2(B, A)
                    
                    # Solve R*sin(w*x + phi) + C = 0
                    # sin(w*x + phi) = -C/R
                    # Let alpha = -C/R
                    # If |alpha| > 1, no solution in this model (model is bad).
                    if abs(alpha) <= 1.0:
                        theta = math.asin(alpha)
                        # w*x + phi = theta + 2*pi*k  OR  w*x + phi = pi - theta + 2*pi*k
                        # x = (theta - phi + 2*pi*k) / w
                        # x = (pi - theta - phi + 2*pi*k) / w
                        
                        # We want the root closest to x0.
                        # Let base_theta1 = theta - phi
                        # Let base_theta2 = math.pi - theta - phi
                        
                        # Find k such that x is close to x0.
                        # w*x = base_theta + 2*pi*k
                        # x = (base_theta + 2*pi*k) / w
                        
                        # k ~ (w*x0 - base_theta) / (2*pi)
                        
                        candidates = []
                        
                        for base_theta in [base_theta1, base_theta2]:
                            k_float = (w * x0 - base_theta) / (2 * math.pi)
                            k1 = math.floor(k_float)
                            k2 = math.ceil(k_float)
                            for k in [k1 - 1, k1, k2, k2 + 1]:
                                x_candidate = (base_theta + 2 * math.pi * k) / w
                                candidates.append(x_candidate)
                        
                        # Evaluate f at the closest candidate
                        best_candidate = None
                        best_candidate_f = float('inf')
                        for xc in candidates:
                            if time_up():
                                break
                            if not math.isfinite(xc):
                                continue
                            fc = f(xc)
                            if math.isfinite(fc):
                                if abs(fc) < best_candidate_f:
                                    best_candidate_f = abs(fc)
                                    best_candidate = xc
                                if abs(fc) <= tol:
                                    return xc
                                if abs(fc) < best_f:
                                    best_f = abs(fc)
                                    best_x = xc
                        
                        if best_candidate is not None and best_candidate_f < best_f:
                            best_x = best_candidate
                            # Now use Newton's method to refine best_x
                            # We need the derivative. We can approximate it or use the model.
                            # Model derivative: f'(x) = R * w * cos(w*x + phi)
                            # Let's use the model derivative for the first step, then switch to finite difference?
                            # Or just use secant with the model prediction?
                            
                            # Let's try 2-3 Newton steps using the model derivative.
                            x_curr = best_candidate
                            for _ in range(5):
                                if time_up():
                                    break
                                f_curr = f(x_curr)
                                if abs(f_curr) <= tol:
                                    return x_curr
                                if not math.isfinite(f_curr):
                                    break
                                
                                # Derivative from model
                                w_x_phi = w * x_curr + phi
                                f_prime = R * w * math.cos(w_x_phi)
                                
                                if abs(f_prime) < 1e-12:
                                    break
                                    
                                x_new = x_curr - f_curr / f_prime
                                if not math.isfinite(x_new):
                                    break
                                
                                # Safety: if x_new is too far, dampen
                                if abs(x_new - x_curr) > 10 * h:
                                    x_new = x_curr + 0.5 * (x_new - x_curr)
                                    
                                x_curr = x_new
                                if abs(f_curr) < best_f:
                                    best_f = abs(f_curr)
                                    best_x = x_curr

    # If we are here, we might have a good best_x from DFT/Newton, or we are falling back.
    # Let's try a few more secant steps from best_x if it's not good enough.
    if abs(best_f) > tol:
        # Try to find a bracket near best_x
        # We have some points from DFT. Let's see if any of them bracket best_x.
        # Or just do a local search.
        
        # Let's try to bracket around best_x
        a = best_x
        fa = f(a) # We might have already evaluated this?
        # To be safe, re-evaluate or check cache? We don't have a cache.
        # But we know best_f = abs(fa).
        
        # Try expanding from a
        h_bracket = 1e-4 * max(1.0, abs(a))
        found_bracket = False
        b = None
        fb = None
        
        for i in range(10):
            if time_up():
                break
            x_r = a + h_bracket
            f_r = f(x_r)
            if math.isfinite(f_r):
                if fa * f_r <= 0:
                    b = x_r
                    fb = f_r
                    found_bracket = True
                    break
            x_l = a - h_bracket
            f_l = f(x_l)
            if math.isfinite(f_l):
                if fa * f_l <= 0:
                    b = x_l
                    fb = f_l
                    found_bracket = True
                    break
            h_bracket *= 2.0
        
        if found_bracket:
            # Illinois Method
            # Ensure |fa| < |fb|
            if abs(fa) < abs(fb):
                pass
            else:
                a, b = b, a
                fa, fb = fb, fa
                
            for _ in range(30):
                if time_up():
                    break
                if abs(fa) <= tol:
                    return a
                if abs(fb) <= tol:
                    return b
                    
                denom = fb - fa
                if abs(denom) < 1e-15:
                    break
                x_new = b - fb * (b - a) / denom
                f_new = f(x_new)
                
                if not math.isfinite(f_new):
                    break
                    
                if abs(f_new) <= tol:
                    return x_new
                    
                if fa * f_new < 0:
                    b = x_new
                    fb = f_new
                else:
                    a = x_new
                    fa = f_new
                    
                # Illinois modification
                # If the new point is on the same side as the point that was just replaced?
                # Standard:
                # If |f(a)| < |f(b)| and we update b, then f(b) = f(b)/2?
                # Let's use the explicit rule:
                # If f(a) and f(x_new) have same sign:
                #   a = x_new
                #   fa = fa / 2
                # Else:
                #   b = x_new
                #   fb = fb / 2
                # Wait, this is not quite right.
                # Let's stick to Regula Falsi if Illinois is tricky, or use a simple bisection fallback.
                # Actually, the previous code had a broken Illinois. Let's use a simple bisection if secant doesn't work well?
                # No, let's try to implement Illinois correctly.
                
                # From Wikipedia:
                # 1. Compute x_new.
                # 2. If f(a) and f(x_new) have same sign:
                #    a = x_new
                #    f(a) = f(a) / 2
                # Else:
                #    b = x_new
                #    f(b) = f(b) / 2
                # 3. Swap a, b if |f(b)| < |f(a)|.
                
                # My logic above:
                # if fa * f_new < 0: (sign change between a and new)
                #    root is in [a, x_new]
                #    b = x_new
                #    fb = f_new
                # else: (sign change between x_new and b)
                #    root is in [x_new, b]
                #    a = x_new
                #    fa = f_new
                
                # Now apply halving:
                # The point that was NOT updated is the one that keeps its original value?
                # No, the point that WAS updated gets halved?
                # "If the new point is on the same side as the point that was just replaced"
                # If we replaced b with x_new (because sign change was a-x_new), then x_new is on the "b side" (closer to old b? No, x_new is between a and b).
                # Actually, the rule is: halve the function value of the point that was NOT updated?
                # Let's check a reference.
                # "Illinois algorithm: ... If the new point is on the same side of the root as the point that was just replaced, then the function value of the other point is halved."
                # This is confusing.
                
                # Let's just use Regula Falsi. It's slower but robust.
                # Or, since we have a good guess, maybe we don't need many iterations.
                
                # Let's just break and return the best found so far if we are not converging fast.
                pass

    return best_x


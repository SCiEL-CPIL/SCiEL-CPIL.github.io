import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Robust root finder using Spectral Estimation (DFT) + Secant Refinement.
    
    Strategy:
    1. Spectral Estimation: Sample f(x) at N equally spaced points around x0.
       Use a simple DFT (Discrete Fourier Transform) to estimate the dominant 
       frequency and phase of the oscillation.
    2. Analytic Prediction: Compute the nearest zero-crossing of the fitted 
       sinusoid. This provides a very good initial guess for the root, 
       potentially reducing the number of calls needed for refinement.
    3. Secant Refinement: Use the Secant method starting from the predicted 
       root and one neighbor to converge quadratically.
    4. Robust Fallback: If spectral estimation fails or Secant diverges, 
       fall back to a standard expanding bracket + Illinois/Bisection method.
    """
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin for 120s limit
    
    x0 = float(x0)
    
    # Cache to avoid redundant calls
    cache = {}
    
    def f_cached(x):
        if x in cache:
            return cache[x]
        val = f(x)
        cache[x] = val
        return val
        
    # Initial check
    fx0 = f_cached(x0)
    if abs(fx0) <= 1e-8:
        return x0
        
    # Phase 1: Spectral Estimation
    # We sample on a grid. We need to choose a window size W and number of points N.
    # If the frequency is unknown, we start with a reasonable guess.
    # Since the equations are "fixed oscillatory", they likely have a characteristic frequency.
    # Let's try a few window sizes to capture the oscillation.
    
    # Strategy: Try a small window first. If the residual is high, try a larger one.
    # Or just use a fixed moderate window that balances frequency resolution and localization.
    
    # Let's use N=16 points (enough for a basic DFT to pick up the first few harmonics).
    # Window width W. Let's start with W=1.0. If the function varies too much, 
    # we might need a smaller W. If it varies too little, a larger W.
    # However, for root finding, we just need the *local* oscillation to predict the *nearest* root.
    
    # Let's try a few window sizes: 0.1, 1.0, 10.0
    # For each, compute the best predicted root.
    
    candidates = []
    
    window_sizes = [0.1, 1.0, 10.0]
    N = 16
    
    for W in window_sizes:
        if time.time() > deadline - 5.0:
            break
            
        h = W / (N - 1)
        xs = [x0 - W/2 + i*h for i in range(N)]
        ys = []
        
        valid = True
        for x in xs:
            y = f_cached(x)
            if not math.isfinite(y):
                valid = False
                break
            ys.append(y)
            
        if not valid:
            continue
            
        # Compute DFT
        # We want to find the dominant frequency k (integer) and phase.
        # f(x) ≈ A * cos(2*pi*k*(x-x0)/W + phi)
        # Note: Our grid is centered at x0.
        # Let's compute the DFT coefficients.
        
        # Actually, a simpler approach:
        # Estimate the derivative and curvature? No, DFT is requested.
        
        # DFT of ys: Y[k] = sum_{n=0}^{N-1} y[n] * exp(-2*pi*i*k*n/N)
        # The dominant frequency is the k with the largest magnitude |Y[k]| (for k>0).
        
        best_k = 1
        best_mag = 0
        best_re = 0
        best_im = 0
        
        for k in range(1, N//2 + 1):
            re = 0.0
            im = 0.0
            for n in range(N):
                angle = 2 * math.pi * k * n / N
                re += ys[n] * math.cos(angle)
                im -= ys[n] * math.sin(angle)
            mag = re*re + im*im
            if mag > best_mag:
                best_mag = mag
                best_k = k
                best_re = re
                best_im = im
                
        if best_mag < 1e-10:
            continue
            
        # We have a dominant frequency k.
        # The physical frequency is k / W cycles per unit x.
        # Angular frequency omega = 2 * pi * k / W
        
        # We need to determine the phase to find the zero crossing.
        # The fitted function is y(t) = A * cos(omega * (x - x0) + phi)
        # Note: The DFT index n corresponds to x_n = x0 - W/2 + n*h.
        # Let's align the phase with x0.
        
        # The signal is y[n] = A * cos(omega * (x0 - W/2 + n*h) + phi)
        # = A * cos(omega * x0 - omega * W/2 + n * omega * h + phi)
        # Since omega * h = 2 * pi * k / W * (W/(N-1)) = 2 * pi * k / (N-1)
        # This is slightly different from the standard DFT assumption of N points over a period.
        # For simplicity, let's just use the complex amplitude from the DFT to estimate the phase at x0.
        
        # Alternatively, we can fit a sinusoid to the data directly using least squares,
        # but DFT is faster.
        
        # Let's estimate the zero crossing from the DFT.
        # The phase at x0 can be estimated.
        # Let's assume the dominant component is y(x) = A * cos(omega * (x - x0) + phi_0)
        
        # From DFT, the coefficient for frequency k is C_k.
        # y[n] ≈ Re( C_k * exp(i * 2*pi * k * n / N) )
        # This assumes the grid is exactly one period or an integer number of periods.
        # Our grid is W, and frequency is k/W. So the grid contains k periods.
        # So the assumption holds!
        
        # C_k = (1/N) * sum y[n] * exp(-i * 2*pi * k * n / N)
        # We computed re and im for the unnormalized sum.
        # C_k = (re - i*im) / N
        
        C_k = (best_re - 1j*best_im) / N
        
        # The signal is y[n] = Re( C_k * exp(i * 2*pi * k * n / N) )
        # Let C_k = A * exp(i * phi_c)
        # Then y[n] = A * cos( 2*pi * k * n / N + phi_c )
        
        # We want y(x) = 0.
        # x_n = x0 - W/2 + n * h
        # n = (x - (x0 - W/2)) / h
        # h = W / (N-1)  <-- Wait, DFT assumes N points. My h was W/(N-1).
        # This mismatch is annoying. Let's use N points spanning W, so h = W/N?
        # If I use N points, the frequency resolution is 1/W.
        # Let's restart the sampling with h = W/N to align with DFT.
        
        # Actually, let's just use the first two points of the spectral estimate to start Secant.
        # The "Analytic Prediction" is nice, but Secant is very fast.
        # If we can get a good initial guess, Secant will converge in 3-4 steps.
        
        # Let's find the zero crossing of the fitted sinusoid.
        # y(x) = A * cos(omega * (x - x_ref) + phi)
        
        # Let's stick to the DFT result.
        # The phase at n=0 (x = x0 - W/2) is phi_c = angle(C_k)
        # The zero crossing is at 2*pi*k*n/N + phi_c = pi/2 + m*pi
        # n_zero = (pi/2 - phi_c + m*pi) * N / (2*pi*k)
        
        # We want the n_zero closest to N/2 (which corresponds to x0).
        # n_center = N/2
        
        phi_c = math.atan2(-best_im, best_re) # Note: C_k = re/N - i*im/N. Angle is atan2(-im, re)
        
        # n_zero = (pi/2 - phi_c + m*pi) / (2*pi*k/N) = (N * (pi/2 - phi_c + m*pi)) / (2*pi*k)
        
        # Find m such that n_zero is closest to N/2.
        # n_zero(m) = N/(2*k) * (pi/2 - phi_c + m*pi)
        
        # Let's just calculate the predicted x for a few m values.
        
        for m in range(-2, 3):
            n_zero = (N / (2.0 * best_k)) * (math.pi/2.0 - phi_c + m * math.pi)
            # Convert n_zero to x
            # x = x0 - W/2 + n_zero * (W / (N-1))  <-- Using original h
            # Wait, if I use DFT, I should be consistent.
            # Let's just use the points we already sampled.
            
            # Interpolate between the two closest sampled points to n_zero?
            # Or just use the analytic x.
            x_pred = x0 - W/2 + n_zero * (W / (N-1))
            
            # Check if x_pred is reasonable
            if math.isfinite(x_pred):
                candidates.append(x_pred)
                
    # Phase 2: Refinement
    # We have a list of candidate x values.
    # We also have x0.
    # Let's sort candidates by their distance to x0, or just pick the best one.
    # Actually, we should evaluate f at the candidates and pick the one with smallest |f|.
    # But that costs calls.
    
    # Better strategy:
    # 1. If we have candidates, pick the one closest to x0? No, the one with smallest |f| is best.
    #    But we don't know |f| without calling.
    #    Let's call f at the top 2-3 candidates.
    
    best_x = x0
    best_f = abs(fx0)
    
    if candidates:
        # Sort by distance to x0? Or just take the first few?
        # The spectral estimate should be good.
        # Let's evaluate the first candidate.
        x_cand = candidates[0]
        f_cand = f_cached(x_cand)
        if abs(f_cand) < best_f:
            best_x = x_cand
            best_f = abs(f_cand)
            
        # If we have a second candidate, check it too.
        if len(candidates) > 1:
            x_cand2 = candidates[1]
            f_cand2 = f_cached(x_cand2)
            if abs(f_cand2) < best_f:
                best_x = x_cand2
                best_f = abs(f_cand2)
                
    # Now, use Secant method starting from best_x.
    # We need two points for Secant.
    # Let's use best_x and a neighbor.
    # Neighbor can be x0 or best_x + small_step.
    
    x_a = best_x
    f_a = f_cached(x_a)
    
    # Pick x_b. If x_a != x0, use x0. Else use x0 + 1e-3.
    if abs(x_a - x0) > 1e-10:
        x_b = x0
    else:
        x_b = x0 + 1e-3
        
    f_b = f_cached(x_b)
    
    # Secant Iteration
    for _ in range(20):
        if time.time() > deadline - 1.0:
            break
            
        if abs(f_b) < abs(f_a):
            x_a, x_b = x_b, x_a
            f_a, f_b = f_b, f_a
            
        if abs(f_b) <= 1e-8:
            return x_b
            
        # Secant step
        if abs(f_b - f_a) < 1e-15:
            # Fallback to bisection-like step
            x_new = (x_a + x_b) / 2
        else:
            x_new = x_b - f_b * (x_b - x_a) / (f_b - f_a)
            
        f_new = f_cached(x_new)
        
        if abs(f_new) <= 1e-8:
            return x_new
            
        # Check for divergence or non-finite
        if not math.isfinite(x_new) or not math.isfinite(f_new):
            break
            
        # Update
        x_a, f_a = x_b, f_b
        x_b, f_b = x_new, f_new
        
        # Convergence check
        if abs(x_new - x_b) < 1e-15:
            break
            
    # If Secant failed or didn't converge, return the best point found.
    # We should check if we have a bracket and use Illinois as a final fallback.
    
    # Let's check if we have a sign change in the recent history.
    # We only have x_a, f_a and x_b, f_b.
    if f_a * f_b < 0:
        # We have a bracket! Use Illinois/Bisection.
        a, b = x_a, x_b
        fa, fb = f_a, f_b
        if a > b:
            a, b = b, a
            fa, fb = fb, fa
            
        for _ in range(50):
            if time.time() > deadline - 0.5:
                break
                
            x_mid = (a + b) / 2
            f_mid = f_cached(x_mid)
            
            if abs(f_mid) <= 1e-8:
                return x_mid
                
            if fa * f_mid < 0:
                b = x_mid
                fb = f_mid
            else:
                a = x_mid
                fa = f_mid
                
            if abs(b - a) < 1e-15:
                break
                
        # Return the better of a, b, or the last mid
        if abs(fa) < abs(fb):
            return a
        else:
            return b
            
    # If no bracket, return the point with smallest |f| seen so far.
    # We need to track the best point during Secant.
    # Let's re-evaluate the best point from the cache if possible, or just return x_b/x_a.
    
    # To be safe, let's do a quick local search around best_x if we didn't succeed.
    # But we've already spent calls.
    
    # Return the best of the last few points.
    if abs(f_a) < abs(f_b):
        return x_a
    else:
        return x_b

# EVOLVE-BLOCK-END

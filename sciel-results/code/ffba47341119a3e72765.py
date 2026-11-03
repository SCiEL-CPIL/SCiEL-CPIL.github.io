import numpy as np
import time
from scipy.optimize import minimize

N = 600

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 110 seconds to leave margin for 120s limit
    
    # Initial guess: uniform (box)
    x0 = np.ones(N)
    
    # We want to minimize the normalized maximum of the autoconvolution.
    # Let f be the vector of heights.
    # Autoconvolution g = f * f.
    # Normalized max: max(g) / (sum(f)^2) ? Or max(g) / (sum(f^2)) ?
    # The task says "minimize the normalized maximum of its autoconvolution".
    # Common normalization for C1 bound context (uncertainty principle or similar) often involves L1 or L2 norms.
    # Let's look at inspiration 3: "J(f) = \frac{\|f * f\|_\i..."
    # Usually for autoconvolution peaks, we care about the peak relative to the total energy or mass.
    # If we scale f by c, (cf * cf) = c^2 (f*f).
    # If metric is max(f*f) / (sum(f))^2: scale invariant.
    # If metric is max(f*f) / (sum(f^2)): scale invariant.
    
    # Let's assume the metric is M(f) = max(f*f) / (sum(f)^2).
    # For a box of height 1, sum(f)=N, f*f has peak N (at lag 0). M = N/N^2 = 1/N.
    # If we can make the peak smaller relative to the square of the mass, we improve.
    # This is equivalent to maximizing the "flatness" of the autoconvolution peak.
    # A spread-out function has a lower peak in autoconvolution relative to its mass?
    # Let's check:
    # f = delta (one bin is 1, rest 0). sum=1. f*f = delta. peak=1. M = 1/1 = 1.
    # f = box (N bins are 1). sum=N. f*f peak=N. M = N/N^2 = 1/N.
    # So spreading out reduces the normalized peak if normalized by mass squared.
    
    # What if normalized by energy squared (sum(f^2))^2?
    # f = delta. sum(f^2)=1. peak=1. M = 1/1 = 1.
    # f = box. sum(f^2)=N. peak=N. M = N/N^2 = 1/N.
    # Same behavior.
    
    # What if normalized by sum(f^2)?
    # f = delta. M = 1/1 = 1.
    # f = box. M = N/N = 1.
    # In this case, scaling doesn't matter, but shape does.
    # Delta gives 1. Box gives 1.
    # Gaussian?
    # Let's stick to the most likely interpretation for "upper bound on C1" in signal processing contexts.
    # Often C1 relates to the concentration of energy.
    # Let's try to minimize max(f*f) subject to sum(f^2) = 1.
    # This is a standard problem. The minimizer is often a Gaussian or similar smooth window.
    # However, we are on a discrete grid with non-negative constraint.
    
    # Let's define the objective function for optimization.
    # We will use the metric: max(f*f) / (sum(f)^2).
    # This is scale invariant, so we can fix sum(f) = 1 or sum(f^2)=1.
    # Let's fix sum(f) = 1.
    # Then we minimize max(f*f).
    
    def objective(x):
        # x is the vector of heights
        # Ensure non-negativity? The optimizer might go negative.
        # We can penalize negative values or project.
        # For now, let's use a soft penalty or just let it run and clip later?
        # Better to use a parameterization that ensures non-negativity, e.g., x = exp(y).
        # But sum(exp(y)) = 1 is a constraint.
        # Alternatively, minimize max(f*f) with constraint sum(f)=1, f>=0.
        
        # For speed, we compute autoconvolution via FFT.
        # f * f is the inverse FFT of |FFT(f)|^2.
        # Note: circular convolution vs linear.
        # The problem specifies intervals on [-1/4, 1/4].
        # Autoconvolution of a function on a bounded interval usually implies linear convolution,
        # but if the function is zero outside, linear convolution on a larger grid is equivalent to circular convolution on a sufficiently large grid.
        # To be safe and accurate, let's use linear convolution.
        # N=600. Linear conv length is 2N-1 = 1199.
        # FFT size should be at least 1199. Next power of 2 is 2048.
        
        f = np.maximum(x, 0) # Clip negative values to 0 to satisfy non-negativity
        # Normalize to sum(f) = 1
        s = np.sum(f)
        if s < 1e-12:
            return 1e12
        f = f / s
        
        # Compute linear autoconvolution
        # Using FFT for efficiency
        fft_size = 1 << (2*N - 1).bit_length() # Next power of 2
        F = np.fft.rfft(f, n=fft_size)
        G = np.fft.irfft(F * F, n=fft_size)
        
        # The autoconvolution peak is the maximum of G.
        # Since f is non-negative, the peak is likely at lag 0 (center of the conv result).
        # But let's just take the max to be safe.
        peak = np.max(G)
        
        # The objective is peak.
        return peak

    # We want to minimize peak(f*f) subject to sum(f)=1, f>=0.
    # Using scipy.optimize.minimize.
    # We need a parameterization that handles the sum constraint and non-negativity.
    # Let x be unconstrained. Let f = softmax(x) or exp(x) / sum(exp(x)).
    # Softmax ensures f > 0 and sum(f) = 1.
    
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)

    def obj_wrapper(y):
        f = softmax(y)
        # Compute peak
        fft_size = 1 << (2*N - 1).bit_length()
        F = np.fft.rfft(f, n=fft_size)
        G = np.fft.irfft(F * F, n=fft_size)
        return np.max(G)

    # Initial guess: uniform distribution (y = 0)
    y0 = np.zeros(N)
    
    # Optimize
    # Use L-BFGS-B as it handles large vectors well.
    # However, the objective is non-smooth due to max.
    # We can use a smooth approximation: log-sum-exp with large beta.
    # max(G) approx log(sum(exp(beta * G)) / beta)
    
    beta = 100.0
    
    def smooth_obj(y):
        f = softmax(y)
        fft_size = 1 << (2*N - 1).bit_length()
        F = np.fft.rfft(f, n=fft_size)
        G = np.fft.irfft(F * F, n=fft_size)
        # Smooth max
        # G can be small.
        return np.log(np.sum(np.exp(beta * G))) / beta

    # Start optimization
    best_y = y0
    best_score = smooth_obj(y0)
    
    try:
        # Run optimization
        # Limit iterations to ensure time safety
        max_iter = 50
        res = minimize(smooth_obj, y0, method='L-BFGS-B', options={'maxiter': max_iter, 'ftol': 1e-9})
        
        if res.success or res.fun < best_score:
            best_y = res.x
            best_score = res.fun
            
        # Time check
        if time.time() > deadline:
            f_best = softmax(best_y)
            return list(f_best)

        # Refine with a local search on the non-smooth objective?
        # Or just use the result from the smooth approximation.
        # The smooth approximation with beta=100 should be close to the true max.
        
    except Exception as e:
        # If optimization fails, fall back to initial guess
        pass

    # Final result
    f_best = softmax(best_y)
    
    # One more check: does a simple triangle or cosine window do better?
    # Let's evaluate a few candidates and pick the best.
    candidates = []
    
    # 1. Optimized result
    candidates.append(f_best)
    
    # 2. Uniform
    candidates.append(np.ones(N) / N)
    
    # 3. Triangle (linear ramp up and down)
    x = np.linspace(0, 1, N)
    tri = np.minimum(x, 1-x) * 2
    tri = tri / np.sum(tri)
    candidates.append(tri)
    
    # 4. Cosine (Hann window)
    hann = np.sin(np.pi * x)**2
    hann = hann / np.sum(hann)
    candidates.append(hann)
    
    # 5. Gaussian
    sigma = 0.25 * (N-1) # Adjusted for index
    t = np.arange(N) - (N-1)/2
    gauss = np.exp(-t**2 / (2*sigma**2))
    gauss = gauss / np.sum(gauss)
    candidates.append(gauss)
    
    # Evaluate all candidates using the true (non-smooth) objective
    def true_obj(f):
        fft_size = 1 << (2*N - 1).bit_length()
        F = np.fft.rfft(f, n=fft_size)
        G = np.fft.irfft(F * F, n=fft_size)
        return np.max(G)
    
    best_f = candidates[0]
    best_val = true_obj(best_f)
    
    for f in candidates[1:]:
        val = true_obj(f)
        if val < best_val:
            best_val = val
            best_f = f
            
    # If we have time, try to optimize starting from the best candidate
    if time.time() < deadline - 20:
        # Re-initialize y from best_f
        # y = log(f) is not quite right for softmax inverse, but close.
        # Actually, we can just run a few more steps of L-BFGS-B from the best candidate's log values?
        # No, let's just stick with the best of the set. The optimization from uniform might have found a local minimum.
        # Let's try optimizing from the best candidate's implicit y.
        # f = softmax(y) => y = log(f) is only valid up to a constant.
        # So y_init = np.log(best_f + 1e-15)
        
        y_init = np.log(best_f + 1e-15)
        
        try:
            res2 = minimize(smooth_obj, y_init, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-9})
            if res2.success or res2.fun < best_val:
                f_refined = softmax(res2.x)
                val_refined = true_obj(f_refined)
                if val_refined < best_val:
                    best_f = f_refined
                    best_val = val_refined
        except:
            pass

    return list(best_f)

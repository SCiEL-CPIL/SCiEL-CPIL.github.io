import numpy as np
import time
import scipy.optimize as opt

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

def construct():
    deadline = time.time() + 120
    
    def score(h):
        s = np.sum(h)
        if s == 0 or not np.isfinite(s):
            return float('inf')
        z = np.convolve(h, h)
        return abs(2 * N * np.max(z) / (s * s))

    x = np.linspace(0, 1, N)
    
    best_h = np.ones(N)
    best_score = score(best_h)
    
    # Parametric families to explore
    # 1. Gaussian: exp(-((x-c)^2)/(2*sigma^2))
    # 2. Super-Gaussian: exp(-((x-c)^2)^p)
    # 3. Raised Cosine: cos^2(pi*(x-c)) clipped to [0,1] effectively, or similar bump
    # We will optimize center (c), width (w), and shape (p)
    
    # Initial guesses for (center, width, shape)
    # Center should be 0.5 for symmetry to minimize edge effects in linear convolution 
    # (though the metric is global max, symmetry often helps concentrate mass)
    # Width should be moderate.
    
    initial_guesses = [
        [0.5, 0.2, 1.0],  # Gaussian-ish
        [0.5, 0.1, 2.0],  # Narrower
        [0.5, 0.3, 2.0],  # Wider
        [0.5, 0.15, 3.0], # Super-Gaussian
        [0.5, 0.25, 0.5], # Very broad/flat
        [0.3, 0.2, 1.0],  # Off-center? Maybe not optimal for max autocorr but worth checking
        [0.7, 0.2, 1.0],
    ]
    
    for guess in initial_guesses:
        if time.time() > deadline - 5:
            break
            
        c0, w0, p0 = guess
        
        def objective(params):
            c, w, p = params
            # Ensure parameters stay in valid ranges
            # c in [0.1, 0.9]
            # w in [0.01, 0.5]
            # p in [0.1, 5.0]
            
            # Soft boundaries or clipping
            c = np.clip(c, 0.1, 0.9)
            w = np.clip(w, 0.01, 0.5)
            p = np.clip(p, 0.1, 5.0)
            
            # Construct h
            # Using a generalized Gaussian / Super-Gaussian form
            # h(x) = exp(-((x-c)/w)^(2*p))
            # This allows p=1 (Gaussian), p=2 (Super-Gaussian exp(-x^4)), p=0.5 (exp(-sqrt(x^2)) which is exp(-|x|), Laplacian-like)
            
            diff = (x - c) / w
            # Handle numerical issues with negative bases and fractional powers
            # diff^2 is always positive, so (diff^2)^p is safe.
            # exp(-(diff^2)^p)
            h = np.exp(-(diff**2) ** p)
            
            # Ensure sum is not zero (it won't be for these forms)
            # Normalize? The metric is scale invariant: 
            # abs(2*N*max(conv(h,h)) / sum(h)^2)
            # If h -> a*h, conv -> a^2*conv, sum -> a*sum. Ratio -> a^2/a^2 = 1.
            # So scaling doesn't matter. We can just use the raw values.
            
            return score(h)
        
        try:
            res = opt.minimize(objective, [c0, w0, p0], method='Nelder-Mead', 
                               options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
            
            if res.success or res.fun < best_score:
                c_opt, w_opt, p_opt = res.x
                c_opt = np.clip(c_opt, 0.1, 0.9)
                w_opt = np.clip(w_opt, 0.01, 0.5)
                p_opt = np.clip(p_opt, 0.1, 5.0)
                
                diff = (x - c_opt) / w_opt
                h_opt = np.exp(-(diff**2) ** p_opt)
                
                # Refine with a local random search or L-BFGS-B if needed?
                # Nelder-Mead is robust for this small dim.
                
                s_opt = score(h_opt)
                if s_opt < best_score:
                    best_score = s_opt
                    best_h = h_opt
                    
        except Exception:
            pass

    # Try a second family: Polynomial bumps
    # h(x) = (1 - ((x-c)/w)^2)^k for |x-c|<w, else 0
    # This has compact support which might reduce the "tail" contributions to max autocorrelation?
    # Actually, compact support makes the autocorrelation triangular-ish.
    
    for guess in [[0.5, 0.5, 1.0], [0.5, 0.3, 2.0], [0.5, 0.4, 1.5]]:
        if time.time() > deadline - 5:
            break
        c0, w0, k0 = guess
        
        def objective_poly(params):
            c, w, k = params
            c = np.clip(c, 0.1, 0.9)
            w = np.clip(w, 0.1, 0.5)
            k = np.clip(k, 0.1, 10.0)
            
            diff = np.abs(x - c) / w
            # h = (1 - diff^2)^k where diff < 1, else 0
            mask = diff < 1
            h = np.zeros(N)
            h[mask] = (1 - diff[mask]**2) ** k
            
            return score(h)
        
        try:
            res = opt.minimize(objective_poly, [c0, w0, k0], method='Nelder-Mead',
                               options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
            if res.fun < best_score:
                c_opt, w_opt, k_opt = res.x
                c_opt = np.clip(c_opt, 0.1, 0.9)
                w_opt = np.clip(w_opt, 0.1, 0.5)
                k_opt = np.clip(k_opt, 0.1, 10.0)
                
                diff = np.abs(x - c_opt) / w_opt
                mask = diff < 1
                h_opt = np.zeros(N)
                h_opt[mask] = (1 - diff[mask]**2) ** k_opt
                
                s_opt = score(h_opt)
                if s_opt < best_score:
                    best_score = s_opt
                    best_h = h_opt
        except Exception:
            pass

    return best_h
# EVOLVE-BLOCK-END

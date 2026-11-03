import numpy as np
import time
from scipy.signal import fftconvolve
from scipy.optimize import minimize

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s margin for 150s limit

    def _autoconv_max_abs(s):
        ac = fftconvolve(s, s, mode='full')
        return np.max(np.abs(ac))

    def _squared_integral(s):
        return (np.sum(s)) ** 2

    def _raw_value(s):
        si = _squared_integral(s)
        if si < 1e-15:
            return np.inf
        return _autoconv_max_abs(s) / si

    def _soft_max_abs(x, temp):
        ax = np.abs(x)
        m = np.max(ax)
        if m == 0:
            return 0.0
        scaled = ax / temp
        max_scaled = np.max(scaled)
        if max_scaled > 50:
            return m
        log_sum = np.log(np.sum(np.exp(scaled - max_scaled))) + max_scaled
        return temp * log_sum

    def _soft_objective(x, temp):
        # x is continuous vector, we want to minimize soft-max of autoconv sidelobes / squared integral
        s = x
        ac = fftconvolve(s, s, mode='full')
        center = N - 1
        ac_sidelobes = np.concatenate([ac[:center], ac[center+1:]])
        sm = _soft_max_abs(ac_sidelobes, temp)
        si = _squared_integral(s)
        if si < 1e-15:
            return 1e10
        return sm / si

    def _soft_gradient(x, temp):
        # Numerical gradient of soft objective
        eps = 1e-5
        grad = np.zeros_like(x)
        f0 = _soft_objective(x, temp)
        for i in range(N):
            xp = x.copy()
            xp[i] += eps
            fp = _soft_objective(xp, temp)
            grad[i] = (fp - f0) / eps
        return grad

    def _soft_objective_fast(x, temp):
        # Faster version for optimization
        s = x
        ac = fftconvolve(s, s, mode='full')
        center = N - 1
        ac_sidelobes = np.concatenate([ac[:center], ac[center+1:]])
        sm = _soft_max_abs(ac_sidelobes, temp)
        si = (np.sum(s)) ** 2
        if si < 1e-15:
            return 1e10
        return sm / si

    def _continuous_optimize(s0, temp_schedule, max_bfgs_iters=200):
        # Run BFGS on continuous space with adaptive temperature
        best_x = s0.copy().astype(float)
        best_val = _raw_value(best_x)
        
        for temp in temp_schedule:
            if time.time() > deadline - 10:
                break
            
            def obj(x):
                return _soft_objective_fast(x, temp)
            
            res = minimize(obj, best_x, method='BFGS', options={'maxiter': max_bfgs_iters, 'gtol': 1e-4})
            if res.success or res.fun < _raw_value(best_x):
                new_x = res.x
                # Normalize to have similar magnitude
                new_s = new_x
                new_val = _raw_value(new_s)
                if new_val < best_val:
                    best_val = new_val
                    best_x = new_x
        
        return best_x, best_val

    def _discretize_and_polish(s_cont):
        # Convert continuous to discrete signs, then local search
        s = np.sign(s_cont)
        if np.all(s == 0):
            s = np.ones(N)
            s[0] = -1
        # Ensure not all same sign
        if np.all(s > 0) or np.all(s < 0):
            s[0] = -s[0]
        
        # Local search polish
        best_s = s.copy()
        best_val = _raw_value(best_s)
        
        for it in range(2000):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(N):
                s_flip = best_s.copy()
                s_flip[i] = -s_flip[i]
                val = _raw_value(s_flip)
                if val < best_val:
                    best_val = val
                    best_s = s_flip
                    improved = True
                    break
            if not improved:
                break
        
        return best_s, best_val

    def _peak_suppression(s, max_iters=500):
        # Directly target the argmax of autoconvolution
        best_s = s.copy()
        best_val = _raw_value(best_s)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            ac = fftconvolve(best_s, best_s, mode='full')
            center = N - 1
            ac_sidelobes = np.concatenate([ac[:center], ac[center+1:]])
            
            # Find the peak
            peak_idx = np.argmax(np.abs(ac_sidelobes))
            peak_val = np.abs(ac_sidelobes[peak_idx])
            
            if peak_val < 1e-10:
                break
            
            # Try flipping elements that contribute most to this peak
            # The autoconv at lag k is sum_j s[j]*s[j+k]
            # For peak at lag k, try flipping s[j] for j near the peak
            k = peak_idx if peak_idx < center else peak_idx - (2*N - 1)
            k_abs = abs(k)
            
            improved = False
            # Try flipping elements that are part of the dominant products
            for j in range(max(0, -k_abs), min(N, N - k_abs)):
                if j + k_abs >= N:
                    break
                s_flip = best_s.copy()
                s_flip[j] = -s_flip[j]
                val = _raw_value(s_flip)
                if val < best_val:
                    best_val = val
                    best_s = s_flip
                    improved = True
                    break
            
            if not improved:
                # Try random flips near the peak
                for _ in range(20):
                    if time.time() > deadline - 5:
                        break
                    idx = np.random.randint(0, N)
                    s_flip = best_s.copy()
                    s_flip[idx] = -s_flip[idx]
                    val = _raw_value(s_flip)
                    if val < best_val:
                        best_val = val
                        best_s = s_flip
                        improved = True
                        break
        
        return best_s, best_val

    # Generate initial candidates
    np.random.seed(42)
    
    candidates = []
    
    # 1. Random signs
    for seed in [42, 123, 456, 789]:
        np.random.seed(seed)
        s = np.random.choice([-1.0, 1.0], size=N)
        candidates.append(s)
    
    # 2. Random Gaussian signs
    for seed in [11, 22, 33]:
        np.random.seed(seed)
        s = np.random.randn(N)
        s = np.sign(s)
        if np.all(s > 0) or np.all(s < 0):
            s[0] = -s[0]
        candidates.append(s)
    
    # 3. Smooth random
    for seed in [7, 8, 9]:
        np.random.seed(seed)
        s = np.random.randn(N)
        kernel = np.ones(5) / 5
        s = np.convolve(s, kernel, mode='same')
        s = np.sign(s)
        if np.all(s > 0) or np.all(s < 0):
            s[0] = -s[0]
        candidates.append(s)

    # Temperature schedule for soft-max annealing
    temp_schedule = [5.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625]

    # Optimize each candidate
    best_s = None
    best_val = np.inf
    
    for s in candidates:
        if time.time() > deadline - 15:
            break
        
        # Phase 1: Continuous optimization with adaptive temperature
        s_cont, val_cont = _continuous_optimize(s, temp_schedule, max_bfgs_iters=100)
        
        # Phase 2: Discretize and polish
        s_disc, val_disc = _discretize_and_polish(s_cont)
        if val_disc < best_val:
            best_val = val_disc
            best_s = s_disc.copy()
        
        # Phase 3: Peak suppression
        if time.time() > deadline - 10:
            break
        s_peak, val_peak = _peak_suppression(s_disc, max_iters=300)
        if val_peak < best_val:
            best_val = val_peak
            best_s = s_peak.copy()
    
    # Final polish: try to improve with more aggressive search
    if best_s is not None and time.time() < deadline - 10:
        # Try random restarts from the best solution
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            s_perturbed = best_s.copy()
            # Flip a few random elements
            for _ in range(3):
                idx = np.random.randint(0, N)
                s_perturbed[idx] = -s_perturbed[idx]
            s_new, val_new = _discretize_and_polish(s_perturbed)
            if val_new < best_val:
                best_val = val_new
                best_s = s_new.copy()
    
    # Ensure the result is a valid sequence
    if best_s is None:
        # Fallback: return a simple sequence
        best_s = np.ones(N)
        best_s[0] = -1.0
    
    return best_s

# EVOLVE-BLOCK-END

"""Rewritten program for third autocorrelation inequality optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _objective_soft_max(x, beta=100.0):
    """
    Objective to minimize: log(2*N) + logsumexp(beta * conv) - 2*log(abs(sum(h)))
    This is monotonic with the target value: 2*N*max(conv)/(sum(h))^2
    """
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    if m < 0:
        # If all conv are negative, max is negative. 
        # But (sum h)^2 is positive. 
        # The formula is abs(2*N*max(conv)/s^2).
        # If max(conv) is negative, the value is small.
        # However, conv[N-1] = s^2 >= 0. So max(conv) >= 0.
        # So m is always >= 0.
        pass
        
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -500, 500)
    log_sum_exp = m + np.log(np.sum(np.exp(shifted))) / beta
    
    # We want to minimize 2*N*log_sum_exp / s^2
    # Taking log: log(2*N) + log(log_sum_exp) - 2*log(|s|)
    # But log_sum_exp is an approximation of max.
    # Let's just minimize the ratio directly using the soft max value.
    # Value approx = 2*N*log_sum_exp / s^2
    val = 2.0 * N * log_sum_exp / (s * s)
    return val

def _objective_hard(x):
    """Hard objective for final evaluation."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    return abs(2.0 * N * np.max(conv) / (s * s))

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_h = np.ones(N)
    best_val = _objective_hard(best_h)
    
    rng = np.random.default_rng(42)
    
    # List of initializations
    initializations = []
    
    # 1. Random normal with small mean
    for _ in range(10):
        h0 = rng.standard_normal(N)
        # Ensure sum is not too small
        if abs(np.sum(h0)) < 1e-3:
            h0[0] += 1.0
        initializations.append(h0)
        
    # 2. Barker-like / Low autocorrelation sequences
    # Try random sign sequences (sum might be small, add bias)
    for _ in range(10):
        h0 = rng.choice([-1, 1], size=N).astype(float)
        # Add a small DC component to ensure sum != 0 and to shape the peak
        h0 += 0.1 * rng.standard_normal()
        initializations.append(h0)
        
    # 3. Sine waves with random phase
    for freq in [1, 2, 5, 10]:
        t = np.linspace(0, 2*np.pi*freq, N, endpoint=False)
        h0 = np.sin(t + rng.random()*2*np.pi)
        h0 += 0.5 # Shift to have positive mean
        initializations.append(h0)

    # 4. Polynomial perturbations of 1s
    for _ in range(5):
        h0 = np.ones(N)
        h0 += 0.1 * rng.standard_normal(N)
        initializations.append(h0)

    # Optimization parameters
    betas = [50.0, 100.0, 200.0]
    
    for init_idx, h0 in enumerate(initializations):
        if time.time() > deadline - 20.0:
            break
        
        h_curr = h0.copy()
        
        # Try different betas, starting with lower for exploration, higher for exploitation
        for beta in betas:
            if time.time() > deadline - 20.0:
                break
            
            # Use L-BFGS-B
            try:
                res = minimize(_objective_soft_max, h_curr, 
                              method='L-BFGS-B',
                              args=(beta,),
                              options={'maxiter': 100, 'ftol': 1e-9, 'gtol': 1e-7, 'maxls': 20})
                h_new = res.x
            except Exception:
                h_new = h_curr
            
            # Evaluate hard objective
            val_new = _objective_hard(h_new)
            
            if val_new < best_val:
                best_val = val_new
                best_h = h_new.copy()
                h_curr = h_new.copy()
            else:
                # If it didn't improve, maybe try a smaller step or different beta?
                # For now, just keep the best found so far as the new current for next beta
                # Actually, if val_new > best_val, h_new might be worse than h_curr?
                # We should compare h_new with h_curr as well.
                val_curr = _objective_hard(h_curr)
                if val_new < val_curr:
                    h_curr = h_new.copy()
                # else h_curr stays

    # Final polish with high beta on the best solution
    if time.time() < deadline - 10.0:
        for beta in [200.0, 500.0]:
            try:
                res = minimize(_objective_soft_max, best_h,
                              method='L-BFGS-B',
                              args=(beta,),
                              options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 30})
                val = _objective_hard(res.x)
                if val < best_val:
                    best_val = val
                    best_h = res.x.copy()
            except Exception:
                pass

    # Ensure sum is not zero
    if abs(np.sum(best_h)) < 1e-10:
        best_h = np.ones(N)
        best_val = _objective_hard(best_h)
        
    return best_h
# EVOLVE-BLOCK-END

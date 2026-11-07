import numpy as np
from scipy.optimize import minimize
import time

N = 100
DEADLINE = time.time() + 130  # safety margin before 150s kill

def _correlations_full(h):
    """Compute all 199 correlation values of h with (1-h) in full mode."""
    one_minus_h = 1.0 - h
    # np.correlate(h, one_minus_h, mode='full') gives 199 values
    # Index 0 corresponds to lag -99, index 99 to lag 0, index 198 to lag +99
    return np.correlate(h, one_minus_h, mode='full')

def _objective_and_grad(params):
    """params: 99 free heights. h[99] = 50 - sum(params)."""
    h = np.empty(N)
    h[:N-1] = params
    h[N-1] = 50.0 - np.sum(params)
    
    # Check bounds
    if np.any(h < -1e-10) or np.any(h > 1 + 1e-10):
        return 1e10, np.zeros(N-1)
    
    corrs = _correlations_full(h)
    # Use log-sum-exp for smooth max
    beta = 50.0
    max_c = np.max(corrs)
    # Soft max: (1/beta) * log(sum(exp(beta * c)))
    log_sum_exp = max_c + np.log(np.sum(np.exp(beta * (corrs - max_c))))
    obj = log_sum_exp / N * 2.0
    
    # Gradient via finite differences would be slow; use analytic-ish approach
    # Actually, let's just use numerical gradient via the optimizer
    return obj, None

def construct():
    t0 = time.time()
    
    # Start from seed: all 0.5
    h_seed = np.full(N, 0.5)
    
    best_h = h_seed.copy()
    best_val = np.max(np.correlate(h_seed, 1-h_seed, mode='full')) / N * 2.0
    
    def eval_val(h):
        return np.max(np.correlate(h, 1-h, mode='full')) / N * 2.0
    
    # Try multiple random initializations and gradient descent
    np.random.seed(42)
    
    for trial in range(20):
        if time.time() - t0 > DEADLINE:
            break
        
        # Random perturbation around 0.5, maintaining sum=50
        if trial == 0:
            h = h_seed.copy()
        else:
            h = np.random.uniform(0, 1, N)
            # Adjust to sum to 50
            h = h - np.mean(h) + 0.5
            h = np.clip(h, 0, 1)
            # Re-adjust sum
            diff = 50.0 - np.sum(h)
            # Distribute diff proportionally to those not at bounds
            mask = (h > 1e-6) & (h < 1 - 1e-6)
            if np.sum(mask) > 0:
                h[mask] += diff / np.sum(mask)
                h = np.clip(h, 0, 1)
                # Final correction
                diff2 = 50.0 - np.sum(h)
                if abs(diff2) < 1e-6:
                    h += diff2 / N
                else:
                    # Try again with smaller adjustment
                    h = np.clip(h + diff2/N, 0, 1)
        
        val = eval_val(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
        
        # Gradient descent with projection
        beta = 30.0
        lr = 0.01
        for step in range(500):
            if time.time() - t0 > DEADLINE:
                break
            
            corrs = _correlations_full(h)
            max_c = np.max(corrs)
            # Soft max gradient
            exps = np.exp(beta * (corrs - max_c))
            weights = exps / np.sum(exps)
            
            # Gradient of soft_max w.r.t. h
            # d/dh_i [sum_k w_k * corr_k] where corr_k = sum_j h[j](1-h[j+k])
            # This is complex; use numerical gradient for the soft max
            grad = np.zeros(N)
            eps = 1e-6
            for i in range(N):
                h_plus = h.copy()
                h_plus[i] += eps
                h_minus = h.copy()
                h_minus[i] -= eps
                # Renormalize to maintain sum
                h_plus = h_plus - (h_plus - h).mean()
                h_minus = h_minus - (h_minus - h).mean()
                h_plus = np.clip(h_plus, 0, 1)
                h_minus = np.clip(h_minus, 0, 1)
                
                cp = _correlations_full(h_plus)
                cm = _correlations_full(h_minus)
                mp = np.max(cp)
                mm = np.max(cm)
                sp = mp + np.log(np.sum(np.exp(beta*(cp-mp))))
                sm = mm + np.log(np.sum(np.exp(beta*(cm-mm))))
                grad[i] = (sp - sm) / (2*eps)
            
            # Update
            h_new = h - lr * grad
            # Project: maintain sum=50 and bounds
            h_new = h_new - (np.sum(h_new) - 50.0) / N
            h_new = np.clip(h_new, 0, 1)
            # Re-project sum
            diff = 50.0 - np.sum(h_new)
            if abs(diff) > 1e-12:
                mask = (h_new > 1e-8) & (h_new < 1 - 1e-8)
                if np.sum(mask) > 0:
                    h_new[mask] += diff / np.sum(mask)
                    h_new = np.clip(h_new, 0, 1)
            
            val_new = eval_val(h_new)
            if val_new < best_val:
                best_val = val_new
                best_h = h_new.copy()
            
            h = h_new
            lr *= 0.995
        
        if time.time() - t0 > DEADLINE:
            break
    
    # Final polish: ensure exact sum
    best_h = np.clip(best_h, 0, 1)
    diff = 50.0 - np.sum(best_h)
    best_h += diff / N
    best_h = np.clip(best_h, 0, 1)
    
    # Verify
    final_val = eval_val(best_h)
    if final_val < best_val:
        best_val = final_val
    
    return best_h

# EVOLVE-BLOCK-START
# (The construct function is defined above)
# EVOLVE-BLOCK-END

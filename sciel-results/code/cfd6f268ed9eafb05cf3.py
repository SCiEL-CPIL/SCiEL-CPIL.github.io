import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin
    
    # Precompute autoconvolution structure
    # For lag k (0 to N-1), autoconv[k] = sum_i h[i]*h[i+k] for i where i+k < N
    # For negative lags, autoconv[-k] = autoconv[k] by symmetry for real functions
    # Actually for cross-correlation it's different, but for autoconvolution of real signal:
    # (f*f)[t] = integral f(s)f(t-s) ds
    # For discrete step function, the autoconvolution at lag k (in units of interval width)
    # is sum_i h[i] * h[i+k] for the positive lags, and by symmetry for negative.
    # But wait, autoconvolution is f*f, not f*conj(f). For real f, (f*f)[k] = sum_i f[i]f[i+k]
    # This is NOT symmetric in general unless f is even.
    
    # Let me reconsider. The autoconvolution (f*f) at shift k:
    # C[k] = sum_{i} h[i] * h[i+k] where the product is nonzero
    # For k >= 0: C[k] = sum_{i=0}^{N-1-k} h[i]*h[i+k]
    # For k < 0: C[k] = sum_{i=0}^{N-1+k} h[i]*h[i+k] = sum_{j=-k}^{N-1} h[j-k]*h[j] ... 
    # Actually C[-k] = sum_i h[i]*h[i-k] = sum_j h[j+k]*h[j] = C[k] only if we define it that way.
    # No: C[k] = sum_i h[i]h[i+k]. C[-k] = sum_i h[i]h[i-k]. Let j=i-k: C[-k]=sum_j h[j+k]h[j]=C[k].
    # So it IS symmetric: C[k]=C[-k].
    
    # So we only need to compute C[0] through C[N-1], and C[k]=C[-k].
    
    # Objective: minimize max_k |C[k]| subject to sum |h[i]| = 1
    # Using smooth max: (1/T) * log(sum_k exp(T*|C[k]|))
    
    # Since we need gradients, let's use the log-sum-exp trick with a parameter T.
    
    def autoconv_values(h):
        """Compute C[0..N-1] where C[k] = sum_i h[i]*h[i+k]"""
        C = np.zeros(N)
        for k in range(N):
            if k == 0:
                C[k] = np.dot(h, h)
            else:
                C[k] = np.dot(h[:N-k], h[k:])
        return C
    
    def smooth_max_abs(C, T=50.0):
        """Smooth approximation of max |C[k]| using log-sum-exp"""
        absC = np.abs(C)
        # log-sum-exp for stability
        m = np.max(absC)
        if m < 1e-12:
            return 0.0
        logsum = m + np.log(np.sum(np.exp(T * (absC - m))))
        return logsum / T
    
    def objective(h):
        C = autoconv_values(h)
        # Normalize by L1 norm squared
        l1 = np.sum(np.abs(h))
        if l1 < 1e-12:
            return 1e10
        return smooth_max_abs(C, T=50.0) / (l1 * l1)
    
    def objective_grad(h):
        """Compute objective and gradient using finite differences or analytical"""
        # Use numerical gradient for simplicity, but this is slow.
        # Let's try analytical approach.
        
        C = autoconv_values(h)
        l1 = np.sum(np.abs(h))
        if l1 < 1e-12:
            return 1e10, np.zeros(N)
        
        # For the smooth max, d/dC_k = T*exp(T*|C[k]|) / sum_j exp(T*|C[j]|) * sign(C[k])
        # But we need d objective / d h[i]
        
        # This is complex. Let's use a simpler approach: 
        # minimize max_k C[k]^2 / l1^2, which is equivalent.
        # Or use a different smooth approximation.
        
        # Alternative: use the fact that we can fix l1=1 by normalizing.
        # Let's work with normalized h: g = h / l1
        # Then objective = max_k |C_g[k]| where C_g = C / l1^2
        # d C_g[k] / d h[i] = (dC[k]/dh[i] * l1^2 - C[k] * 2*l1 * dh_l1/dh[i]) / l1^4
        # = (dC[k]/dh[i]) / l1^2 - C[k] * 2 * sign(h[i]) / l1^3
        
        # This is getting complex. Let's just use a simpler optimization.
        
        # Use scipy's minimize with BFGS or L-BFGS-B
        pass
    
    # Let's use scipy.optimize
    from scipy.optimize import minimize
    
    def obj_and_grad(h):
        C = autoconv_values(h)
        l1 = np.sum(np.abs(h))
        if l1 < 1e-12:
            return 1e10, np.zeros(N)
        
        # Smooth max of |C[k]| / l1^2
        T = 30.0
        absC = np.abs(C)
        m = np.max(absC)
        if m < 1e-12:
            return 0.0, np.zeros(N)
        
        # Weights for soft-max
        exp_vals = np.exp(T * (absC - m))
        weights = exp_vals / np.sum(exp_vals)
        
        # d smooth_max / d C[k] = T * weights[k] * sign(C[k])
        d_smooth_dC = T * weights * np.sign(C)
        
        # d C[k] / d h[i]:
        # C[k] = sum_j h[j]*h[j+k] for k>=0
        # dC[k]/dh[i] = h[i+k] (if i+k < N) + h[i-k] (if i-k >= 0, from the term where j=i-k, h[j+k]=h[i])
        # Wait, C[k] = sum_j h[j]*h[j+k]. 
        # d/dh[i]: term where j=i: h[i+k] (if i+k<N). Term where j+k=i, i.e., j=i-k: h[i-k] (if i-k>=0).
        # So dC[k]/dh[i] = h[i+k] * (i+k<N) + h[i-k] * (i-k>=0)
        
        # d objective / d h[i] = (1/l1^2) * sum_k d_smooth_dC[k] * dC[k]/dh[i] 
        #                        - (2/l1^3) * smooth_max * sign(h[i])
        
        # This is O(N^2) per gradient evaluation. With N=400, that's 160K ops, manageable.
        
        dC_dh = np.zeros((N, N))
        for k in range(N):
            for i in range(N):
                val = 0.0
                if i + k < N:
                    val += h[i + k]
                if i - k >= 0:
                    val += h[i - k]
                dC_dh[k, i] = val
        
        # d objective / d h = (1/l1^2) * dC_dh.T @ d_smooth_dC - (2/l1^3) * smooth_max_val * sign(h)
        smooth_max_val = m + np.log(np.sum(exp_vals)) / T
        
        grad = (dC_dh.T @ d_smooth_dC) / (l1 * l1) - (2.0 / (l1 * l1 * l1)) * smooth_max_val * np.sign(h)
        
        return smooth_max_val / (l1 * l1), grad
    
    # Initial guess: random signed function
    np.random.seed(42)
    h0 = np.random.choice([-1.0, 1.0], size=N)
    
    # Normalize
    h0 = h0 / np.sum(np.abs(h0))
    
    best_val, best_h = objective(h0), h0.copy()
    
    # Try multiple initializations and optimization
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        
        if restart == 0:
            h_init = h0.copy()
        else:
            # Perturb best or random
            if np.random.rand() < 0.5:
                h_init = best_h + 0.1 * np.random.randn(N)
            else:
                h_init = np.random.choice([-1.0, 1.0], size=N)
            h_init = h_init / np.sum(np.abs(h_init))
        
        try:
            result = minimize(
                lambda x: obj_and_grad(x)[0],
                h_init,
                jac=lambda x: obj_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 500, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if result.success or result.fun < best_val:
                h_opt = result.x
                val_opt = objective(h_opt)
                if val_opt < best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final: round to sign function for cleaner output? No, keep continuous.
    # The task says "signed step heights", so continuous values should be fine.
    
    return best_h

# EVOLVE-BLOCK-START
# (The construct function is defined above, but we need to keep the structure)
# Actually, let me restructure to match the original format.
# EVOLVE-BLOCK-END

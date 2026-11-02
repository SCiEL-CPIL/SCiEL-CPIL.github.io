import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin
    
    def compute_ratio(h):
        h = np.asarray(h, dtype=float)
        l2_sq = np.sum(h**2)
        l1 = np.sum(h)
        l_inf = np.max(h)
        if l1 < 1e-15 or l_inf < 1e-15:
            return 0.0
        h_conv = np.convolve(h, h, mode='full')
        max_conv = np.max(h_conv)
        if max_conv < 1e-15:
            return 0.0
        return (l2_sq) / (l1 * l_inf * max_conv)
    
    def neg_log_ratio(x):
        """Negative log of the ratio for minimization."""
        h = np.exp(x)  # Ensure positivity via log-parameterization
        h = h / np.max(h)  # Normalize to max=1
        l2_sq = np.sum(h**2)
        l1 = np.sum(h)
        if l1 < 1e-15:
            return 1e10
        h_conv = np.convolve(h, h, mode='full')
        max_conv = np.max(h_conv)
        if max_conv < 1e-15:
            return 1e10
        ratio = l2_sq / (l1 * max_conv)  # l_inf=1 due to normalization
        if ratio <= 0:
            return 1e10
        return -np.log(ratio)
    
    # Generate multiple initial candidates
    candidates = []
    
    # Candidate 1: Sidon set
    S1 = [0, 1, 4, 10, 18, 23, 30]
    h1 = np.zeros(N)
    for idx in S1:
        h1[idx] = 1.0
    candidates.append(h1 / np.max(h1))
    
    # Candidate 2: Concentrated peak
    h2 = np.zeros(N)
    h2[N//2] = 1.0
    candidates.append(h2)
    
    # Candidate 3: Triangular pulse
    h3 = np.zeros(N)
    center = N // 2
    for i in range(N):
        h3[i] = max(0, 1 - abs(i - center) / 10.0)
    if np.max(h3) > 0:
        candidates.append(h3 / np.max(h3))
    
    # Candidate 4: Sparse random
    np.random.seed(42)
    h4 = np.zeros(N)
    indices = np.random.choice(N, 8, replace=False)
    h4[indices] = 1.0
    candidates.append(h4 / np.max(h4))
    
    # Candidate 5: Two peaks
    h5 = np.zeros(N)
    h5[10] = 1.0
    h5[40] = 1.0
    candidates.append(h5)
    
    best_h = None
    best_ratio = 0.0
    
    for h0 in candidates:
        if time.time() > deadline:
            break
        
        # Initialize log-parameterization
        h0_safe = np.clip(h0, 1e-10, None)
        x0 = np.log(h0_safe)
        
        # Optimize using L-BFGS-B
        try:
            result = minimize(
                neg_log_ratio,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            # Reconstruct h from optimized x
            h_opt = np.exp(result.x)
            h_opt = h_opt / np.max(h_opt)
            h_opt = np.clip(h_opt, 0, None)
            
            ratio = compute_ratio(h_opt)
            if ratio > best_ratio:
                best_ratio = ratio
                best_h = h_opt.copy()
        except Exception:
            continue
        
        # Also try local refinement with random restarts
        if time.time() > deadline:
            break
        
        for restart in range(50):
            if time.time() > deadline:
                break
            
            # Perturb the best solution
            if best_h is not None:
                h_pert = best_h.copy()
                idx = np.random.randint(N)
                h_pert[idx] = max(0, h_pert[idx] + np.random.uniform(-0.3, 0.3))
                if np.max(h_pert) > 0:
                    h_pert = h_pert / np.max(h_pert)
            else:
                h_pert = h0.copy()
            
            h_pert_safe = np.clip(h_pert, 1e-10, None)
            x_pert = np.log(h_pert_safe)
            
            try:
                result = minimize(
                    neg_log_ratio,
                    x_pert,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                h_opt = np.exp(result.x)
                h_opt = h_opt / np.max(h_opt)
                h_opt = np.clip(h_opt, 0, None)
                
                ratio = compute_ratio(h_opt)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_h = h_opt.copy()
            except Exception:
                continue
    
    # Final high-precision polish
    if best_h is not None and time.time() < deadline - 5:
        h_pert = best_h.copy()
        h_pert_safe = np.clip(h_pert, 1e-10, None)
        x_pert = np.log(h_pert_safe)
        
        try:
            result = minimize(
                neg_log_ratio,
                x_pert,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            h_opt = np.exp(result.x)
            h_opt = h_opt / np.max(h_opt)
            h_opt = np.clip(h_opt, 0, None)
            
            ratio = compute_ratio(h_opt)
            if ratio > best_ratio:
                best_ratio = ratio
                best_h = h_opt.copy()
        except Exception:
            pass
    
    if best_h is None:
        # Fallback to original Sidon set
        S = [0, 1, 4, 10, 18, 23, 30]
        best_h = np.zeros(N)
        for idx in S:
            best_h[idx] = 1.0
        best_h = best_h / np.max(best_h)
    
    return list(best_h)

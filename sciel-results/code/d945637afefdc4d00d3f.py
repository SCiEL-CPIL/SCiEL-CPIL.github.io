import numpy as np
from scipy.optimize import minimize
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Initial guess
    h = np.full(N, 0.5)
    
    def objective(h_flat, tau, lam):
        h = h_flat.reshape(N)
        h_shift1 = np.roll(h, 1)
        h_shift2 = np.roll(h, 2)
        
        O1 = np.sum(h * (1 - h_shift1))
        O2 = np.sum(h * (1 - h_shift2))
        
        max_O = max(O1, O2)
        exp1 = np.exp(tau * (O1 - max_O))
        exp2 = np.exp(tau * (O2 - max_O))
        soft_max = max_O + (1.0 / tau) * np.log(exp1 + exp2)
        
        sum_h = np.sum(h)
        penalty = lam * (sum_h - N / 2.0) ** 2
        
        return soft_max + penalty
    
    def gradient(h_flat, tau, lam):
        h = h_flat.reshape(N)
        
        h_shift1 = np.roll(h, 1)
        h_shift_m1 = np.roll(h, -1)
        
        h_shift2 = np.roll(h, 2)
        h_shift_m2 = np.roll(h, -2)
        
        grad_O1 = (1 - h_shift1) - h_shift_m1
        grad_O2 = (1 - h_shift2) - h_shift_m2
        
        O1 = np.sum(h * (1 - h_shift1))
        O2 = np.sum(h * (1 - h_shift2))
        
        max_O = max(O1, O2)
        exp1 = np.exp(tau * (O1 - max_O))
        exp2 = np.exp(tau * (O2 - max_O))
        denom = exp1 + exp2
        
        w1 = exp1 / denom
        w2 = exp2 / denom
        
        grad_soft = w1 * grad_O1 + w2 * grad_O2
        
        sum_h = np.sum(h)
        grad_pen = 2 * lam * (sum_h - N / 2.0) * np.ones(N)
        
        return (grad_soft + grad_pen).flatten()

    def project_to_sum(h, target_sum):
        lo = min(h) - 1.0
        hi = max(h) + 1.0
        
        for _ in range(50):
            mid = (lo + hi) / 2.0
            h_proj = np.clip(h - mid, 0.0, 1.0)
            s = np.sum(h_proj)
            if s > target_sum:
                lo = mid
            else:
                hi = mid
                
        h_final = np.clip(h - (lo + hi) / 2.0, 0.0, 1.0)
        return h_final

    best_h = h.copy()
    best_score = float('inf')
    
    def get_score(h):
        h = np.clip(h, 0, 1)
        O1 = np.sum(h * (1 - np.roll(h, 1)))
        O2 = np.sum(h * (1 - np.roll(h, 2)))
        return max(O1, O2)

    current_h = h.copy()
    
    taus = [1.0, 5.0, 10.0, 50.0, 100.0, 500.0]
    lams = [10.0, 100.0, 1000.0, 10000.0]
    
    for tau in taus:
        for lam in lams:
            if time.time() > deadline - 10:
                break
            
            res = minimize(
                objective, 
                current_h, 
                method='L-BFGS-B', 
                jac=lambda x: gradient(x, tau, lam),
                bounds=[(0, 1)] * N,
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            if res.success:
                current_h = res.x
                current_h = project_to_sum(current_h, N/2.0)
                score = get_score(current_h)
                
                if score < best_score:
                    best_score = score
                    best_h = current_h.copy()
                    
        if time.time() > deadline - 10:
            break
            
    if time.time() < deadline - 5:
        res = minimize(
            objective, 
            best_h, 
            method='L-BFGS-B', 
            jac=lambda x: gradient(x, 1000.0, 10000.0),
            bounds=[(0, 1)] * N,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-10}
        )
        if res.success:
            h_final = project_to_sum(res.x, N/2.0)
            score = get_score(h_final)
            if score < best_score:
                best_score = score
                best_h = h_final.copy()
                
    return best_h

"""Second autocorrelation inequality: maximize ratio via smooth optimization."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # The grid: 50 points on [-1/4, 1/4]
    x = np.linspace(-0.25, 0.25, N)
    dx = x[1] - x[0]
    
    def autoconv(h):
        return np.convolve(h, h)
    
    def compute_ratio(h):
        h = np.asarray(h, dtype=float)
        h = np.maximum(h, 0.0)
        norm2_sq = np.sum(h**2)
        if norm2_sq < 1e-15:
            return 0.0
        norm1 = np.sum(h)
        if norm1 < 1e-15:
            return 0.0
        ac = np.convolve(h, h)
        sup_ac = np.max(np.abs(ac))
        if sup_ac < 1e-15:
            return 0.0
        return norm2_sq / (norm1 * sup_ac)
    
    def soft_max(v, beta):
        """Smooth approximation of max(v) using log-sum-exp."""
        vmax = np.max(v)
        if vmax < 1e-30:
            return 1e-30
        shifted = v / beta
        shifted = shifted - np.max(shifted)
        s = np.sum(np.exp(shifted))
        return beta * (np.log(s) + np.max(shifted))
    
    def smooth_objective(h, beta):
        """Minimize ||h||_1 * soft_max(h*h) subject to ||h||_2 = 1, h >= 0"""
        h = np.maximum(h, 0.0)
        norm2_sq = np.sum(h**2)
        if norm2_sq < 1e-15:
            return 1e10
        h_norm = h / np.sqrt(norm2_sq)
        norm1 = np.sum(h_norm)
        ac = np.convolve(h_norm, h_norm)
        smax = soft_max(np.abs(ac), beta)
        if norm1 < 1e-15 or smax < 1e-15:
            return 1e10
        return norm1 * smax
    
    def smooth_gradient(h, beta):
        """Numerical gradient of smooth objective."""
        eps = 1e-7
        grad = np.zeros(N)
        f0 = smooth_objective(h, beta)
        for i in range(N):
            h_plus = h.copy()
            h_plus[i] += eps
            fp = smooth_objective(h_plus, beta)
            
            h_minus = h.copy()
            h_minus[i] = max(0, h[i] - eps)
            fm = smooth_objective(h_minus, beta)
            
            if h[i] <= eps:
                grad[i] = (fp - f0) / eps
            else:
                grad[i] = (fp - fm) / (2 * eps)
        return grad
    
    def nonsmooth_objective(h):
        """Minimize ||h||_1 * ||h*h||_inf subject to ||h||_2 = 1, h >= 0"""
        h = np.maximum(h, 0.0)
        norm2_sq = np.sum(h**2)
        if norm2_sq < 1e-15:
            return 1e10
        h_norm = h / np.sqrt(norm2_sq)
        norm1 = np.sum(h_norm)
        ac = np.convolve(h_norm, h_norm)
        sup_ac = np.max(np.abs(ac))
        if sup_ac < 1e-15:
            return 1e10
        return norm1 * sup_ac
    
    def nonsmooth_gradient(h):
        """Numerical gradient of nonsmooth objective."""
        eps = 1e-7
        grad = np.zeros(N)
        f0 = nonsmooth_objective(h)
        for i in range(N):
            h_plus = h.copy()
            h_plus[i] += eps
            fp = nonsmooth_objective(h_plus)
            
            h_minus = h.copy()
            h_minus[i] = max(0, h[i] - eps)
            fm = nonsmooth_objective(h_minus)
            
            if h[i] <= eps:
                grad[i] = (fp - f0) / eps
            else:
                grad[i] = (fp - fm) / (2 * eps)
        return grad
    
    # Multi-start optimization
    best_score = 0.0
    best_h = None
    
    rng = np.random.RandomState(42)
    n_starts = 25
    
    for s in range(n_starts):
        if time.time() > deadline:
            break
        
        # Different initialization strategies
        if s == 0:
            h0 = np.ones(N)
        elif s == 1:
            h0 = np.exp(-((np.arange(N) - N/2)**2) / (2 * (N/4)**2))
        elif s == 2:
            h0 = np.zeros(N)
            h0[0] = 1.0
            h0[-1] = 1.0
        elif s == 3:
            h0 = np.zeros(N)
            idx = rng.choice(N, size=N//5, replace=False)
            h0[idx] = 1.0
        elif s == 4:
            h0 = 1.0 - np.abs(np.arange(N) - N/2) / (N/2)
            h0 = np.maximum(h0, 0)
        elif s == 5:
            h0 = np.cos(np.pi * np.arange(N) / N) ** 2
            h0 = np.maximum(h0, 0)
        elif s == 6:
            h0 = np.abs(rng.randn(N)) + 0.1
        elif s == 7:
            h0 = np.zeros(N)
            h0[N//4] = 1.0
            h0[3*N//4] = 1.0
        elif s == 8:
            h0 = np.exp(-((np.arange(N) - N/2)**2) / (2 * (N/10)**2))
        elif s == 9:
            h0 = np.exp(-((np.arange(N) - N/2)**2) / (2 * (N/2)**2))
        elif s == 10:
            h0 = np.sinc(np.linspace(-1, 1, N))
            h0 = np.maximum(h0, 0)
        elif s == 11:
            h0 = np.zeros(N)
            h0[N//3:2*N//3] = 1.0
        elif s == 12:
            h0 = np.zeros(N)
            h0[::2] = 1.0
        elif s == 13:
            h0 = np.zeros(N)
            h0[::3] = 1.0
        elif s == 14:
            h0 = np.concatenate([np.linspace(0, 1, N//2), np.linspace(1, 0, N - N//2)])
        else:
            h0 = np.abs(rng.randn(N)) * np.exp(-((np.arange(N) - N/2)**2) / (2 * (N/3)**2))
        
        h0 = np.maximum(h0, 0.0)
        n2 = np.sum(h0**2)
        if n2 < 1e-15:
            continue
        h0 = h0 / np.sqrt(n2)
        
        # Phase 1: SLSQP on nonsmooth objective
        if time.time() < deadline - 20:
            def objective_sl(h):
                return -compute_ratio(h)
            
            bounds = [(0.0, None)] * N
            constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h**2) - 1.0,
                            'jac': lambda h: 2 * h}]
            
            try:
                res = minimize(objective_sl, h0, method='SLSQP', bounds=bounds,
                              constraints=constraints, options={'maxiter': 100, 'ftol': 1e-12})
                h_opt = np.maximum(res.x, 0.0)
                n2 = np.sum(h_opt**2)
                if n2 > 1e-15:
                    h_opt = h_opt / np.sqrt(n2)
                    score = compute_ratio(h_opt)
                    if score > best_score:
                        best_score = score
                        best_h = h_opt.copy()
            except Exception:
                pass
        
        # Phase 2: L-BFGS-B with smooth surrogate
        if time.time() < deadline - 10:
            betas = [0.05, 0.1, 0.2]
            for beta in betas:
                if time.time() > deadline - 5:
                    break
                
                def objective_smooth(h, b=beta):
                    return smooth_objective(h, b)
                
                def gradient_smooth(h, b=beta):
                    return smooth_gradient(h, b)
                
                bounds = [(0.0, None)] * N
                try:
                    res = minimize(objective_smooth, h0, method='L-BFGS-B', bounds=bounds,
                                  jac=gradient_smooth,
                                  options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-8})
                    h_opt = np.maximum(res.x, 0.0)
                    n2 = np.sum(h_opt**2)
                    if n2 > 1e-15:
                        h_opt = h_opt / np.sqrt(n2)
                        score = compute_ratio(h_opt)
                        if score > best_score:
                            best_score = score
                            best_h = h_opt.copy()
                except Exception:
                    pass
    
    # Final polish with high precision
    if best_h is not None and time.time() < deadline - 5.0:
        # Perturbation search
        for _ in range(20):
            if time.time() > deadline - 2:
                break
            h_pert = best_h + 0.01 * rng.standard_normal(N)
            h_pert = np.maximum(h_pert, 0)
            n2 = np.sum(h_pert**2)
            if n2 < 1e-15:
                continue
            h_pert = h_pert / np.sqrt(n2)
            score = compute_ratio(h_pert)
            if score > best_score:
                best_score = score
                best_h = h_pert
        
        # High-precision SLSQP polish
        if time.time() < deadline - 3:
            def objective_final(h):
                return -compute_ratio(h)
            
            bounds = [(0.0, None)] * N
            constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h**2) - 1.0,
                            'jac': lambda h: 2 * h}]
            
            try:
                res = minimize(objective_final, best_h, method='SLSQP', bounds=bounds,
                              constraints=constraints, options={'maxiter': 300, 'ftol': 1e-14})
                h_polish = np.maximum(res.x, 0.0)
                n2 = np.sum(h_polish**2)
                if n2 > 1e-15:
                    h_polish = h_polish / np.sqrt(n2)
                    score = compute_ratio(h_polish)
                    if score > best_score:
                        best_score = score
                        best_h = h_polish
            except Exception:
                pass
    
    if best_h is None:
        return [1.0] * N
    
    return list(best_h)

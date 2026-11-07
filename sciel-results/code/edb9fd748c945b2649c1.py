import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        
        d = self.dim
        rng = np.random.default_rng(42)
        
        x = np.zeros(d)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        
        sigma = 2.0
        sigma_min = 1e-6
        sigma_max = 15.0
        
        # Sliding window for surrogate model
        hist_X = []
        hist_f = []
        max_hist = 30
        min_hist_for_surrogate = 21  # Need at least d+1=6, but 21 for stable quadratic fit
        
        calls = 1
        phase1_end = 200  # Switch to polish mode after ~200 calls
        
        for i in range(self.budget - 1):
            if time.time() > deadline:
                break
            
            # Add current point to history
            hist_X.append(x.copy())
            hist_f.append(fx)
            if len(hist_X) > max_hist:
                hist_X.pop(0)
                hist_f.pop(0)
            
            # Phase transition: reduce sigma in polish mode
            if i >= phase1_end:
                sigma = min(sigma, 0.1)
            
            # Try to use surrogate model if we have enough data
            used_surrogate = False
            if len(hist_X) >= min_hist_for_surrogate and i > 10:
                try:
                    X = np.array(hist_X)
                    f_vals = np.array(hist_f)
                    
                    # Build design matrix for quadratic: f = c0 + c1^T x + x^T C x
                    n = len(X)
                    d_quad = d * (d + 1) // 2
                    A = np.ones((n, 1 + d + d_quad))
                    
                    for k in range(n):
                        A[k, 1:1+d] = X[k]
                        idx = 1 + d
                        for ii in range(d):
                            for jj in range(ii, d):
                                A[k, idx] = X[k, ii] * X[k, jj]
                                idx += 1
                    
                    # Solve least squares
                    coeffs, _, _, _ = np.linalg.lstsq(A, f_vals, rcond=None)
                    
                    # Extract linear and quadratic coefficients
                    c1 = coeffs[1:1+d]
                    
                    # Reconstruct Hessian C from coefficients
                    C = np.zeros((d, d))
                    idx = 1 + d
                    for ii in range(d):
                        for jj in range(ii, d):
                            val = coeffs[idx]
                            C[ii, jj] = val
                            if ii != jj:
                                C[jj, ii] = val
                            idx += 1
                    
                    # Check if Hessian is positive definite (local minimum)
                    eigvals = np.linalg.eigvalsh(C)
                    
                    if eigvals[0] > 1e-8:  # Positive definite -> local min
                        # Solve C * delta = -c1/2 for the step
                        # grad f = c1 + 2 C x = 0 => C x = -c1/2
                        try:
                            x_pred = -np.linalg.solve(C, c1) / 2.0
                            x_pred = np.clip(x_pred, -5, 5)
                            
                            # Only take the step if it's reasonable
                            dist = np.linalg.norm(x_pred - x)
                            if dist < sigma * 10 and dist > 1e-10:
                                fy = func(x_pred)
                                calls += 1
                                if fy < fx:
                                    x, fx = x_pred, fy
                                    if fx < best_f:
                                        best_f = fx
                                        best_x = x.copy()
                                    used_surrogate = True
                                    # Increase sigma slightly on success
                                    sigma = min(sigma * 1.1, sigma_max)
                                else:
                                    # Surrogate was wrong, fall back to ES
                                    pass
                        except np.linalg.LinAlgError:
                            pass
                    elif eigvals[-1] < -1e-8:
                        # Local maximum, move away from prediction
                        pass
                except Exception:
                    pass
            
            if not used_surrogate:
                # Standard (1+1)-ES step
                y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma = min(sigma * 1.25, sigma_max)
                else:
                    sigma = max(sigma * 0.85, sigma_min)
            
            sigma = np.clip(sigma, sigma_min, sigma_max)
        
        return best_x, best_f

import numpy as np
import time
from scipy.optimize import minimize

N = 130

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin before 150s kill

    # 1. Initialization: Fibonacci Spiral
    def _fibonacci_sphere(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    x = _fibonacci_sphere(N)

    # 2. Define Loss Function: Soft-Max Covering Radius Surrogate
    # We use a dense set of test points to approximate the true covering radius.
    # The loss is a smooth approximation of the max distance.
    
    # Generate test points (Fibonacci spiral, denser)
    N_test = 20000
    y = _fibonacci_sphere(N_test)
    
    # Precompute y^2
    y_sq = np.sum(y**2, axis=1)[:, None]

    def _covering_loss(x):
        # x: (N, 3)
        # Calculate squared distances from all test points to all configuration points
        # d2[m, i] = |y_m - x_i|^2 = |y_m|^2 + |x_i|^2 - 2 y_m . x_i
        x_sq = np.sum(x**2, axis=1)[None, :]
        d2 = y_sq + x_sq - 2.0 * (y @ x.T)
        
        # For each test point m, find the minimum squared distance
        min_d2 = np.min(d2, axis=1)
        
        # Soft-max approximation of the maximum min_d2
        # We want to minimize the max of min_d2.
        # Let u = min_d2. We approximate max(u) with (1/tau) log(sum(exp(tau * u)))
        # To avoid overflow, we shift by max(u)
        max_u = np.max(u) if (u := min_d2).size > 0 else 0.0
        # Actually, let's just use a stable log-sum-exp
        m = np.max(min_d2)
        if m > 0:
            exp_terms = np.exp((min_d2 - m) * 50.0)  # tau = 50
            loss = (m + np.log(np.sum(exp_terms)) / 50.0)
        else:
            loss = 0.0
            
        return loss

    # 3. Optimization
    # We use L-BFGS-B which is efficient for large N.
    # We will run this from a few different starting points to avoid local minima.
    
    best_x = x.copy()
    best_loss = _covering_loss(x)
    
    # Helper to check time
    def _time_left():
        return time.time() - start_time < deadline

    # Run L-BFGS-B
    # We define the gradient numerically or analytically?
    # Analytical gradient of soft-max min-distance is complex due to min.
    # However, L-BFGS-B can use numerical gradients. For N=130, 390 variables,
    # numerical gradient is 390 function evals per step.
    # _covering_loss takes ~ (20000 * 130) ops = 2.6M ops.
    # 390 * 2.6M = 1B ops per step. Too slow for Python loop if many steps.
    
    # Better: Use a smoother surrogate or reduce test points for gradient?
    # Or use scipy.optimize.minimize with method='L-BFGS-B' and let it handle it.
    # But 150s limit is tight.
    
    # Alternative: Repulsion + Covering.
    # The "Thomson" problem (minimize sum 1/d) is a good proxy for uniform distribution.
    # But we specifically need covering.
    
    # Let's try a hybrid:
    # 1. Minimize Thomson energy (fast, smooth) to get a good initial spread.
    # 2. Polish with covering radius surrogate.
    
    # Step 1: Thomson Energy Minimization
    def _thomson_loss(x):
        d2 = np.sum(x**2, axis=1)[:, None] + np.sum(x**2, axis=1)[None, :] - 2.0 * (x @ x.T)
        np.fill_diagonal(d2, 1e-10)
        d = np.sqrt(d2)
        E = np.sum(1.0 / d)
        return E
    
    # Gradient of Thomson Energy
    def _thomson_grad(x):
        d2 = np.sum(x**2, axis=1)[:, None] + np.sum(x**2, axis=1)[None, :] - 2.0 * (x @ x.T)
        np.fill_diagonal(d2, 1e-10)
        d = np.sqrt(d2)
        inv_d3 = 1.0 / (d2 * d)
        np.fill_diagonal(inv_d3, 0.0)
        grad = - (x * np.sum(inv_d3, axis=1)[:, None]) + (inv_d3 @ x)
        return grad

    # Optimize Thomson Energy first
    res = minimize(_thomson_loss, x, jac=_thomson_grad, method='L-BFGS-B',
                   options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
    x = res.x
    # Normalize
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    
    if not _time_left():
        return x

    # Step 2: Covering Radius Polish
    # Use a smaller set of test points for the gradient to speed up L-BFGS-B
    # The test points should be the "worst-case" points, i.e., centers of the largest caps.
    # We can approximate this by taking the test points y that are far from x.
    
    # For the polish, we use a loss that is the sum of the largest K min-distances,
    # or a soft-max.
    # Let's use a soft-max with a moderate tau.
    
    # Reduce test points for speed in the polish phase
    N_test_polish = 5000
    y_polish = _fibonacci_sphere(N_test_polish)
    y_sq_polish = np.sum(y_polish**2, axis=1)[:, None]

    def _polish_loss(x):
        x_sq = np.sum(x**2, axis=1)[None, :]
        d2 = y_sq_polish + x_sq - 2.0 * (y_polish @ x.T)
        min_d2 = np.min(d2, axis=1)
        
        # Soft max of min_d2
        # We want to minimize the max of min_d2 (squared distance)
        # Let's use a power mean approximation: (mean(u^p))^(1/p)
        # Or log-sum-exp.
        # Log-sum-exp is smooth.
        m = np.max(min_d2)
        if m > 1e-10:
            exp_terms = np.exp((min_d2 - m) * 20.0)
            loss = m + np.log(np.sum(exp_terms)) / 20.0
        else:
            loss = 0.0
        return loss

    def _polish_grad(x):
        # Numerical gradient is too slow.
        # Let's use a simple gradient descent with analytic gradient of the soft-max.
        # d/dx_i L = sum_m w_m * d/dx_i min_d2[m]
        # where w_m = exp(tau * min_d2[m]) / sum(exp(tau * min_d2))
        # d/dx_i min_d2[m] is non-zero only if i is the argmin for m.
        # If i == argmin_m, d/dx_i |y_m - x_i|^2 = -2(y_m - x_i)
        
        x_sq = np.sum(x**2, axis=1)[None, :]
        d2 = y_sq_polish + x_sq - 2.0 * (y_polish @ x.T)
        min_d2 = np.min(d2, axis=1)
        argmins = np.argmin(d2, axis=1)
        
        tau = 20.0
        m = np.max(min_d2)
        exp_terms = np.exp((min_d2 - m) * tau)
        sum_exp = np.sum(exp_terms)
        w = exp_terms / sum_exp
        
        grad = np.zeros_like(x)
        diffs = y_polish - x[argmins]
        contribs = -2.0 * w[:, None] * diffs
        np.add.at(grad, argmins, contribs)
        
        return grad

    # Run L-BFGS-B on the polish loss
    # Since we have an analytic gradient, this should be faster.
    res2 = minimize(_polish_loss, x, jac=_polish_grad, method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-6})
    
    if res2.success or res2.fun < best_loss:
        x = res2.x
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        
    # Final check and return
    if not _time_left():
        return x

    # One more pass with higher precision if time permits
    # Increase test points and tau for final polish
    if _time_left():
        N_test_final = 10000
        y_final = _fibonacci_sphere(N_test_final)
        y_sq_final = np.sum(y_final**2, axis=1)[:, None]
        
        def _final_loss(x):
            x_sq = np.sum(x**2, axis=1)[None, :]
            d2 = y_sq_final + x_sq - 2.0 * (y_final @ x.T)
            min_d2 = np.min(d2, axis=1)
            m = np.max(min_d2)
            if m > 1e-10:
                exp_terms = np.exp((min_d2 - m) * 50.0)
                loss = m + np.log(np.sum(exp_terms)) / 50.0
            else:
                loss = 0.0
            return loss

        def _final_grad(x):
            x_sq = np.sum(x**2, axis=1)[None, :]
            d2 = y_sq_final + x_sq - 2.0 * (y_final @ x.T)
            min_d2 = np.min(d2, axis=1)
            argmins = np.argmin(d2, axis=1)
            
            tau = 50.0
            m = np.max(min_d2)
            exp_terms = np.exp((min_d2 - m) * tau)
            sum_exp = np.sum(exp_terms)
            w = exp_terms / sum_exp
            
            grad = np.zeros_like(x)
            diffs = y_final - x[argmins]
            contribs = -2.0 * w[:, None] * diffs
            np.add.at(grad, argmins, contribs)
            
            return grad

        res3 = minimize(_final_loss, x, jac=_final_grad, method='L-BFGS-B',
                        options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-6})
        
        if res3.success or res3.fun < best_loss:
            x = res3.x
            x = x / np.linalg.norm(x, axis=1, keepdims=True)

    return x

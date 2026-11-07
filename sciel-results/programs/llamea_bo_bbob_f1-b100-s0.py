import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        n = self.dim
        budget = self.budget
        time_limit = 290.0
        
        # State
        best_x = np.zeros(n)
        best_f = float('inf')
        calls = 0
        
        # Helper to check time
        def time_exceeded():
            return time.time() - start_time > time_limit

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or time_exceeded():
                return None
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # 1. Build initial quadratic model (2n+1 points)
        # Start at center
        x0 = np.zeros(n)
        f0 = evaluate(x0)
        if f0 is None:
            return best_x, best_f

        # Create simplex points: x0 + delta * e_i
        delta = 1.0
        points = [x0]
        values = [f0]
        for i in range(n):
            xi = x0.copy()
            xi[i] += delta
            fi = evaluate(xi)
            if fi is None:
                break
            points.append(xi)
            values.append(fi)
        
        if len(points) < n + 1:
            # Not enough budget for model, just return best
            return best_x, best_f

        # Fit quadratic model: f(x) ~ c + g^T x + 0.5 x^T H x
        # We have n+1 points. This is underdetermined for full Hessian (n(n+1)/2 + n + 1 params).
        # But for Sphere, H is diagonal (or close if rotated, but we can approximate).
        # Actually, let's use a simpler approach:
        # Estimate gradient and Hessian diagonal from these points.
        
        # Gradient estimate: (f(x_i) - f(x_0)) / delta
        g = np.zeros(n)
        for i in range(n):
            g[i] = (values[i+1] - f0) / delta
            
        # Hessian diagonal estimate: (f(x_i) - f(x_0) - g[i]*delta) / (0.5 * delta^2) ?
        # f(x_i) = f(x_0) + g_i delta + 0.5 H_ii delta^2
        # H_ii = 2 * (f(x_i) - f(x_0) - g_i delta) / delta^2
        # But g_i is estimated from the same points, so this is noisy.
        # Let's use a 3-point stencil if budget allows? No, we want to start fast.
        # Let's assume H is proportional to Identity initially? Or estimate diagonal.
        
        # A better initial model:
        # If we assume f(x) = ||x - z||^2 + c, then grad = 2(x-z), H = 2I.
        # We can estimate z from the first step.
        # x0 = 0. f(0) = ||z||^2 + c.
        # x_i = delta e_i. f(x_i) = ||delta e_i - z||^2 + c = delta^2 - 2 delta z_i + ||z||^2 + c.
        # f(x_i) - f(0) = delta^2 - 2 delta z_i.
        # z_i = (delta^2 - (f(x_i) - f(0))) / (2 delta).
        
        z_est = np.zeros(n)
        for i in range(n):
            z_est[i] = (delta**2 - (values[i+1] - f0)) / (2 * delta)
            
        # Now we have an estimate of the minimizer z_est.
        # We can switch to Newton's method or just CMA-ES around z_est.
        # Since Sphere is quadratic, Newton's method should converge in 1-2 steps if H is correct.
        # H is 2I for unrotated. If rotated, H is 2 R^T R = 2I.
        # So H is always 2I for Sphere!
        # The only unknowns are z and c.
        # So we can solve for z more accurately by doing a few more evaluations?
        # Or just use CMA-ES with initial mean = z_est and sigma = 0.5.
        
        # Let's refine z_est with a few more points?
        # Or just start CMA-ES.
        
        # Strategy: Use the estimated z as the initial mean for CMA-ES.
        # This should give a huge head start.
        
        # Initialize CMA-ES state
        mean = z_est.copy()
        sigma = 0.5  # Start small since we are close
        C = np.eye(n)
        L = np.eye(n)
        
        # CMA-ES Parameters
        lambda_ = 4 + int(3 * np.log(n))
        mu = lambda_ // 2
        weights = np.zeros(mu)
        for i in range(mu):
            weights[i] = np.log(mu + 0.5) - np.log(i + 1)
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mueff + 2) / (n + mueff + 5)
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Evaluate the initial mean to update best
        fm = evaluate(mean)
        if fm is None:
            return best_x, best_f
            
        while calls < budget:
            if time_exceeded():
                break
                
            # Generate population
            population = np.zeros((lambda_, n))
            for i in range(lambda_):
                z = np.random.randn(n)
                y = sigma * (L @ z)
                x = mean + y
                x = np.clip(x, -5, 5)
                population[i] = x
                
            f_values = np.zeros(lambda_)
            for i in range(lambda_):
                fx = evaluate(population[i])
                if fx is None:
                    break
                f_values[i] = fx
                if calls >= budget:
                    break
                    
            if calls >= budget:
                break
            
            # Sort
            idx = np.argsort(f_values)
            population = population[idx]
            f_values = f_values[idx]
            
            # Update mean
            new_mean = np.sum(weights[:, np.newaxis] * population[:mu], axis=0)
            
            # Update step size and covariance
            try:
                L_old = np.linalg.cholesky(C)
                B_old = np.linalg.inv(L_old.T)
            except:
                B_old = np.eye(n)
                
            diff = (mean - new_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (B_old @ diff)
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * (B_old @ diff)
            
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            d_sigma = 1 + 2 * max(0, (mueff - 1) / (n + 1)) + c_sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            h_c = 1.0
            C_update = np.zeros((n, n))
            for i in range(mu):
                y_i = (population[i] - mean) / sigma
                C_update += weights[i] * np.outer(y_i, y_i)
                
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_c) * np.eye(n)) + c_mu * C_update
            C = (C + C.T) / 2.0
            mean = new_mean
            
            try:
                L = np.linalg.cholesky(C)
            except:
                C += 1e-8 * np.eye(n)
                L = np.linalg.cholesky(C)

        return best_x, best_f

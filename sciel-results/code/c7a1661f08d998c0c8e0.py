import numpy as np
import time
from scipy.optimize import minimize

N = 150

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    # Energy function (vectorized)
    def energy(X):
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        dist_sq = np.maximum(dist_sq, 1e-12)
        dist = np.sqrt(dist_sq)
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        energy_val = -np.sum(np.log(dist[mask]))
        return energy_val

    # Gradient function (vectorized)
    def gradient(X):
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        dist_sq = np.maximum(dist_sq, 1e-12)
        
        inv_dist_sq = 1.0 / dist_sq
        np.fill_diagonal(inv_dist_sq, 0.0)
        
        grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        return grad

    def energy_and_gradient(X):
        """Combined energy and gradient for scipy minimize."""
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        dist_sq = np.maximum(dist_sq, 1e-12)
        
        # Energy
        dist = np.sqrt(dist_sq)
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        energy_val = -np.sum(np.log(dist[mask]))
        
        # Gradient
        inv_dist_sq = 1.0 / dist_sq
        np.fill_diagonal(inv_dist_sq, 0.0)
        grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        
        return energy_val, grad.ravel()

    def lbfgs_optimize(X_init, time_limit, maxiter=500, gtol=1e-10):
        """L-BFGS-B optimization with time guard."""
        X = X_init.copy()
        
        def fun_and_grad(x):
            p = x.reshape(-1, 3)
            norms = np.linalg.norm(p, axis=1, keepdims=True)
            p = p / norms
            e, g = energy_and_gradient(p)
            return e, g.ravel()
        
        x0 = X.ravel()
        best_x = x0.copy()
        best_e = energy(X)
        
        try:
            def callback(xk):
                nonlocal best_x, best_e
                p = xk.reshape(-1, 3)
                norms = np.linalg.norm(p, axis=1, keepdims=True)
                p = p / norms
                e = energy(p)
                if e < best_e:
                    best_e = e
                    best_x = xk.copy()
                if time.time() > time_limit:
                    return True
                return False
            
            res = minimize(fun_and_grad, x0, method='L-BFGS-B',
                          callback=callback,
                          options={'maxiter': maxiter, 'gtol': gtol, 'maxfun': maxiter * 3})
            
            if res.fun < best_e:
                p = res.x.reshape(-1, 3)
                norms = np.linalg.norm(p, axis=1, keepdims=True)
                p = p / norms
                e = energy(p)
                if e < best_e:
                    best_e = e
                    best_x = res.x.copy()
        except Exception:
            pass
        
        p = best_x.reshape(-1, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        p = p / norms
        return p, best_e

    def momentum_optimize(X_init, time_limit):
        """Gradient descent with momentum and adaptive step size."""
        X = X_init.copy()
        v = np.zeros_like(X)
        beta = 0.9
        eta = 0.05
        best_energy = energy(X)
        best_X = X.copy()
        
        t_start = time.time()
        
        for t in range(20000):
            if time.time() > time_limit:
                break
            
            g = gradient(X)
            
            # Momentum update
            v = beta * v - eta * g
            X_new = X + v
            
            # Normalize
            norms = np.linalg.norm(X_new, axis=1, keepdims=True)
            X_new = X_new / norms
            
            e_new = energy(X_new)
            
            if e_new < best_energy:
                best_energy = e_new
                best_X = X_new.copy()
                X = X_new
                # Slowly decay learning rate
                eta *= 0.9995
            else:
                # Reduce learning rate
                eta *= 0.5
                if eta < 1e-8:
                    break
            
            # Additional decay
            eta *= 0.999
            
        return best_X, best_energy

    def polish(X_init, time_limit):
        """High-precision polish with very tight tolerances."""
        X = X_init.copy()
        
        def fun_and_grad(x):
            p = x.reshape(-1, 3)
            norms = np.linalg.norm(p, axis=1, keepdims=True)
            p = p / norms
            e, g = energy_and_gradient(p)
            return e, g.ravel()
        
        x0 = X.ravel()
        best_x = x0.copy()
        best_e = energy(X)
        
        try:
            def callback(xk):
                nonlocal best_x, best_e
                p = xk.reshape(-1, 3)
                norms = np.linalg.norm(p, axis=1, keepdims=True)
                p = p / norms
                e = energy(p)
                if e < best_e:
                    best_e = e
                    best_x = xk.copy()
                if time.time() > time_limit:
                    return True
                return False
            
            res = minimize(fun_and_grad, x0, method='L-BFGS-B',
                          callback=callback,
                          options={'maxiter': 2000, 'gtol': 1e-14, 'ftol': 1e-15, 'maxfun': 6000})
            
            if res.fun < best_e:
                p = res.x.reshape(-1, 3)
                norms = np.linalg.norm(p, axis=1, keepdims=True)
                p = p / norms
                e = energy(p)
                if e < best_e:
                    best_e = e
                    best_x = res.x.copy()
        except Exception:
            pass
        
        p = best_x.reshape(-1, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        p = p / norms
        return p, best_e

    # Generate initial configurations
    def fibonacci_spiral(n, offset=0.0):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i + offset
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    rng = np.random.default_rng(42)
    
    # Multiple Fibonacci spirals with different offsets
    inits = []
    inits.append(fibonacci_spiral(N, offset=0.0))
    inits.append(fibonacci_spiral(N, offset=0.5))
    inits.append(fibonacci_spiral(N, offset=1.0))
    inits.append(fibonacci_spiral(N, offset=1.5))
    inits.append(fibonacci_spiral(N, offset=2.0))
    
    # Random initializations
    for k in range(5):
        X_rand = rng.normal(size=(N, 3))
        X_rand = X_rand / np.linalg.norm(X_rand, axis=1, keepdims=True)
        inits.append(X_rand)
    
    best_overall_X = inits[0]
    best_overall_energy = energy(inits[0])
    
    # Phase 1: Quick momentum optimization on all inits to find good candidates
    # Allocate time: 10s per init for quick pass
    time_per_init = 10.0
    for init in inits:
        if time.time() > deadline - 30:  # Leave 30s for final optimization
            break
        
        X_opt, e_opt = momentum_optimize(init, time.time() + time_per_init)
        if e_opt < best_overall_energy:
            best_overall_energy = e_opt
            best_overall_X = X_opt
    
    # Phase 2: L-BFGS-B optimization on the best candidate
    if time.time() < deadline - 15:
        X_opt, e_opt = lbfgs_optimize(best_overall_X, time.time() + 25.0, maxiter=1000, gtol=1e-12)
        if e_opt < best_overall_energy:
            best_overall_energy = e_opt
            best_overall_X = X_opt
    
    # Phase 3: Try L-BFGS on other good inits if time remains
    # Re-sort inits by their momentum-optimized energy
    candidates = []
    for init in inits:
        if time.time() > deadline - 20:
            break
        X_opt, e_opt = momentum_optimize(init, time.time() + 5.0)
        candidates.append((e_opt, X_opt))
    
    candidates.sort(key=lambda x: x[0])
    
    for e_cand, X_cand in candidates[:3]:
        if time.time() > deadline - 10:
            break
        if e_cand >= best_overall_energy:
            continue
        X_opt, e_opt = lbfgs_optimize(X_cand, time.time() + 15.0, maxiter=500, gtol=1e-12)
        if e_opt < best_overall_energy:
            best_overall_energy = e_opt
            best_overall_X = X_opt
    
    # Phase 4: Final high-precision polish
    if time.time() < deadline - 5:
        X_polished, e_polished = polish(best_overall_X, deadline - 1.0)
        if e_polished < best_overall_energy:
            best_overall_energy = e_polished
            best_overall_X = X_polished
    
    # Ensure all points are on unit sphere and valid
    norms = np.linalg.norm(best_overall_X, axis=1, keepdims=True)
    best_overall_X = best_overall_X / norms
    
    # Check for very close points
    diff = best_overall_X[:, np.newaxis, :] - best_overall_X[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    min_dist = np.sqrt(np.min(dist_sq))
    if min_dist < 1e-9:
        # Fallback to Fibonacci if points are too close
        best_overall_X = fibonacci_spiral(N)
        norms = np.linalg.norm(best_overall_X, axis=1, keepdims=True)
        best_overall_X = best_overall_X / norms
    
    return best_overall_X

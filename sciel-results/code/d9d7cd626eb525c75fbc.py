import math
import time
import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    t_start = time.time()
    # Evaluator kills at 150s. Leave a 5s margin.
    DEADLINE = t_start + 145.0

    def fib_sphere():
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = math.pi * (1.0 + math.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def norm_rows(x):
        n = np.linalg.norm(x, axis=1, keepdims=True)
        n = np.maximum(n, 1e-15)
        return x / n

    def log_energy(pts):
        # Efficient energy calculation
        dots = pts @ pts.T
        d2 = 2.0 - 2.0 * dots
        np.fill_diagonal(d2, 1.0)
        d2 = np.maximum(d2, 1e-15)
        dist = np.sqrt(d2)
        iu = np.triu_indices(N, k=1)
        # Use fsum for higher precision as requested by task contract
        return -math.fsum(np.log(dist[iu]))

    def gradient_log_energy(pts):
        # Efficient O(N^2) gradient calculation
        dots = pts @ pts.T
        d2 = 2.0 - 2.0 * dots
        np.fill_diagonal(d2, 1.0)
        d2 = np.maximum(d2, 1e-15)
        
        inv_d2 = 1.0 / d2
        np.fill_diagonal(inv_d2, 0.0)
        
        # grad_i = - sum_{j!=i} (x_i - x_j) / d_ij^2
        # = - x_i * sum_{j!=i} (1/d_ij^2) + sum_{j!=i} x_j / d_ij^2
        sum_inv_d2 = inv_d2.sum(axis=1)
        sum_xj_inv_d2 = (pts.T @ inv_d2).T
        
        grad = -(pts * sum_inv_d2[:, np.newaxis] - sum_xj_inv_d2)
        
        # Project to tangent space
        grad_dot_xi = np.sum(grad * pts, axis=1, keepdims=True)
        grad_tan = grad - grad_dot_xi * pts
        
        return grad_tan

    def project(x):
        return norm_rows(x)

    def riemann_gd(x0, max_iter=2000, tol=1e-10, deadline=None):
        x = project(x0.copy())
        f = log_energy(x)
        best_x = x.copy()
        best_f = f
        
        for _ in range(max_iter):
            if deadline is not None and time.time() > deadline:
                break
            
            g = gradient_log_energy(x)
            gnorm = np.linalg.norm(g, ord=np.inf)
            if gnorm < tol:
                break

            # Step size heuristic
            step = 1.0 / (1.0 + 0.5 * gnorm)
            
            accepted = False
            for _ in range(30):
                new_x = project(x - step * g)
                new_f = log_energy(new_x)
                
                if np.isfinite(new_f) and new_f < f:
                    # Simple Armijo-like condition
                    if new_f < f - 1e-4 * step * gnorm * gnorm or new_f < f - 1e-8:
                        accepted = True
                        break
                step *= 0.5
            
            if accepted:
                x = new_x
                f = new_f
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            else:
                break
                
        return best_x, best_f

    def lbfgs_polish(x0, maxiter=500, deadline=None):
        if time.time() > deadline - 2:
            return x0, log_energy(x0)
            
        def obj_and_grad(v):
            x = project(v.reshape(-1, 3))
            f = log_energy(x)
            g = gradient_log_energy(x).ravel()
            return f, g

        try:
            res = minimize(
                obj_and_grad, 
                x0.ravel(), 
                jac=True, 
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 2000}
            )
            xr = project(res.x.reshape(-1, 3))
            e = log_energy(xr)
            return xr, e
        except Exception:
            return x0, log_energy(x0)

    # Phase 1: Main optimization from Fibonacci start
    best_pts = fib_sphere()
    best_e = log_energy(best_pts)
    
    # Quick RGD to get into basin
    deadline_rgd = time.time() + 30.0
    pts, e = riemann_gd(best_pts, max_iter=1000, tol=1e-8, deadline=deadline_rgd)
    if e < best_e:
        best_pts, best_e = pts, e

    # L-BFGS polish
    deadline_lbfgs = time.time() + 20.0
    pts, e = lbfgs_polish(best_pts, maxiter=500, deadline=deadline_lbfgs)
    if e < best_e:
        best_pts, best_e = pts, e

    # Phase 2: Random restarts
    rng = np.random.default_rng(42)
    
    # Define a set of diverse starting points
    starts = []
    # 1. Best solution perturbed
    starts.append(project(best_pts + 0.1 * rng.normal(size=(N, 3))))
    starts.append(project(best_pts + 0.05 * rng.normal(size=(N, 3))))
    
    # 2. Random
    for _ in range(5):
        starts.append(project(rng.normal(size=(N, 3))))
        
    # 3. Fibonacci with slight rotation
    base_fib = fib_sphere()
    # Simple rotation
    theta = 0.5
    R = np.array([
        [math.cos(theta), -math.sin(theta), 0],
        [math.sin(theta), math.cos(theta), 0],
        [0, 0, 1]
    ])
    starts.append(project(base_fib @ R.T))
    
    for start in starts:
        if time.time() > DEADLINE - 15:
            break
            
        # Quick RGD
        deadline_rgd = min(time.time() + 10.0, DEADLINE - 10.0)
        pts, e = riemann_gd(start, max_iter=500, tol=1e-6, deadline=deadline_rgd)
        
        # Quick L-BFGS
        if time.time() < DEADLINE - 5:
            pts, e2 = lbfgs_polish(pts, maxiter=300, deadline=DEADLINE - 2)
            if e2 < best_e:
                best_pts, best_e = pts, e2

    # Final validation and safety
    best_pts = project(best_pts)
    if not np.all(np.isfinite(best_pts)):
        best_pts = fib_sphere()
        
    # Check for coincident points
    d = best_pts[:, None, :] - best_pts[None, :, :]
    dsq = np.einsum('ijk,ijk->ij', d, d)
    np.fill_diagonal(dsq, 1.0)
    if np.min(dsq) < 1e-18:
        best_pts += 1e-8 * rng.normal(size=(N, 3))
        best_pts = project(best_pts)

    return best_pts

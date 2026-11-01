import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150

def _halton_sphere(n):
    """Generate points using Halton sequence projected to sphere."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    points = []
    for i in range(1, n + 1):
        x = halton(i, 2)
        y = halton(i, 3)
        z = halton(i, 5)
        # Map to sphere: use spherical coordinates
        theta = 2.0 * math.pi * x
        phi = math.acos(2.0 * y - 1.0)
        px = math.sin(phi) * math.cos(theta)
        py = math.sin(phi) * math.sin(theta)
        pz = math.cos(phi)
        points.append([px, py, pz])
    
    pts = np.array(points)
    # Normalize
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    pts = pts / norms
    return pts

def _fibonacci_sphere(n):
    """Generate points using Fibonacci spiral on sphere."""
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    points = []
    for i in range(n):
        y = 1.0 - (i / float(n - 1)) * 2.0
        radius = math.sqrt(1.0 - y * y)
        theta = golden_angle * i
        px = math.cos(theta) * radius
        py = math.sin(theta) * radius
        pz = y
        points.append([px, py, pz])
    
    pts = np.array(points)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    pts = pts / norms
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 138.0  # 12s safety margin for 150s limit

    # Try multiple initializations and pick the best
    initializations = [_halton_sphere(N), _fibonacci_sphere(N)]
    
    def calc_energy(pts):
        """Vectorized log energy computation."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        dist_sq = np.maximum(dist_sq, 1e-24)
        # log(1/d) = -0.5 * log(d^2)
        log_d2 = np.log(dist_sq)
        np.fill_diagonal(log_d2, 0.0)
        energy = -0.25 * np.sum(log_d2)
        return energy

    def gradient(pts):
        """Compute gradient of log energy on sphere (projected to tangent)."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        dist_sq = np.maximum(dist_sq, 1e-24)
        np.fill_diagonal(dist_sq, 1.0)
        # Gradient of -0.25 * sum log(d2) w.r.t. x_i
        # d/dx_i [-0.25 * log(|x_i - x_j|^2)] = -(x_i-x_j)/(2*d2)
        # Sum over j != i
        forces = -0.5 * np.sum(diff / dist_sq[:, :, np.newaxis], axis=1)
        # Project to tangent space
        dot_products = np.sum(forces * pts, axis=1)
        proj_forces = forces - dot_products[:, np.newaxis] * pts
        return proj_forces

    def _riemannian_cg(pts, max_iters=500, tol=1e-12):
        """Riemannian conjugate gradient on the sphere with proper beta."""
        x = pts.copy()
        g = gradient(x)
        g_norm = np.linalg.norm(g)
        if g_norm < tol:
            return x
        p = -g.copy()
        
        for k in range(max_iters):
            if time.time() > deadline - 15.0:
                break
            
            # Line search (backtracking with Armijo)
            alpha = 0.1
            f_k = calc_energy(x)
            slope = np.sum(g * p)
            if slope >= 0:
                # Restart
                p = -g.copy()
                slope = np.sum(g * p)
            
            for _ in range(60):
                x_trial = x + alpha * p
                norms = np.linalg.norm(x_trial, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-15)
                x_trial = x_trial / norms
                
                f_trial = calc_energy(x_trial)
                if f_trial < f_k + 1e-4 * alpha * slope:
                    break
                alpha *= 0.5
            
            x_new = x_trial
            g_new = gradient(x_new)
            g_new_norm = np.linalg.norm(g_new)
            
            if g_new_norm < tol:
                x = x_new
                break
            
            # Polak-Ribiere beta
            beta = max(0.0, np.sum(g_new * (g_new - g)) / (g_norm * g_norm + 1e-30))
            
            # Riemannian transport: project p to tangent at x_new
            dot_p_x = np.sum(p * x_new, axis=1)
            p_tangent = p - dot_p_x[:, np.newaxis] * x_new
            
            p = -g_new + beta * p_tangent
            
            x = x_new
            g = g_new
            g_norm = g_new_norm
        
        return x

    # Phase 0: Basin hopping to find better basins
    def _basin_hopping(pts, n_restarts=5, max_time=60.0):
        """Simulated annealing style basin hopping to escape local minima."""
        best = pts.copy()
        best_e = calc_energy(pts)
        t_start = time.time()
        
        for restart in range(n_restarts):
            if time.time() > t_start + max_time or time.time() > deadline - 30.0:
                break
            
            # Perturb: random rotation + small noise
            rng = np.random.RandomState(42 + restart * 1000)
            # Random rotation matrix
            A = rng.randn(3, 3)
            Q, _ = np.linalg.qr(A)
            if np.linalg.det(Q) < 0:
                Q[:, 0] *= -1
            perturbed = pts @ Q.T
            # Add small noise
            noise = rng.randn(N, 3) * 0.05
            perturbed = perturbed + noise
            norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-15)
            perturbed = perturbed / norms
            
            # Local optimization
            local = _riemannian_cg(perturbed, max_iters=150, tol=1e-8)
            local_e = calc_energy(local)
            
            if local_e < best_e:
                best_e = local_e
                best = local.copy()
        
        return best, best_e
    
    # Try multiple initializations with basin hopping
    best_energy = float('inf')
    best_pts = None
    
    for init in initializations:
        e = calc_energy(init)
        if e < best_energy:
            best_energy = e
            best_pts = init.copy()
    
    # Basin hopping from best init
    if time.time() < deadline - 50.0:
        bh_pts, bh_e = _basin_hopping(best_pts, n_restarts=8, max_time=40.0)
        if bh_e < best_energy:
            best_energy = bh_e
            best_pts = bh_pts
    
    # Phase 1: Riemannian CG for main optimization
    if time.time() < deadline - 35.0:
        pts = _riemannian_cg(best_pts, max_iters=400, tol=1e-10)
        e = calc_energy(pts)
        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()
    
    # Phase 2: L-BFGS-B polish
    if time.time() < deadline - 20.0:
        x0 = best_pts.flatten()
        
        def objective(x):
            p = x.reshape(N, 3)
            norms = np.linalg.norm(p, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-15)
            p = p / norms
            return calc_energy(p)
        
        def gradient_func(x):
            p = x.reshape(N, 3)
            norms = np.linalg.norm(p, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-15)
            p = p / norms
            return gradient(p).flatten()
        
        try:
            result = minimize(
                objective, x0, jac=gradient_func,
                method='L-BFGS-B',
                options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100}
            )
            if result.fun < best_energy:
                res_pts = result.x.reshape(N, 3)
                norms = np.linalg.norm(res_pts, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-15)
                res_pts = res_pts / norms
                best_energy = result.fun
                best_pts = res_pts
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish with very tight tolerances
    if time.time() < deadline - 5.0:
        x0 = best_pts.flatten()
        
        def objective_final(x):
            p = x.reshape(N, 3)
            norms = np.linalg.norm(p, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-15)
            p = p / norms
            return calc_energy(p)
        
        def gradient_final(x):
            p = x.reshape(N, 3)
            norms = np.linalg.norm(p, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-15)
            p = p / norms
            return gradient(p).flatten()
        
        try:
            result2 = minimize(
                objective_final, x0, jac=gradient_final,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 200}
            )
            if result2.fun < best_energy:
                res_pts = result2.x.reshape(N, 3)
                norms = np.linalg.norm(res_pts, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-15)
                res_pts = res_pts / norms
                best_energy = result2.fun
                best_pts = res_pts
        except Exception:
            pass

    # Final normalization to ensure unit sphere
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    best_pts = best_pts / norms

    return best_pts

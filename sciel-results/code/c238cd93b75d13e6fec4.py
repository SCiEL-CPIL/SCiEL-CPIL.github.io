"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

def _energy_and_grad(pts):
    """
    Vectorized computation of logarithmic energy and gradient on the sphere.
    pts: (N, 3) array of points on the unit sphere.
    Returns: (energy, grad) where grad is the projected gradient in R^{N*3}.
    """
    # Compute pairwise squared distances
    diff = pts[:, None, :] - pts[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Avoid log(0) by clipping
    dist_sq = np.clip(dist_sq, 1e-12, None)
    
    # Energy: sum_{i<j} log(1/d) = -0.5 * sum_{i<j} log(d_sq)
    # S = sum_{i != j} log(d_sq_ij). Then sum_{i<j} log(d_sq_ij) = S/2.
    # E = -0.5 * (S/2) = -0.25 * S.
    
    # Gradient computation:
    # dE/dx_i = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    w = inv_dist_sq
    sum_w = np.sum(w, axis=1)
    
    term1 = (pts.T @ w).T
    term2 = pts * sum_w[:, None]
    
    grad_raw = term1 - term2
    
    # Project gradient to tangent space of the sphere
    dot = np.sum(grad_raw * pts, axis=1)
    grad_tan = grad_raw - dot[:, None] * pts
    
    # Compute energy
    log_dist_sq = np.log(dist_sq)
    # Mask diagonal to 0 to avoid contribution from self
    log_dist_sq = np.where(np.eye(N, dtype=bool), 0.0, log_dist_sq)
    
    energy = -0.25 * np.sum(log_dist_sq)
    
    return energy, grad_tan

def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _initial_fibonacci():
    """Golden-angle (Fibonacci) spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _log_energy_fsum(pts):
    """Compute log(1/d) energy sum over all pairs using chordal distances with math.fsum."""
    diff = pts[:, None, :] - pts[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    iu = np.triu_indices(N, k=1)
    d_sq = dist_sq[iu]
    d_sq = np.clip(d_sq, 1e-30, None)
    terms = [-0.5 * math.log(di) for di in d_sq]
    return math.fsum(terms)

def _conjugate_gradient(pts, max_iter=1000, tol=1e-12, deadline=None):
    """
    Conjugate Gradient method on the sphere with improved line search.
    """
    x = pts.copy()
    energy, grad = _energy_and_grad(x)
    p = -grad
    
    # Project p to tangent space
    dot_p_x = np.sum(p * x, axis=1)
    p = p - dot_p_x[:, None] * x
    
    # Store previous gradient for Polak-Ribiere
    grad_prev = grad.copy()
    
    for k in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        # Line search: minimize f(x + alpha * p)
        # Use backtracking line search with Wolfe conditions
        alpha = 0.1
        c1 = 1e-4
        c2 = 0.9
        rho = 0.5
        
        dir_deriv = np.sum(grad * p)
        
        # If direction is not a descent direction, restart
        if dir_deriv > 0:
            p = -grad
            dot_p_x = np.sum(p * x, axis=1)
            p = p - dot_p_x[:, None] * x
            dir_deriv = np.sum(grad * p)
            if dir_deriv > 0:
                break
            
        found_step = False
        for _ in range(60):
            x_new = x + alpha * p
            x_new = _project_to_sphere(x_new)
            e_new, g_new = _energy_and_grad(x_new)
            
            # Armijo condition (sufficient decrease)
            armijo = e_new <= energy + c1 * alpha * dir_deriv
            # Strong Wolfe condition (curvature)
            curvature = np.sum(g_new * p)
            wolfe = abs(curvature) <= c2 * abs(dir_deriv)
            
            if armijo and wolfe:
                found_step = True
                break
            elif armijo:
                # Accept if Armijo is met even if Wolfe isn't (relaxed)
                found_step = True
                break
            
            alpha *= rho
            
        if not found_step:
            # If line search fails, try a smaller step or break
            break
        
        # Update
        x = x_new
        energy = e_new
        grad = g_new
        
        # Check convergence
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        
        # Compute beta (Polak-Ribiere)
        grad_dot = np.sum(grad * (grad - grad_prev))
        grad_prev_dot = np.sum(grad_prev * grad_prev)
        
        if grad_prev_dot < 1e-16:
            beta = 0.0
        else:
            beta = max(0.0, grad_dot / grad_prev_dot)
            
        # New direction
        p_new = -grad + beta * p
        
        # Project p_new to tangent space
        dot_p_new_x = np.sum(p_new * x, axis=1)
        p_new = p_new - dot_p_new_x[:, None] * x
        
        # Check if p_new is a descent direction
        if np.sum(grad * p_new) > 0:
            beta = 0.0 # Restart
            p_new = -grad
            dot_p_new_x = np.sum(p_new * x, axis=1)
            p_new = p_new - dot_p_new_x[:, None] * x
            
        p = p_new
        grad_prev = grad.copy()
        
    return x

def _simulated_annealing(pts, deadline, n_steps=20000):
    """
    Simulated annealing for escaping local minima.
    """
    x = pts.copy()
    energy = _energy_and_grad(x)[0]
    best_x = x.copy()
    best_energy = energy
    
    T = 0.5
    cooling = 0.995
    step_size = 0.05
    
    rng = np.random.default_rng(12345)
    
    for i in range(n_steps):
        if deadline and time.time() > deadline:
            break
        
        # Random perturbation of one or two points
        n_perturb = 1 if rng.random() < 0.7 else 2
        idx = rng.choice(N, size=n_perturb, replace=False)
        
        x_new = x.copy()
        for idx_i in idx:
            noise = rng.normal(size=3) * step_size
            x_new[idx_i] += noise
            x_new[idx_i] /= np.linalg.norm(x_new[idx_i])
        
        new_energy = _energy_and_grad(x_new)[0]
        delta_e = new_energy - energy
        
        if delta_e < 0 or rng.random() < math.exp(-delta_e / max(T, 1e-10)):
            x = x_new
            energy = new_energy
            if energy < best_energy:
                best_energy = energy
                best_x = x.copy()
        
        T *= cooling
        step_size = max(0.001, step_size * 0.999)
    
    return best_x

def _optimize_lbfgs(start_pts, deadline, maxiter=5000):
    """Run L-BFGS optimization on the sphere."""
    x0 = start_pts.flatten()
    
    def objective(x):
        if deadline and time.time() > deadline:
            return 1e10
        pts = x.reshape(N, 3)
        pts = _project_to_sphere(pts)
        e, _ = _energy_and_grad(pts)
        return e
    
    def jac(x):
        if deadline and time.time() > deadline:
            return np.ones(N * 3) * 1e6
        pts = x.reshape(N, 3)
        pts = _project_to_sphere(pts)
        _, g = _energy_and_grad(pts)
        return g.flatten()
    
    try:
        res = minimize(objective, x0, jac=jac, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'maxfun': maxiter * 10,
                                'ftol': 1e-14, 'gtol': 1e-10})
        result = _project_to_sphere(res.x.reshape(N, 3))
        return result
    except Exception:
        return None

def construct():
    """
    Main entry point.
    Strategy:
    1. Generate multiple diverse initial configurations.
    2. Run CG optimization on each within a time budget.
    3. Use simulated annealing to escape local minima.
    4. Final polish with tight tolerance.
    5. Return the best configuration found.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # List of initial configurations
    inits = []
    
    # 1. Fibonacci spiral
    inits.append(_initial_fibonacci())
    
    # 2. Random points
    rng = np.random.default_rng(42)
    for i in range(8):
        pts = rng.normal(size=(N, 3))
        inits.append(_project_to_sphere(pts))
        
    # 3. Jittered Fibonacci
    for i in range(8):
        pts = _initial_fibonacci() + rng.normal(size=(N, 3)) * 0.1
        inits.append(_project_to_sphere(pts))
    
    best_pts = inits[0].copy()
    best_energy = _energy_and_grad(best_pts)[0]
    
    # Phase 1: Optimize each initial configuration with CG
    for i, init_pts in enumerate(inits):
        if time.time() > deadline - 30:
            break
            
        time_left = deadline - time.time()
        run_deadline = time.time() + min(8.0, time_left - 5.0)
        
        if run_deadline <= time.time():
            continue
            
        optimized_pts = _conjugate_gradient(init_pts, max_iter=1000, deadline=run_deadline)
        
        e_opt, _ = _energy_and_grad(optimized_pts)
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = optimized_pts.copy()
            
    # Phase 2: Simulated annealing from best point
    if time.time() < deadline - 20:
        sa_deadline = time.time() + min(20.0, deadline - time.time() - 5)
        sa_pts = _simulated_annealing(best_pts, sa_deadline, n_steps=15000)
        
        e_sa, _ = _energy_and_grad(sa_pts)
        if e_sa < best_energy:
            best_energy = e_sa
            best_pts = sa_pts.copy()
            
        # Polish after SA
        if time.time() < deadline - 10:
            polish_deadline = time.time() + min(10.0, deadline - time.time() - 2)
            polished = _conjugate_gradient(best_pts, max_iter=2000, tol=1e-12, deadline=polish_deadline)
            e_pol, _ = _energy_and_grad(polished)
            if e_pol < best_energy:
                best_energy = e_pol
                best_pts = polished.copy()
            
    # Phase 3: L-BFGS polish from best point
    if time.time() < deadline - 15:
        lbfgs_deadline = time.time() + min(15.0, deadline - time.time() - 3)
        lbfgs_pts = _optimize_lbfgs(best_pts, lbfgs_deadline, maxiter=3000)
        
        if lbfgs_pts is not None:
            e_lbfgs, _ = _energy_and_grad(lbfgs_pts)
            if e_lbfgs < best_energy:
                best_energy = e_lbfgs
                best_pts = lbfgs_pts.copy()
                
    # Phase 4: Final high-precision CG polish
    if time.time() < deadline - 5:
        final_deadline = deadline - 1
        final_pts = _conjugate_gradient(best_pts, max_iter=5000, tol=1e-13, deadline=final_deadline)
        e_final, _ = _energy_and_grad(final_pts)
        if e_final < best_energy:
            best_energy = e_final
            best_pts = final_pts.copy()
    
    # Final validation
    best_pts = _project_to_sphere(best_pts)
    
    # Check for close points
    diff = best_pts[:, None, :] - best_pts[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(N, k=1)
    min_dist = np.min(dist[iu])
    
    if min_dist < 1e-9:
        # Fallback to Fibonacci if invalid
        best_pts = _initial_fibonacci()
    
    return best_pts
# EVOLVE-BLOCK-END

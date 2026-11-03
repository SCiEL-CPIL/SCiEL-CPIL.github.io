"""Logarithmic energy minimization of 150 points on the unit sphere."""
import numpy as np
import time
import math

N = 150
DEADLINE = 145.0  # seconds, safety margin before 150s kill


def _energy_and_grad(pts):
    """Compute log(1/|x_i - x_j|) energy sum over i<j and gradient.
    
    pts: (N, 3) array on unit sphere.
    Returns: (energy, grad) where grad is (N, 3).
    """
    n = pts.shape[0]
    # Pairwise squared distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (n, n)
    # Mask diagonal
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d = np.sqrt(dist_sq[mask])  # (n*(n-1)/2,)
    energy = -np.sum(np.log(d))
    
    # Gradient: dE/dx_i = -sum_{j != i} (x_i - x_j)/|x_i - x_j|^2
    # = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    inv_d2 = np.zeros_like(dist_sq)
    np.fill_diagonal(inv_d2, 0.0)
    inv_d2[mask] = 1.0 / dist_sq[mask]
    
    # grad[i] = sum_j inv_d2[i,j] * (pts[j] - pts[i])
    row_sum = inv_d2.sum(axis=1)  # (n,)
    grad = inv_d2 @ pts - pts * row_sum[:, np.newaxis]
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _tangent_project(pts, vec):
    """Project vector to tangent space of sphere at pts."""
    dot = np.sum(vec * pts, axis=1, keepdims=True)
    return vec - dot * pts


def _fibonacci_sphere(n):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=None):
    """Generate n random points on unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _velocity_verlet_step(pts, vel, dt, gamma):
    """One velocity-Verlet step with damping for rigid body on sphere.
    
    Uses the fact that on a sphere, the acceleration has both a tangential
    component (from the force) and a normal component (centripetal, which
    is handled by projection). We integrate the tangential dynamics.
    """
    n = pts.shape[0]
    
    # Compute forces (negative gradient of energy)
    _, grad = _energy_and_grad(pts)
    # Force = -grad (we want to minimize energy)
    force = -grad
    
    # Project force to tangent space: f_tan = f - (f·x)x
    f_tan = _tangent_project(pts, force)
    
    # Velocity Verlet:
    # x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
    # v(t+dt/2) = v(t) + 0.5*a(t)*dt
    # Then recompute a at new position, v(t+dt) = v(t+dt/2) + 0.5*a(t+dt)*dt
    
    # Half-step velocity
    v_half = vel + 0.5 * f_tan * dt
    
    # Full-step position
    pts_new = pts + v_half * dt
    
    # Project to sphere
    pts_new = _project_to_sphere(pts_new)
    
    # Recompute force at new position
    _, grad_new = _energy_and_grad(pts_new)
    force_new = -grad_new
    f_tan_new = _tangent_project(pts_new, force_new)
    
    # Full-step velocity with damping
    vel_new = v_half + 0.5 * f_tan_new * dt
    vel_new *= gamma  # damping
    
    # Project velocity to tangent space
    vel_new = _tangent_project(pts_new, vel_new)
    
    return pts_new, vel_new


def _refine_lbfgs(pts, max_iter=500, tol=1e-12):
    """Use L-BFGS-B style optimization with projection to refine points."""
    from scipy.optimize import minimize
    
    def energy_fn(x):
        pts = _project_to_sphere(x.reshape(-1, 3))
        e, _ = _energy_and_grad(pts)
        return e
    
    def grad_fn(x):
        pts = _project_to_sphere(x.reshape(-1, 3))
        _, g = _energy_and_grad(pts)
        # Project gradient to tangent space for proper constrained optimization
        g_tan = _tangent_project(pts, g)
        return g_tan.flatten()
    
    x0 = pts.flatten()
    result = minimize(energy_fn, x0, jac=grad_fn, method='L-BFGS-B',
                      options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol})
    pts_opt = _project_to_sphere(result.x.reshape(-1, 3))
    return pts_opt


def construct():
    t_start = time.time()
    
    # Multiple diverse starting points
    starts = [
        _fibonacci_sphere(N),
        _random_sphere(N, seed=42),
        _random_sphere(N, seed=123),
        _random_sphere(N, seed=777),
    ]
    
    e_best = float('inf')
    pts_best = None
    
    # Phase 1: For each start, do velocity-Verlet annealing + L-BFGS
    for si, start_pts in enumerate(starts):
        if time.time() - t_start > DEADLINE - 40:
            break
        
        pts = start_pts.copy()
        vel = np.zeros_like(pts)
        
        # Velocity-Verlet annealing schedule
        schedule = [
            (150, 0.04, 0.99),
            (150, 0.02, 0.995),
            (200, 0.01, 0.998),
            (300, 0.005, 0.999),
            (300, 0.002, 0.9995),
        ]
        
        e_local = float('inf')
        pts_local = pts.copy()
        
        for n_steps, dt_val, gamma_val in schedule:
            for step in range(n_steps):
                if time.time() - t_start > DEADLINE - 40:
                    break
                pts, vel = _velocity_verlet_step(pts, vel, dt_val, gamma_val)
                
                # Track best locally
                e_curr, _ = _energy_and_grad(pts)
                if e_curr < e_local:
                    e_local = e_curr
                    pts_local = pts.copy()
            
            if time.time() - t_start > DEADLINE - 40:
                break
        
        # L-BFGS refinement
        if time.time() - t_start < DEADLINE - 20:
            pts_refined = _refine_lbfgs(pts_local, max_iter=300, tol=1e-10)
            e_ref, _ = _energy_and_grad(pts_refined)
            if e_ref < e_local:
                e_local = e_ref
                pts_local = pts_refined
        
        # Update global best
        if e_local < e_best:
            e_best = e_local
            pts_best = pts_local.copy()
    
    # Phase 2: Basin hopping from best point
    if time.time() - t_start < DEADLINE - 15:
        pts_bh = pts_best.copy()
        e_bh = e_best
        rng = np.random.RandomState(999)
        
        for bh_iter in range(30):
            if time.time() - t_start > DEADLINE - 15:
                break
            
            # Perturb
            perturbation = 0.05 * rng.randn(N, 3)
            pts_pert = _project_to_sphere(pts_bh + perturbation)
            
            # Local minimize
            pts_opt = _refine_lbfgs(pts_pert, max_iter=200, tol=1e-10)
            e_opt, _ = _energy_and_grad(pts_opt)
            
            # Accept if better or with probability
            if e_opt < e_bh or rng.random() < math.exp(-(e_opt - e_bh) / 0.1):
                pts_bh = pts_opt
                e_bh = e_opt
            
            if e_bh < e_best:
                e_best = e_bh
                pts_best = pts_bh.copy()
    
    # Phase 3: Final high-precision polish
    if time.time() - t_start < DEADLINE - 5:
        pts_polish = _refine_lbfgs(pts_best, max_iter=500, tol=1e-14)
        e_pol, _ = _energy_and_grad(pts_polish)
        if e_pol < e_best:
            e_best = e_pol
            pts_best = pts_polish
    
    # Ensure valid output
    pts_best = _project_to_sphere(pts_best)
    
    # Check for near-duplicate points
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(pts_best[i] - pts_best[j])
            if d < 1e-9:
                # Nudge point j slightly
                pts_best[j] += 1e-8 * np.random.randn(3)
                pts_best[j] = _project_to_sphere(pts_best[j].reshape(1, 3))[0]
    
    return pts_best

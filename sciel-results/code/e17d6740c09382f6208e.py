"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _fibonacci_sphere(n, offset=0.0):
    """Generate n points on the unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _log_energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = len(pts)
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    # Mask diagonal and lower triangle
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d = dist[mask]
    # Avoid log(0) by clipping
    d = np.maximum(d, 1e-15)
    terms = np.log(1.0 / d)
    return math.fsum(terms.tolist())

def _log_energy_grad(pts):
    """Compute gradient of log energy w.r.t. points (in R^3, not on manifold).
    E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|x_i - x_j|
    dE/dx_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(pts)
    grad = np.zeros_like(pts)
    # Vectorized: for each i, sum over j != i of (x_i - x_j)/|x_i - x_j|^2
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # diff[i,j] = x_i - x_j
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)  # avoid div by 0
    # term[i,j] = (x_i - x_j) / |x_i - x_j|^2
    # grad[i] = sum_j term[i,j]
    inv_dist2 = 1.0 / dist2
    # term = diff * inv_dist2[..., np.newaxis]
    term = diff * inv_dist2[:, :, np.newaxis]
    grad = np.sum(term, axis=1)
    return grad

def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _velocity_verlet_step(pts, velocities, dt, beta):
    """One velocity-Verlet step for dynamics on sphere with damping."""
    # Compute acceleration: a = -grad_E_tangent - beta * v
    grad = _log_energy_grad(pts)
    grad_dot_x = np.sum(grad * pts, axis=1, keepdims=True)
    grad_tan = grad - grad_dot_x * pts
    acc = -grad_tan - beta * velocities
    
    # Velocity Verlet
    v_half = velocities + 0.5 * dt * acc
    pts_new = pts + dt * v_half
    pts_new = _project_to_sphere(pts_new)
    
    # Recompute acceleration at new positions
    grad_new = _log_energy_grad(pts_new)
    grad_dot_x_new = np.sum(grad_new * pts_new, axis=1, keepdims=True)
    grad_tan_new = grad_new - grad_dot_x_new * pts_new
    acc_new = -grad_tan_new - beta * v_half
    
    v_new = v_half + 0.5 * dt * acc_new
    # Project velocity to tangent space
    v_dot_x = np.sum(v_new * pts_new, axis=1, keepdims=True)
    v_new = v_new - v_dot_x * pts_new
    
    return pts_new, v_new

def _polish_lbfgs(pts, deadline):
    """Polish points using L-BFGS-B with tight tolerances."""
    try:
        from scipy.optimize import minimize
        
        def _energy_and_grad(x_flat):
            x = x_flat.reshape(N, 3)
            x = _project_to_sphere(x)
            e = _log_energy(x)
            g = _log_energy_grad(x)
            g_dot_x = np.sum(g * x, axis=1, keepdims=True)
            g_tan = g - g_dot_x * x
            return e, g_tan.flatten()
        
        x0 = pts.flatten()
        res = minimize(_energy_and_grad, x0, method='L-BFGS-B',
                      jac=True, options={'maxiter': 500, 'gtol': 1e-10, 'factr': 1e10})
        if res.success and np.all(np.isfinite(res.x)):
            pts_polished = _project_to_sphere(res.x.reshape(N, 3))
            e_polished = _log_energy(pts_polished)
            if e_polished < _log_energy(pts):
                return pts_polished
    except Exception:
        pass
    return pts

def _final_gradient_descent(pts, deadline):
    """Final high-precision gradient descent on the sphere."""
    step_size = 1e-4
    min_step = 1e-10
    patience = 200
    steps_since_improve = 0
    best_e = _log_energy(pts)
    best_pts = pts.copy()
    
    for step in range(50000):
        if time.time() > deadline:
            break
        
        grad = _log_energy_grad(pts)
        grad_dot_x = np.sum(grad * pts, axis=1, keepdims=True)
        grad_tan = grad - grad_dot_x * pts
        
        # Adaptive step size
        pts_new = pts - step_size * grad_tan
        pts_new = _project_to_sphere(pts_new)
        
        e_new = _log_energy(pts_new)
        if e_new < best_e - 1e-14:
            best_e = e_new
            best_pts = pts_new.copy()
            pts = pts_new
            steps_since_improve = 0
            step_size = min(step_size * 1.02, 1e-2)
        else:
            steps_since_improve += 1
            step_size = max(step_size * 0.9, min_step)
            if steps_since_improve > patience:
                break
    
    return best_pts

def construct():
    """Construct 150 points on unit sphere minimizing log energy."""
    deadline = time.time() + 130.0  # Safety margin before 150s limit
    
    # Start with Fibonacci sphere
    pts = _fibonacci_sphere(N)
    best_pts = pts.copy()
    best_energy = _log_energy(best_pts)
    
    # Multi-start with different offsets
    offsets = [0.0, 0.25, 0.5, 0.75, 0.1, 0.3, 0.6, 0.9, 0.15, 0.45, 0.65, 0.85]
    
    for offset in offsets:
        if time.time() > deadline:
            break
        
        pts = _fibonacci_sphere(N, offset)
        
        # Velocity-Verlet dynamics with adaptive timestep
        velocities = np.zeros_like(pts)
        dt = 0.0005
        beta = 0.5  # Higher damping for stability
        min_dt = 1e-7
        max_dt = 0.005
        max_steps = 30000
        patience = 2000
        
        energy = _log_energy(pts)
        best_local_energy = energy
        steps_since_improve = 0
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            pts, velocities = _velocity_verlet_step(pts, velocities, dt, beta)
            
            # Check for validity
            if not np.all(np.isfinite(pts)):
                break
            
            # Recompute energy periodically
            if step % 5 == 0:
                energy = _log_energy(pts)
                if energy < best_local_energy - 1e-12:
                    best_local_energy = energy
                    steps_since_improve = 0
                    dt = min(dt * 1.03, max_dt)
                else:
                    steps_since_improve += 1
                    if steps_since_improve > patience:
                        dt = max(dt * 0.5, min_dt)
                        beta *= 1.2
                        steps_since_improve = 0
            
            # If energy is blowing up, reduce dt and rollback
            if energy > best_local_energy + 0.5:
                dt = max(dt * 0.1, min_dt)
                beta *= 1.5
                pts = _project_to_sphere(pts)
        
        # Polish with L-BFGS-B
        if time.time() < deadline - 5:
            pts = _polish_lbfgs(pts, deadline)
            e_polished = _log_energy(pts)
            if e_polished < best_local_energy:
                best_local_energy = e_polished
        
        # Update best
        if best_local_energy < best_energy:
            best_energy = best_local_energy
            best_pts = pts.copy()
    
    # Final high-precision gradient descent polish
    if time.time() < deadline - 2:
        best_pts = _final_gradient_descent(best_pts, deadline)
        final_e = _log_energy(best_pts)
        if final_e < best_energy:
            best_energy = final_e
    
    # Final check and normalization
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points are too close
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    
    if min_dist < 1e-9:
        # Fallback to original Fibonacci
        best_pts = _fibonacci_sphere(N)
    
    return best_pts

# EVOLVE-BLOCK-END

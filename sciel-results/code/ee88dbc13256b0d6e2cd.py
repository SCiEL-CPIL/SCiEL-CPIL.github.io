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
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(pts)
    # Compute all pairwise distances
    # Use broadcasting for efficiency
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    # Mask diagonal and lower triangle
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d = dist[mask]
    # Avoid log(0) by clipping
    d = np.maximum(d, 1e-15)
    return float(np.sum(np.log(1.0 / d)))

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
    """One velocity-Verlet step for the rigid body dynamics on sphere.
    
    Force on point i: F_i = -dE/dx_i projected to tangent space
    Since we want to minimize E, the force is -grad(E)
    But we also add damping: F_damped = -grad(E) - beta * v_i
    
    Actually, for energy minimization via dynamics:
    m * a_i = F_i where F_i is the negative gradient projected to tangent
    With damping: m * a_i = -grad_E_i^tangent - beta * v_i
    
    Velocity-Verlet:
    1. v_half = v + 0.5 * dt * a
    2. x_new = x + dt * v_half
    3. Project x_new to sphere
    4. Compute a_new at x_new
    5. v_new = v_half + 0.5 * dt * a_new
    6. Project v_new to tangent space at x_new
    """
    n = len(pts)
    # Compute acceleration: a = -grad_E_tangent - beta * v
    grad = _log_energy_grad(pts)
    # Project gradient to tangent space: grad_tan = grad - (grad . x) * x
    grad_dot_x = np.sum(grad * pts, axis=1, keepdims=True)
    grad_tan = grad - grad_dot_x * pts
    # Acceleration = -grad_tan - beta * v
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

def construct():
    """Construct 150 points on unit sphere minimizing log energy."""
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    
    # Start with Fibonacci sphere
    pts = _fibonacci_sphere(N)
    best_pts = pts.copy()
    best_energy = _log_energy(best_pts)
    
    # Multi-start with different offsets
    offsets = [0.0, 0.25, 0.5, 0.75, 0.1, 0.3, 0.6, 0.9]
    
    for offset in offsets:
        if time.time() > deadline:
            break
        
        pts = _fibonacci_sphere(N, offset)
        
        # Velocity-Verlet dynamics with adaptive timestep
        velocities = np.zeros_like(pts)
        dt = 0.001
        beta = 0.1  # Damping coefficient
        min_dt = 1e-6
        max_dt = 0.01
        max_steps = 50000
        patience = 5000  # Steps without improvement before reducing dt
        
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
            
            # Recompute energy periodically to save time
            if step % 10 == 0:
                energy = _log_energy(pts)
                if energy < best_local_energy - 1e-12:
                    best_local_energy = energy
                    steps_since_improve = 0
                    # Increase dt slightly if improving
                    dt = min(dt * 1.05, max_dt)
                else:
                    steps_since_improve += 1
                    if steps_since_improve > patience:
                        # Reduce dt and increase damping
                        dt = max(dt * 0.5, min_dt)
                        beta *= 1.1
                        steps_since_improve = 0
            
            # If energy is blowing up, reduce dt
            if energy > best_local_energy + 1.0:
                dt = max(dt * 0.1, min_dt)
                beta *= 1.5
                # Roll back
                pts = _project_to_sphere(pts)
        
        # Final polish with L-BFGS-B on the sphere using tangent coordinates
        # Use scipy if available for faster convergence
        try:
            from scipy.optimize import minimize
            from scipy.spatial.distance import cdist
            
            def _energy_and_grad(x_flat):
                """Energy and gradient for flat vector representation."""
                x = x_flat.reshape(N, 3)
                x = _project_to_sphere(x)
                e = _log_energy(x)
                g = _log_energy_grad(x)
                # Project gradient to tangent
                g_dot_x = np.sum(g * x, axis=1, keepdims=True)
                g_tan = g - g_dot_x * x
                return e, g_tan.flatten()
            
            # Run L-BFGS-B from current position
            x0 = pts.flatten()
            res = minimize(_energy_and_grad, x0, method='L-BFGS-B', 
                          jac=True, options={'maxiter': 200, 'gtol': 1e-8})
            if res.success and np.all(np.isfinite(res.x)):
                pts_polished = _project_to_sphere(res.x.reshape(N, 3))
                e_polished = _log_energy(pts_polished)
                if e_polished < best_local_energy:
                    best_local_energy = e_polished
                    pts = pts_polished
        except Exception:
            pass
        
        # Update best
        if best_local_energy < best_energy:
            best_energy = best_local_energy
            best_pts = pts.copy()
    
    # Final check and normalization
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points are too close
    min_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < min_dist:
                min_dist = d
    
    if min_dist < 1e-9:
        # Fallback to original Fibonacci
        best_pts = _fibonacci_sphere(N)
    
    return best_pts

# EVOLVE-BLOCK-END

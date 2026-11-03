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
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d = dist[mask]
    d = np.maximum(d, 1e-15)
    # Use math.fsum for accuracy as required
    return float(math.fsum(math.log(1.0 / di) for di in d))

def _log_energy_grad(pts):
    """Compute gradient of log energy w.r.t. points (in R^3).
    E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|x_i - x_j|
    dE/dx_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # diff[i,j] = x_i - x_j
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)  # avoid div by 0
    inv_dist2 = 1.0 / dist2
    term = diff * inv_dist2[:, :, np.newaxis]
    grad = np.sum(term, axis=1)
    return grad

def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _tangent_project(grad, pts):
    """Project gradient to tangent space at each point."""
    g_dot_x = np.sum(grad * pts, axis=1, keepdims=True)
    return grad - g_dot_x * pts

def _velocity_verlet_step(pts, velocities, dt, beta):
    """One velocity-Verlet step for the rigid body dynamics on sphere."""
    n = len(pts)
    grad = _log_energy_grad(pts)
    grad_tan = _tangent_project(grad, pts)
    acc = -grad_tan - beta * velocities
    
    v_half = velocities + 0.5 * dt * acc
    pts_new = pts + dt * v_half
    pts_new = _project_to_sphere(pts_new)
    
    grad_new = _log_energy_grad(pts_new)
    grad_tan_new = _tangent_project(grad_new, pts_new)
    acc_new = -grad_tan_new - beta * v_half
    
    v_new = v_half + 0.5 * dt * acc_new
    v_dot_x = np.sum(v_new * pts_new, axis=1, keepdims=True)
    v_new = v_new - v_dot_x * pts_new
    
    return pts_new, v_new

def _polish_lbfgs(pts, deadline):
    """Polish using L-BFGS-B with tangent space gradient."""
    try:
        from scipy.optimize import minimize
        
        def _energy_and_grad(x_flat):
            x = x_flat.reshape(N, 3)
            x = _project_to_sphere(x)
            e = _log_energy(x)
            g = _log_energy_grad(x)
            g_tan = _tangent_project(g, x)
            return e, g_tan.flatten()
        
        x0 = pts.flatten()
        res = minimize(_energy_and_grad, x0, method='L-BFGS-B', 
                      jac=True, options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-12})
        if res.success and np.all(np.isfinite(res.x)):
            pts_polished = _project_to_sphere(res.x.reshape(N, 3))
            e_polished = _log_energy(pts_polished)
            return pts_polished, e_polished
    except Exception:
        pass
    
    # Fallback: simple gradient descent with adaptive step
    pts_curr = pts.copy()
    e_curr = _log_energy(pts_curr)
    lr = 0.001
    for _ in range(2000):
        if time.time() > deadline:
            break
        g = _log_energy_grad(pts_curr)
        g_tan = _tangent_project(g, pts_curr)
        pts_next = pts_curr - lr * g_tan
        pts_next = _project_to_sphere(pts_next)
        e_next = _log_energy(pts_next)
        if e_next < e_curr:
            pts_curr = pts_next
            e_curr = e_next
            lr = min(lr * 1.1, 0.01)
        else:
            lr *= 0.5
            if lr < 1e-10:
                break
    return pts_curr, e_curr

def _greedy_farthest_point(n, rng):
    """Greedy farthest-point sampling for diverse initial configurations."""
    pts = np.zeros((n, 3))
    # Start with a random point
    pts[0] = rng.normal(size=3)
    pts[0] = pts[0] / np.linalg.norm(pts[0])
    
    for i in range(1, n):
        # Compute distances from new candidate to all existing points
        # We'll pick the point that maximizes the minimum distance
        best_dist = -1
        best_pt = None
        # Try a few random candidates and pick the best
        for _ in range(50):
            cand = rng.normal(size=3)
            cand = cand / np.linalg.norm(cand)
            d = np.min(np.linalg.norm(pts[:i] - cand, axis=1))
            if d > best_dist:
                best_dist = d
                best_pt = cand
        pts[i] = best_pt
    return pts

def construct():
    """Construct 150 points on unit sphere minimizing log energy."""
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    rng = np.random.default_rng(42)
    
    # Generate diverse initial configurations
    offsets = [0.0, 0.25, 0.5, 0.75, 0.1, 0.3, 0.6, 0.9, 0.15, 0.45, 0.55, 0.85]
    
    best_pts = _fibonacci_sphere(N)
    best_energy = _log_energy(best_pts)
    
    for offset in offsets:
        if time.time() > deadline:
            break
        
        # Mix of Fibonacci and greedy starts
        if offset < 0.5:
            pts = _fibonacci_sphere(N, offset)
        else:
            pts = _greedy_farthest_point(N, rng)
            # Add small perturbation to break symmetry
            pts = pts + 0.01 * rng.normal(size=(N, 3))
            pts = _project_to_sphere(pts)
        
        # Velocity-Verlet dynamics with adaptive timestep
        velocities = np.zeros_like(pts)
        dt = 0.002
        beta = 0.05  # Damping coefficient
        min_dt = 1e-7
        max_dt = 0.02
        max_steps = 30000
        patience = 3000
        
        energy = _log_energy(pts)
        best_local_energy = energy
        steps_since_improve = 0
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            pts, velocities = _velocity_verlet_step(pts, velocities, dt, beta)
            
            if not np.all(np.isfinite(pts)):
                break
            
            # Recompute energy periodically
            if step % 20 == 0:
                energy = _log_energy(pts)
                if energy < best_local_energy - 1e-12:
                    best_local_energy = energy
                    steps_since_improve = 0
                    dt = min(dt * 1.08, max_dt)
                else:
                    steps_since_improve += 1
                    if steps_since_improve > patience:
                        dt = max(dt * 0.5, min_dt)
                        beta *= 1.2
                        steps_since_improve = 0
            
            # If energy is blowing up significantly, reduce dt and roll back
            if energy > best_local_energy + 2.0:
                dt = max(dt * 0.1, min_dt)
                beta *= 1.5
                pts = _project_to_sphere(pts)
        
        # Final polish with L-BFGS
        pts_polished, e_polished = _polish_lbfgs(pts, deadline)
        
        if e_polished < best_local_energy:
            best_local_energy = e_polished
            pts = pts_polished
        
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
        best_pts = _fibonacci_sphere(N)
    
    return best_pts

# EVOLVE-BLOCK-END

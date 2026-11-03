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
    dist2 = np.sum(diff * diff, axis=2)
    # Get upper triangle indices
    iu = np.triu_indices(n, k=1)
    d2 = dist2[iu]
    # Avoid log(0) by clipping
    d2 = np.maximum(d2, 1e-30)
    d = np.sqrt(d2)
    log_terms = np.log(1.0 / d)
    # Use math.fsum for accurate summation as required by evaluator
    return math.fsum(log_terms.tolist())

def _log_energy_grad(pts):
    """Compute gradient of log energy w.r.t. points (in R^3, not on manifold).
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

def _velocity_verlet_step(pts, velocities, dt, beta):
    """One velocity-Verlet step for dynamics on sphere with damping."""
    grad = _log_energy_grad(pts)
    grad_dot_x = np.sum(grad * pts, axis=1, keepdims=True)
    grad_tan = grad - grad_dot_x * pts
    acc = -grad_tan - beta * velocities
    
    v_half = velocities + 0.5 * dt * acc
    pts_new = pts + dt * v_half
    pts_new = _project_to_sphere(pts_new)
    
    grad_new = _log_energy_grad(pts_new)
    grad_dot_x_new = np.sum(grad_new * pts_new, axis=1, keepdims=True)
    grad_tan_new = grad_new - grad_dot_x_new * pts_new
    acc_new = -grad_tan_new - beta * v_half
    
    v_new = v_half + 0.5 * dt * acc_new
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
        # Increase maxiter and tighten tolerances for better convergence
        res = minimize(_energy_and_grad, x0, method='L-BFGS-B', 
                      jac=True, options={'maxiter': 2000, 'gtol': 1e-12, 'maxls': 100})
        if np.all(np.isfinite(res.x)):
            pts_polished = _project_to_sphere(res.x.reshape(N, 3))
            return pts_polished
    except Exception:
        pass
    return pts

def _final_gradient_descent(pts, deadline):
    """Final high-precision gradient descent to squeeze out residual energy."""
    if time.time() > deadline - 1.0:
        return pts
    
    lr = 1e-8
    for _ in range(500):
        if time.time() > deadline - 0.5:
            break
        grad = _log_energy_grad(pts)
        g_dot_x = np.sum(grad * pts, axis=1, keepdims=True)
        g_tan = grad - g_dot_x * pts
        pts = pts - lr * g_tan
        pts = _project_to_sphere(pts)
    return pts

def construct():
    """Construct 150 points on unit sphere minimizing log energy."""
    deadline = time.time() + 140.0
    
    # Multi-start configurations: Fibonacci with various offsets + random perturbations
    initial_configs = []
    
    # Fibonacci-based starts
    for offset in [0.0, 0.25, 0.5, 0.75, 0.1, 0.3, 0.6, 0.9]:
        initial_configs.append(_fibonacci_sphere(N, offset))
    
    # Add some random perturbed starts for diversity
    rng = np.random.default_rng(42)
    for _ in range(4):
        base = _fibonacci_sphere(N, rng.random() * 0.5)
        # Small random perturbation
        noise = rng.normal(0, 0.01, size=base.shape)
        perturbed = base + noise
        initial_configs.append(_project_to_sphere(perturbed))
    
    best_pts = initial_configs[0].copy()
    best_energy = _log_energy(best_pts)
    
    for config_idx, init_pts in enumerate(initial_configs):
        if time.time() > deadline - 20.0:  # Reserve time for final polish
            break
        
        pts = init_pts.copy()
        
        # Phase 1: Velocity-Verlet dynamics with adaptive timestep
        velocities = np.zeros_like(pts)
        dt = 0.002
        beta = 0.2
        min_dt = 1e-7
        max_dt = 0.005
        max_steps = 30000
        patience = 3000
        
        energy = _log_energy(pts)
        best_local_energy = energy
        steps_since_improve = 0
        energy_history = []
        
        for step in range(max_steps):
            if time.time() > deadline - 20.0:
                break
            
            pts, velocities = _velocity_verlet_step(pts, velocities, dt, beta)
            
            if not np.all(np.isfinite(pts)):
                break
            
            # Recompute energy every 5 steps
            if step % 5 == 0:
                energy = _log_energy(pts)
                energy_history.append(energy)
                
                if energy < best_local_energy - 1e-14:
                    best_local_energy = energy
                    steps_since_improve = 0
                    # Gradually increase dt when improving
                    if len(energy_history) > 10:
                        recent_improvement = energy_history[-1] - energy_history[-10]
                        if recent_improvement < 0:  # Still improving
                            dt = min(dt * 1.02, max_dt)
                else:
                    steps_since_improve += 1
                    if steps_since_improve > patience:
                        dt = max(dt * 0.7, min_dt)
                        beta *= 1.05
                        steps_since_improve = 0
                        # If stuck, try smaller dt
                        if dt <= min_dt * 2:
                            break
            
            # If energy blows up, rollback and reduce dt
            if energy > best_local_energy + 0.5:
                dt = max(dt * 0.5, min_dt)
                beta *= 1.2
                velocities *= 0.5
        
        # Phase 2: Final polish with L-BFGS-B
        pts = _polish_lbfgs(pts, deadline)
        local_energy = _log_energy(pts)
        
        if local_energy < best_energy:
            best_energy = local_energy
            best_pts = pts.copy()
    
    # Final high-precision polish on the best configuration
    if time.time() < deadline - 5.0:
        best_pts = _polish_lbfgs(best_pts, deadline)
        final_energy = _log_energy(best_pts)
        if final_energy < best_energy:
            best_energy = final_energy
        
        # Additional micro-polish with gradient descent
        if time.time() < deadline - 1.0:
            best_pts = _final_gradient_descent(best_pts, deadline)
            final_energy = _log_energy(best_pts)
            if final_energy < best_energy:
                best_energy = final_energy

    # Ensure validity
    best_pts = _project_to_sphere(best_pts)
    
    # Verify minimum distance
    min_dist = np.inf
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    
    if min_dist < 1e-9:
        best_pts = _fibonacci_sphere(N)
    
    return best_pts

# EVOLVE-BLOCK-END

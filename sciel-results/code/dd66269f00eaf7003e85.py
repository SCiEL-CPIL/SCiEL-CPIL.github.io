import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150

def _energy_and_grad_flat(flat):
    """Energy and gradient for L-BFGS-B on flattened coordinates."""
    x = flat.reshape(N, 3)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    mask = dist_sq > 1e-12
    inv_dist_sq = np.zeros_like(dist_sq)
    inv_dist_sq[mask] = 1.0 / dist_sq[mask]
    grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    log_dist_sq = np.where(mask, np.log(dist_sq), 0.0)
    total_log_dist_sq = np.sum(log_dist_sq)
    energy = -0.25 * total_log_dist_sq
    return energy, grad.flatten()

def _project_to_sphere(x):
    norms = np.linalg.norm(x, axis=1)
    norms[norms < 1e-12] = 1e-12
    return x / norms[:, np.newaxis]

def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _geodesic_dome_init(n):
    """Generate a geodesic dome (icosahedron subdivision) initialization."""
    # Start with icosahedron vertices
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Faces of icosahedron
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    # Subdivide until we have at least n points
    points = verts.copy()
    face_list = [list(f) for f in faces]
    
    while len(points) < n:
        new_faces = []
        new_points = []
        midpoint_cache = {}
        
        for f in face_list:
            midpoints = []
            for i in range(3):
                a, b = f[i], f[(i+1) % 3]
                key = (min(a, b), max(a, b))
                if key not in midpoint_cache:
                    mid = (points[a] + points[b]) / 2.0
                    mid = mid / np.linalg.norm(mid)
                    midpoint_cache[key] = len(points) + len(new_points)
                    new_points.append(mid)
                midpoints.append(midpoint_cache[key])
            
            new_faces.append([f[0], midpoints[0], midpoints[1]])
            new_faces.append([f[1], midpoints[1], midpoints[2]])
            new_faces.append([f[2], midpoints[2], midpoints[0]])
            new_faces.append([midpoints[0], midpoints[1], midpoints[2]])
        
        points = np.vstack([points, np.array(new_points)]) if new_points else points
        face_list = new_faces
    
    # Select n points from the geodesic dome
    indices = np.linspace(0, len(points)-1, n, dtype=int)
    return points[indices]

def _adam_optimize(x, start_time, time_limit, lr=0.01):
    """Adam optimizer on the sphere with tangent space projection."""
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    t = 0
    best_x = x.copy()
    best_energy, _ = _energy_and_grad_flat(x.flatten())
    
    while time.time() - start_time < time_limit:
        t += 1
        energy, grad = _energy_and_grad_flat(x.flatten())
        x_curr = x
        # Project gradient to tangent space
        grad_tan = grad - np.sum(grad * x_curr, axis=1, keepdims=True) * x_curr
        
        m = beta1 * m + (1 - beta1) * grad_tan
        v = beta2 * v + (1 - beta2) * (grad_tan ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        x_new = x_curr - lr * m_hat / (np.sqrt(v_hat) + eps)
        x_new = _project_to_sphere(x_new)
        
        new_energy, _ = _energy_and_grad_flat(x_new.flatten())
        if new_energy < best_energy:
            best_energy = new_energy
            best_x = x_new.copy()
        x = x_new
    
    return best_x, best_energy

def _lbfgs_polish(x, time_budget, tight=False):
    """Polish with L-BFGS-B using tangent space projection."""
    start = time.time()
    deadline = start + time_budget
    
    def obj_and_grad(flat):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        x = flat.reshape(N, 3)
        x = _project_to_sphere(x)
        e, g = _energy_and_grad_flat(flat)
        g_tan = g - np.sum(g * x, axis=1, keepdims=True) * x
        return e, g_tan.flatten()
    
    maxiter = 500 if tight else 200
    ftol = 1e-15 if tight else 1e-12
    gtol = 1e-12 if tight else 1e-10
    
    try:
        res = minimize(obj_and_grad, x.flatten(), jac=True, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxls': 50})
        if res.fun < float('inf'):
            x_new = _project_to_sphere(res.x.reshape(N, 3))
            return x_new
    except Exception:
        pass
    return x

def _momentum_optimize(x, start_time, time_limit):
    """Momentum-based optimization with adaptive step size."""
    momentum = 0.9
    v = np.zeros_like(x)
    step_size = 0.05
    min_step = 1e-8
    max_step = 0.2
    
    best_x = x.copy()
    best_energy, _ = _energy_and_grad_flat(x.flatten())
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - start_time < time_limit:
        energy, grad = _energy_and_grad_flat(x.flatten())
        x_curr = x
        grad_tan = grad - np.sum(grad * x_curr, axis=1, keepdims=True) * x_curr
        
        v = momentum * v - step_size * grad_tan
        x_new = x_curr + v
        x_new = _project_to_sphere(x_new)
        
        new_energy, _ = _energy_and_grad_flat(x_new.flatten())
        
        if new_energy < best_energy - 1e-12:
            x = x_new
            best_energy = new_energy
            best_x = x.copy()
            no_improve_count = 0
            step_size = min(max_step, step_size * 1.02)
        else:
            no_improve_count += 1
            step_size = max(min_step, step_size * 0.85)
            if no_improve_count >= max_no_improve:
                v = np.zeros_like(x)
                no_improve_count = 0
                if step_size <= min_step * 1.1:
                    break
    
    return best_x, best_energy

def _basin_hopping(x, start_time, time_limit, n_restarts=5):
    """Basin hopping: perturb and re-minimize to escape local minima."""
    best_x = x.copy()
    best_energy, _ = _energy_and_grad_flat(x.flatten())
    rng = np.random.default_rng(42)
    
    for _ in range(n_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Random perturbation
        noise_scale = 0.1
        noise = rng.normal(0, noise_scale, size=x.shape)
        # Project noise to tangent space
        noise_tan = noise - np.sum(noise * x, axis=1, keepdims=True) * x
        x_perturbed = _project_to_sphere(x + noise_tan)
        
        # Quick local minimization
        x_opt, e_opt = _momentum_optimize(x_perturbed, start_time, time.time() - start_time + 15)
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_x = x_opt.copy()
    
    return best_x, best_energy

def construct():
    start_time = time.time()
    total_time_limit = 145.0  # Safety margin for 150s limit
    
    # Phase 1: Multi-start initialization and optimization
    inits = [
        _fibonacci_spiral(N),
        _geodesic_dome_init(N),
        _fibonacci_spiral(N) + np.random.default_rng(1).normal(0, 0.1, (N, 3))
    ]
    inits = [_project_to_sphere(x) for x in inits]
    
    best_x = None
    best_energy = float('inf')
    
    # Try each initialization with momentum optimization
    per_init_time = 25.0
    for i, x_init in enumerate(inits):
        if time.time() - start_time > total_time_limit - 30:
            break
        x_opt, e_opt = _momentum_optimize(x_init, start_time, start_time + per_init_time)
        if e_opt < best_energy:
            best_energy = e_opt
            best_x = x_opt.copy()
    
    # Phase 2: Basin hopping from the best solution
    if best_x is not None and time.time() - start_time < total_time_limit - 25:
        remaining_for_bh = min(40.0, total_time_limit - (time.time() - start_time) - 20)
        if remaining_for_bh > 5:
            best_x, best_energy = _basin_hopping(best_x, start_time, start_time + remaining_for_bh, n_restarts=3)
    
    # Phase 3: L-BFGS polish
    remaining = total_time_limit - (time.time() - start_time)
    if remaining > 15 and best_x is not None:
        best_x = _lbfgs_polish(best_x, remaining - 10, tight=False)
    
    # Phase 4: Final tight L-BFGS polish
    remaining = total_time_limit - (time.time() - start_time)
    if remaining > 10 and best_x is not None:
        best_x = _lbfgs_polish(best_x, remaining - 5, tight=True)
    
    # Final validation
    if best_x is None:
        best_x = _fibonacci_spiral(N)
    best_x = _project_to_sphere(best_x)
    return best_x

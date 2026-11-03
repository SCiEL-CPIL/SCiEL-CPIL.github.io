"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial import cKDTree


def _log_energy(points):
    """Compute log(1/d) energy for points on unit sphere using math.fsum."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, 1.0)
    iu = np.triu_indices(n, k=1)
    log_dists = np.log(dists[iu])
    # Use math.fsum for accurate summation as specified
    return -math.fsum(log_dists.tolist())


def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _geodesic_dome(n):
    """Generate points via geodesic dome (icosahedron subdivision) then project to n points."""
    # Start with icosahedron vertices
    t = (1.0 + math.sqrt(5.0)) / 2.0
    ico = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    ico /= np.linalg.norm(ico, axis=1, keepdims=True)
    
    # Subdivide faces until we have enough vertices
    def subdivide(points, faces, levels):
        for _ in range(levels):
            new_faces = []
            mid_cache = {}
            for f in faces:
                mids = []
                for i in range(3):
                    j = (i + 1) % 3
                    key = tuple(sorted([f[i], f[j]]))
                    if key not in mid_cache:
                        mid = (points[key[0]] + points[key[1]]) / 2.0
                        mid /= np.linalg.norm(mid)
                        mid_cache[key] = len(points)
                        points = np.vstack([points, mid])
                    mids.append(mid_cache[key])
                new_faces.append([f[0], mids[0], mids[2]])
                new_faces.append([f[1], mids[1], mids[0]])
                new_faces.append([f[2], mids[2], mids[1]])
                new_faces.append([mids[0], mids[1], mids[2]])
            faces = new_faces
        return points, faces
    
    ico_faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    pts, _ = subdivide(ico, ico_faces, 3)  # 1280 vertices
    
    # Select n points from the dome using farthest-point sampling
    rng = np.random.RandomState(123)
    selected = [rng.randint(len(pts))]
    min_dists = np.linalg.norm(pts - pts[selected[0]], axis=1)
    
    for _ in range(n - 1):
        next_idx = np.argmax(min_dists)
        selected.append(next_idx)
        new_dists = np.linalg.norm(pts - pts[next_idx], axis=1)
        min_dists = np.minimum(min_dists, new_dists)
    
    result = pts[selected]
    # Normalize
    result /= np.linalg.norm(result, axis=1, keepdims=True)
    return result


def _random_uniform_sphere(n, seed=42):
    """Generate random uniform points on sphere."""
    rng = np.random.RandomState(seed)
    points = rng.randn(n, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms


def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _energy_and_grad(x):
    """Energy and gradient for L-BFGS-B. x is flat array of shape (N*3,)."""
    points = x.reshape(N, 3)
    points = _project_to_sphere(points)
    
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    
    # Gradient of E = -sum_{i<j} log(d_ij)
    # dE/dx_i = sum_{j!=i} (x_j - x_i) / d_ij^2
    weights = np.zeros_like(dists_sq)
    mask = ~np.eye(n, dtype=bool)
    weights[mask] = 1.0 / dists_sq[mask]
    
    sum_weights = np.sum(weights, axis=1)
    weighted_sum = weights @ points
    
    grads = weighted_sum - points * sum_weights[:, np.newaxis]
    
    # Project gradient to tangent space
    dot_products = np.sum(grads * points, axis=1, keepdims=True)
    grads_tangent = grads - dot_products * points
    
    energy = _log_energy(points)
    return energy, grads_tangent.flatten()


def _simulated_annealing(points, start_time, deadline, max_iter=50000):
    """Simulated annealing on the sphere with multi-point moves."""
    best_points = points.copy()
    best_energy = _log_energy(best_points)
    current_points = points.copy()
    current_energy = best_energy
    
    T = 0.01
    T_min = 1e-8
    cooling = 0.9995
    
    for _ in range(max_iter):
        if time.time() > deadline - 2.0:
            break
        
        # Occasionally do multi-point moves (move 2-3 points at once)
        if np.random.rand() < 0.1:
            n_move = np.random.randint(2, 4)
            indices = np.random.choice(N, size=n_move, replace=False)
            new_points = current_points.copy()
            for idx in indices:
                rand_vec = np.random.randn(3)
                tangent = rand_vec - np.dot(rand_vec, current_points[idx]) * current_points[idx]
                tangent_norm = np.linalg.norm(tangent)
                if tangent_norm < 1e-15:
                    continue
                tangent = tangent / tangent_norm
                step_size = T * 2.0
                new_point = current_points[idx] + step_size * tangent
                new_point = new_point / np.linalg.norm(new_point)
                new_points[idx] = new_point
            
            new_energy = _log_energy(new_points)
            delta_e = new_energy - current_energy
            
            if delta_e < 0 or np.random.rand() < math.exp(-delta_e / max(T, T_min)):
                current_points = new_points
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_points = current_points.copy()
        else:
            # Single point move
            idx = np.random.randint(N)
            rand_vec = np.random.randn(3)
            tangent = rand_vec - np.dot(rand_vec, current_points[idx]) * current_points[idx]
            tangent_norm = np.linalg.norm(tangent)
            if tangent_norm < 1e-15:
                continue
            tangent = tangent / tangent_norm
            
            step_size = T * 2.0
            new_point = current_points[idx] + step_size * tangent
            new_point = new_point / np.linalg.norm(new_point)
            
            new_points = current_points.copy()
            new_points[idx] = new_point
            
            new_energy = _log_energy(new_points)
            delta_e = new_energy - current_energy
            
            if delta_e < 0 or np.random.rand() < math.exp(-delta_e / max(T, T_min)):
                current_points = new_points
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_points = current_points.copy()
        
        T = max(T * cooling, T_min)
    
    return best_points, best_energy


def _velocity_verlet_sphere(points, start_time, deadline, max_iter=20000, dt=0.01, damping=0.999):
    """Velocity-Verlet dynamics on the sphere for basin exploration."""
    best_points = points.copy()
    best_energy = _log_energy(best_points)
    current_points = points.copy()
    velocities = np.zeros_like(current_points)
    current_energy = best_energy
    
    for _ in range(max_iter):
        if time.time() > deadline - 2.0:
            break
        
        # Compute forces (negative gradient)
        n = len(current_points)
        diff = current_points[:, np.newaxis, :] - current_points[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(dists_sq, 1.0)
        
        weights = np.zeros_like(dists_sq)
        mask = ~np.eye(n, dtype=bool)
        weights[mask] = 1.0 / dists_sq[mask]
        
        sum_weights = np.sum(weights, axis=1)
        weighted_sum = weights @ current_points
        
        grads = weighted_sum - current_points * sum_weights[:, np.newaxis]
        
        # Project to tangent space
        dot_products = np.sum(grads * current_points, axis=1, keepdims=True)
        grads_tangent = grads - dot_products * current_points
        
        # Update velocities with damping
        velocities = velocities * damping - grads_tangent * dt
        
        # Update positions
        new_points = current_points + velocities * dt
        new_points = _project_to_sphere(new_points)
        
        new_energy = _log_energy(new_points)
        
        # Accept if energy decreased or randomly with Metropolis criterion
        if new_energy < current_energy:
            current_points = new_points
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = current_points.copy()
        else:
            # Occasionally accept worse states to escape local minima
            if np.random.rand() < 0.05:
                current_points = new_points
                current_energy = new_energy
    
    return best_points, best_energy


def _gradient_descent_sphere(points, max_iter=10000, lr=0.01, start_time=None, deadline=None):
    """Gradient descent on the sphere with adaptive learning rate."""
    current_points = points.copy()
    best_points = points.copy()
    best_energy = _log_energy(current_points)
    current_energy = best_energy
    
    for i in range(max_iter):
        if start_time is not None and time.time() > deadline - 2.0:
            break
        
        # Compute gradient
        n = len(current_points)
        diff = current_points[:, np.newaxis, :] - current_points[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(dists_sq, 1.0)
        
        weights = np.zeros_like(dists_sq)
        mask = ~np.eye(n, dtype=bool)
        weights[mask] = 1.0 / dists_sq[mask]
        
        sum_weights = np.sum(weights, axis=1)
        weighted_sum = weights @ current_points
        
        grads = weighted_sum - current_points * sum_weights[:, np.newaxis]
        
        # Project to tangent space
        dot_products = np.sum(grads * current_points, axis=1, keepdims=True)
        grads_tangent = grads - dot_products * current_points
        
        # Adaptive learning rate
        grad_norm = np.linalg.norm(grads_tangent)
        if grad_norm < 1e-10:
            break
        
        # Update
        new_points = current_points - lr * grads_tangent
        new_points = _project_to_sphere(new_points)
        
        new_energy = _log_energy(new_points)
        if new_energy < current_energy:
            current_points = new_points
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = current_points.copy()
            lr = min(lr * 1.05, 0.1)  # increase lr slightly
        else:
            lr = lr * 0.9  # decrease lr
        
        if lr < 1e-10:
            break
    
    return best_points, best_energy


def _adam_optimize(points, start_time, deadline, max_iter=5000, lr=0.005, beta1=0.9, beta2=0.999, eps=1e-8):
    """Adam optimizer on the sphere with tangent-space projection."""
    current_points = points.copy()
    best_points = points.copy()
    best_energy = _log_energy(current_points)
    current_energy = best_energy
    
    m = np.zeros_like(current_points)
    v = np.zeros_like(current_points)
    
    for t in range(1, max_iter + 1):
        if time.time() > deadline - 2.0:
            break
        
        n = len(current_points)
        diff = current_points[:, np.newaxis, :] - current_points[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(dists_sq, 1.0)
        
        weights = np.zeros_like(dists_sq)
        mask = ~np.eye(n, dtype=bool)
        weights[mask] = 1.0 / dists_sq[mask]
        
        sum_weights = np.sum(weights, axis=1)
        weighted_sum = weights @ current_points
        
        grads = weighted_sum - current_points * sum_weights[:, np.newaxis]
        
        # Project to tangent space
        dot_products = np.sum(grads * current_points, axis=1, keepdims=True)
        grads_tangent = grads - dot_products * current_points
        
        # Adam update
        m = beta1 * m + (1 - beta1) * grads_tangent
        v = beta2 * v + (1 - beta2) * grads_tangent ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        update = m_hat / (np.sqrt(v_hat) + eps)
        
        # Adaptive learning rate: scale by gradient norm
        grad_norm = np.linalg.norm(grads_tangent)
        if grad_norm < 1e-12:
            break
        effective_lr = lr * min(1.0, 1.0 / (grad_norm + 1e-6))
        
        new_points = current_points - effective_lr * update
        new_points = _project_to_sphere(new_points)
        
        new_energy = _log_energy(new_points)
        if new_energy < current_energy:
            current_points = new_points
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = current_points.copy()
        else:
            # Halve learning rate if stuck
            lr = max(lr * 0.95, 1e-10)
    
    return best_points, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Multiple initializations
    inits = [
        _fibonacci_sphere(N),
        _geodesic_dome(N),
        _random_uniform_sphere(N, seed=1),
        _random_uniform_sphere(N, seed=2),
        _random_uniform_sphere(N, seed=3),
        _random_uniform_sphere(N, seed=4),
        _random_uniform_sphere(N, seed=5),
        _random_uniform_sphere(N, seed=6),
        _random_uniform_sphere(N, seed=7),
    ]
    
    # Add perturbed Fibonacci
    rng = np.random.RandomState(42)
    for k in range(3):
        perturbed = _fibonacci_sphere(N) + rng.randn(N, 3) * 0.01
        inits.append(_project_to_sphere(perturbed))
    
    # Add perturbed geodesic dome
    for k in range(2):
        perturbed = _geodesic_dome(N) + rng.randn(N, 3) * 0.01
        inits.append(_project_to_sphere(perturbed))
    
    best_points = inits[0].copy()
    best_energy = _log_energy(best_points)
    
    # Phase 1: L-BFGS-B optimization from multiple starts
    for idx, init in enumerate(inits):
        if time.time() > deadline - 45.0:
            break
        
        x0 = init.flatten()
        
        try:
            res = minimize(
                _energy_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 3000, 'maxfun': 30000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            
            if res.fun < best_energy:
                p_res = _project_to_sphere(res.x.reshape(N, 3))
                e_check = _log_energy(p_res)
                if e_check < best_energy:
                    best_energy = e_check
                    best_points = p_res.copy()
        except Exception:
            pass
    
    # Phase 2: Velocity-Verlet dynamics for basin exploration
    if time.time() < deadline - 25.0:
        vv_points, vv_energy = _velocity_verlet_sphere(
            best_points, start_time, deadline, max_iter=15000, dt=0.01, damping=0.999
        )
        if vv_energy < best_energy:
            best_energy = vv_energy
            best_points = vv_points.copy()
    
    # Phase 3: Gradient descent refinement
    if time.time() < deadline - 15.0:
        gd_points, gd_energy = _gradient_descent_sphere(
            best_points, max_iter=20000, lr=0.005, 
            start_time=start_time, deadline=deadline
        )
        if gd_energy < best_energy:
            best_energy = gd_energy
            best_points = gd_points.copy()
    
    # Phase 4: Simulated annealing for fine-tuning
    if time.time() < deadline - 10.0:
        sa_points, sa_energy = _simulated_annealing(
            best_points, start_time, deadline, max_iter=30000
        )
        if sa_energy < best_energy:
            best_energy = sa_energy
            best_points = sa_points.copy()
    
    # Phase 5: Adam optimizer for tight local polish
    if time.time() < deadline - 8.0:
        adam_points, adam_energy = _adam_optimize(
            best_points, start_time, deadline, max_iter=5000, lr=0.005
        )
        if adam_energy < best_energy:
            best_energy = adam_energy
            best_points = adam_points.copy()
    
    # Phase 6: Final high-precision L-BFGS-B polish
    if time.time() < deadline - 5.0:
        x_best = best_points.flatten()
        try:
            res = minimize(
                _energy_and_grad,
                x_best,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 10000, 'maxfun': 100000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 100}
            )
            p_res = _project_to_sphere(res.x.reshape(N, 3))
            e_check = _log_energy(p_res)
            if e_check < best_energy:
                best_energy = e_check
                best_points = p_res.copy()
        except Exception:
            pass
    
    # Phase 7: Very tight final polish if time remains
    if time.time() < deadline - 2.0:
        x_best = best_points.flatten()
        try:
            res = minimize(
                _energy_and_grad,
                x_best,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 5000, 'maxfun': 50000, 'ftol': 1e-17, 'gtol': 1e-15, 'maxls': 200}
            )
            p_res = _project_to_sphere(res.x.reshape(N, 3))
            e_check = _log_energy(p_res)
            if e_check < best_energy:
                best_energy = e_check
                best_points = p_res.copy()
        except Exception:
            pass
    
    # Final Adam micro-polish if time remains
    if time.time() < deadline - 1.5:
        adam_points, adam_energy = _adam_optimize(
            best_points, start_time, deadline, max_iter=2000, lr=0.001
        )
        if adam_energy < best_energy:
            best_energy = adam_energy
            best_points = adam_points.copy()
    
    best_points = _project_to_sphere(best_points)
    
    if not np.all(np.isfinite(best_points)):
        return _fibonacci_sphere(N)
    
    # Check for points too close
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    iu = np.triu_indices(N, k=1)
    if np.min(dists[iu]) < 1e-9:
        return _fibonacci_sphere(N)
    
    return best_points
# EVOLVE-BLOCK-END

<<<<<<< SEARCH
import numpy as np
import time
import math

N = 150
TIME_LIMIT = 115.0  # Safety margin before 120s kill

def _energy_and_grad(pos):
    """
    Compute logarithmic energy and gradient for points on unit sphere.
    
    pos: (N, 3) array of points on unit sphere.
    Returns (energy, grad) where grad is the unconstrained gradient w.r.t. pos.
    Energy is sum_{i<j} log(1 / |x_i - x_j|).
    """
    # Pairwise differences
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero for self-pairs
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    # Energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    iu = np.triu_indices(N, k=1)
    d = dist[iu]
    # Clip to avoid log(0) if points are extremely close
    d = np.clip(d, 1e-12, None)
    energy = -np.sum(np.log(d))
    
    # Gradient computation:
    # d/dp_i [ -log(|p_i - p_j|) ] = -(p_i - p_j) / |p_i - p_j|^2
    # grad_i = sum_{j != i} (p_j - p_i) / |p_i - p_j|^2
    #         = sum_{j != i} p_j / d_ij^2 - p_i * sum_{j != i} 1/d_ij^2
    
    inv_d2 = 1.0 / dist_sq
    np.fill_diagonal(inv_d2, 0.0) # Exclude self-interaction for gradient
    
    # sum_j p_j / d_ij^2 for each i
    sum_pj_over_d2 = inv_d2 @ pos
    # sum_j 1/d_ij^2 for each i
    sum_inv_d2 = np.sum(inv_d2, axis=1)
    
    grad = sum_pj_over_d2 - pos * sum_inv_d2[:, np.newaxis]
    
    return energy, grad

def _project_to_sphere(pos):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pos, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pos / norms

def _optimize_adam(pos_init, time_budget, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Optimize using Projected Adam Optimizer.
    """
    start_time = time.time()
    pos = _project_to_sphere(pos_init.copy())
    energy, grad = _energy_and_grad(pos)
    best_pos = pos.copy()
    best_energy = energy
    
    # Adam state
    m = np.zeros_like(pos)
    v = np.zeros_like(pos)
    t = 0
    
    # Convergence check
    min_improvement = 1e-12
    last_energy = energy
    stall_count = 0
    max_stall = 500 # If no improvement for 500 steps, restart or perturb
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
        
        t += 1
        
        # Project gradient onto tangent space
        grad_dot_x = np.sum(grad * pos, axis=1, keepdims=True)
        grad_tangent = grad - grad_dot_x * pos
        
        # Check convergence
        grad_norm = np.linalg.norm(grad_tangent)
        if grad_norm < 1e-10:
            break
            
        # Adam update
        m = beta1 * m + (1 - beta1) * grad_tangent
        v = beta2 * v + (1 - beta2) * (grad_tangent ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        # Update direction
        direction = -m_hat / (np.sqrt(v_hat) + epsilon)
        
        # Step size
        # We can use a fixed lr or adaptive. Let's stick to fixed lr for simplicity in projection
        # But we need to ensure the step is not too large.
        # The magnitude of direction is roughly 1.
        
        new_pos = pos + lr * direction
        new_pos = _project_to_sphere(new_pos)
        
        new_energy, new_grad = _energy_and_grad(new_pos)
        
        # Accept step if it improves energy (or even if it doesn't, Adam is usually robust, 
        # but for safety we can do a simple check)
        # Actually, Adam can diverge if lr is too high. 
        # Let's add a simple backtracking if energy increases significantly.
        if new_energy > energy + 1.0:
            # Try smaller step
            new_pos = pos + 0.5 * lr * direction
            new_pos = _project_to_sphere(new_pos)
            new_energy, new_grad = _energy_and_grad(new_pos)
            if new_energy > energy + 1.0:
                # Skip this step or reduce lr
                lr *= 0.5
                if lr < 1e-10:
                    break
                continue
        
        pos = new_pos
        energy = new_energy
        grad = new_grad
        
        if energy < best_energy:
            best_energy = energy
            best_pos = pos.copy()
            stall_count = 0
        else:
            stall_count += 1
            if stall_count > max_stall:
                # Perturb and reset
                rng = np.random.default_rng()
                pos = pos + rng.normal(0, 0.1, size=(N, 3))
                pos = _project_to_sphere(pos)
                energy, grad = _energy_and_grad(pos)
                m = np.zeros_like(pos)
                v = np.zeros_like(pos)
                t = 0
                stall_count = 0
                # Reset lr to default
                lr = 0.01
                # If energy is much worse, revert to best
                if energy > best_energy * 1.1:
                     pos = best_pos.copy()
                     energy, grad = _energy_and_grad(pos)
                     m = np.zeros_like(pos)
                     v = np.zeros_like(pos)
                     t = 0

    return _project_to_sphere(best_pos), best_energy

def construct():
    start_time = time.time()
    
    starts = []
    
    # 1. Fibonacci spiral (standard)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pos_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    starts.append(_project_to_sphere(pos_fib))
    
    # 2. Fibonacci spiral with small perturbation
    rng = np.random.default_rng(42)
    pos_fib_pert = pos_fib + rng.normal(0, 0.01, size=(N, 3))
    starts.append(_project_to_sphere(pos_fib_pert))
    
    # 3. Random uniform points
    for _ in range(3):
        pos_rand = rng.normal(0, 1, size=(N, 3))
        starts.append(_project_to_sphere(pos_rand))
        
    # 4. More random ones for diversity
    for _ in range(2):
        pos_rand = rng.normal(0, 1, size=(N, 3))
        starts.append(_project_to_sphere(pos_rand))

    best_pos = starts[0]
    best_energy = float('inf')
    
    total_opt_time = TIME_LIMIT - 10.0
    time_per_start = [
        total_opt_time * 0.40, # Fib (best chance)
        total_opt_time * 0.20, # Fib Pert
        total_opt_time * 0.10, # Rand 1
        total_opt_time * 0.10, # Rand 2
        total_opt_time * 0.10, # Rand 3
        total_opt_time * 0.10, # Rand 4
        total_opt_time * 0.10, # Rand 5
    ]
    
    for idx, start_pos in enumerate(starts):
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 5:
            break
            
        alloc_time = min(time_per_start[idx], TIME_LIMIT - 5 - elapsed)
        if alloc_time < 1.0:
            continue
            
        pos, energy = _optimize_adam(start_pos, alloc_time)
        
        if energy < best_energy:
            best_energy = energy
            best_pos = pos.copy()
            
    # Final validation and projection
    best_pos = _project_to_sphere(best_pos)
    
    if not np.all(np.isfinite(best_pos)):
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        best_pos = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        best_pos = _project_to_sphere(best_pos)
        
    # Check minimum distance
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, 1.0)
    min_dist = np.min(dist)
    
    if min_dist < 1e-9:
        rng = np.random.default_rng(123)
        perturbed = best_pos + rng.normal(0, 0.01, size=(N, 3))
        perturbed = _project_to_sphere(perturbed)
        pos, energy = _optimize_adam(perturbed, 5.0)
        if energy < best_energy:
            best_pos = pos
            best_energy = energy
            best_pos = _project_to_sphere(best_pos)

    return best_pos
=======
import numpy as np
import time
import math

N = 150
TIME_LIMIT = 115.0  # Safety margin before 120s kill

def _energy_and_grad(pos):
    """
    Compute logarithmic energy and gradient for points on unit sphere.
    
    pos: (N, 3) array of points on unit sphere.
    Returns (energy, grad) where grad is the unconstrained gradient w.r.t. pos.
    Energy is sum_{i<j} log(1 / |x_i - x_j|).
    """
    # Pairwise differences
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero for self-pairs
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    # Energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    iu = np.triu_indices(N, k=1)
    d = dist[iu]
    # Clip to avoid log(0) if points are extremely close
    d = np.clip(d, 1e-12, None)
    energy = -np.sum(np.log(d))
    
    # Gradient computation:
    # d/dp_i [ -log(|p_i - p_j|) ] = -(p_i - p_j) / |p_i - p_j|^2
    # grad_i = sum_{j != i} (p_j - p_i) / |p_i - p_j|^2
    #         = sum_{j != i} p_j / d_ij^2 - p_i * sum_{j != i} 1/d_ij^2
    
    inv_d2 = 1.0 / dist_sq
    np.fill_diagonal(inv_d2, 0.0) # Exclude self-interaction for gradient
    
    # sum_j p_j / d_ij^2 for each i
    sum_pj_over_d2 = inv_d2 @ pos
    # sum_j 1/d_ij^2 for each i
    sum_inv_d2 = np.sum(inv_d2, axis=1)
    
    grad = sum_pj_over_d2 - pos * sum_inv_d2[:, np.newaxis]
    
    return energy, grad

def _hessian_diag(pos):
    """
    Compute diagonal of Hessian for Newton step.
    H_ii = sum_{j!=i} 2/d_ij^2
    """
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    
    inv_d2 = 1.0 / dist_sq
    np.fill_diagonal(inv_d2, 0.0)
    
    return 2.0 * np.sum(inv_d2, axis=1)

def _project_to_sphere(pos):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pos, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pos / norms

def _geodesic_dome(n_points, seed=42):
    """
    Generate points on sphere using geodesic dome (subdivided icosahedron).
    """
    # Icosahedron vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico_verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    # Icosahedron faces
    ico_faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    verts = ico_verts.copy()
    faces = ico_faces.copy()
    
    # Subdivide until we have enough points
    while len(verts) < n_points:
        new_faces = []
        mid_cache = {}
        
        def get_midpoint(i, j):
            key = (min(i, j), max(i, j))
            if key in mid_cache:
                return mid_cache[key]
            mid = (verts[i] + verts[j]) / 2.0
            mid /= np.linalg.norm(mid)
            idx = len(verts)
            verts = np.vstack([verts, mid.reshape(1, -1)])
            mid_cache[key] = idx
            return idx
        
        for face in faces:
            a, b, c = face
            ab = get_midpoint(a, b)
            bc = get_midpoint(b, c)
            ca = get_midpoint(c, a)
            new_faces.extend([
                [a, ab, ca],
                [ab, b, bc],
                [ca, bc, c],
                [ab, bc, ca]
            ])
        faces = new_faces
    
    # Select n_points from the vertices
    if len(verts) <= n_points:
        return verts
    else:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(verts), size=n_points, replace=False)
        return verts[indices]

def _optimize_adam(pos_init, time_budget, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Optimize using Projected Adam Optimizer.
    """
    start_time = time.time()
    pos = _project_to_sphere(pos_init.copy())
    energy, grad = _energy_and_grad(pos)
    best_pos = pos.copy()
    best_energy = energy
    
    # Adam state
    m = np.zeros_like(pos)
    v = np.zeros_like(pos)
    t = 0
    
    # Convergence check
    min_improvement = 1e-12
    last_energy = energy
    stall_count = 0
    max_stall = 500 # If no improvement for 500 steps, restart or perturb
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
        
        t += 1
        
        # Project gradient onto tangent space
        grad_dot_x = np.sum(grad * pos, axis=1, keepdims=True)
        grad_tangent = grad - grad_dot_x * pos
        
        # Check convergence
        grad_norm = np.linalg.norm(grad_tangent)
        if grad_norm < 1e-10:
            break
            
        # Adam update
        m = beta1 * m + (1 - beta1) * grad_tangent
        v = beta2 * v + (1 - beta2) * (grad_tangent ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        # Update direction
        direction = -m_hat / (np.sqrt(v_hat) + epsilon)
        
        # Step size
        new_pos = pos + lr * direction
        new_pos = _project_to_sphere(new_pos)
        
        new_energy, new_grad = _energy_and_grad(new_pos)
        
        # Accept step if it improves energy
        if new_energy > energy + 1.0:
            # Try smaller step
            new_pos = pos + 0.5 * lr * direction
            new_pos = _project_to_sphere(new_pos)
            new_energy, new_grad = _energy_and_grad(new_pos)
            if new_energy > energy + 1.0:
                # Skip this step or reduce lr
                lr *= 0.5
                if lr < 1e-10:
                    break
                continue
        
        pos = new_pos
        energy = new_energy
        grad = new_grad
        
        if energy < best_energy:
            best_energy = energy
            best_pos = pos.copy()
            stall_count = 0
        else:
            stall_count += 1
            if stall_count > max_stall:
                # Perturb and reset
                rng = np.random.default_rng()
                pos = pos + rng.normal(0, 0.1, size=(N, 3))
                pos = _project_to_sphere(pos)
                energy, grad = _energy_and_grad(pos)
                m = np.zeros_like(pos)
                v = np.zeros_like(pos)
                t = 0
                stall_count = 0
                # Reset lr to default
                lr = 0.01
                # If energy is much worse, revert to best
                if energy > best_energy * 1.1:
                     pos = best_pos.copy()
                     energy, grad = _energy_and_grad(pos)
                     m = np.zeros_like(pos)
                     v = np.zeros_like(pos)
                     t = 0

    return _project_to_sphere(best_pos), best_energy

def _newton_polish(pos, time_budget):
    """
    Newton polish step on tangent space with diagonal Hessian approximation.
    """
    start_time = time.time()
    pos = _project_to_sphere(pos.copy())
    energy, grad = _energy_and_grad(pos)
    best_pos = pos.copy()
    best_energy = energy
    
    max_iter = 500
    
    for it in range(max_iter):
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
        
        # Project gradient to tangent space
        grad_dot_x = np.sum(grad * pos, axis=1, keepdims=True)
        grad_tangent = grad - grad_dot_x * pos
        
        grad_norm = np.linalg.norm(grad_tangent)
        if grad_norm < 1e-12:
            break
        
        # Diagonal Hessian approximation
        diag_hess = _hessian_diag(pos)
        
        # Newton step: delta = -grad_tangent / diag_hess
        delta = -grad_tangent / (diag_hess[:, np.newaxis] + 1e-6)
        
        # Limit step size
        step_norm = np.linalg.norm(delta, axis=1, keepdims=True)
        max_step = 0.1
        mask = step_norm > max_step
        delta[mask] = delta[mask] * (max_step / step_norm[mask])
        
        new_pos = pos + delta
        new_pos = _project_to_sphere(new_pos)
        
        new_energy, new_grad = _energy_and_grad(new_pos)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_pos = new_pos.copy()
        
        if new_energy < energy:
            pos = new_pos
            energy = new_energy
            grad = new_grad
        else:
            # Damp step
            delta = delta * 0.5
            new_pos = pos + delta
            new_pos = _project_to_sphere(new_pos)
            new_energy, new_grad = _energy_and_grad(new_pos)
            if new_energy < energy:
                pos = new_pos
                energy = new_energy
                grad = new_grad
            else:
                break
    
    return _project_to_sphere(best_pos), best_energy

def construct():
    start_time = time.time()
    
    starts = []
    
    # 1. Geodesic dome (primary initialization)
    pos_dome = _geodesic_dome(N)
    starts.append(_project_to_sphere(pos_dome))
    
    # 2. Geodesic dome with small perturbation
    rng = np.random.default_rng(42)
    pos_dome_pert = pos_dome + rng.normal(0, 0.01, size=(N, 3))
    starts.append(_project_to_sphere(pos_dome_pert))
    
    # 3. Fibonacci spiral (secondary)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pos_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    starts.append(_project_to_sphere(pos_fib))
    
    # 4. One random start for diversity
    pos_rand = rng.normal(0, 1, size=(N, 3))
    starts.append(_project_to_sphere(pos_rand))
        
    best_pos = starts[0]
    best_energy = float('inf')
    
    total_opt_time = TIME_LIMIT - 15.0  # Reserve 15s for Newton polish
    time_per_start = [
        total_opt_time * 0.40, # Dome (best chance)
        total_opt_time * 0.25, # Dome Pert
        total_opt_time * 0.25, # Fib
        total_opt_time * 0.10, # Rand
    ]
    
    for idx, start_pos in enumerate(starts):
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 20:
            break
            
        alloc_time = min(time_per_start[idx], TIME_LIMIT - 20 - elapsed)
        if alloc_time < 1.0:
            continue
            
        pos, energy = _optimize_adam(start_pos, alloc_time)
        
        if energy < best_energy:
            best_energy = energy
            best_pos = pos.copy()
    
    # Newton polish on the best result
    elapsed = time.time() - start_time
    polish_time = min(10.0, TIME_LIMIT - 5 - elapsed)
    if polish_time > 1.0:
        polished_pos, polished_energy = _newton_polish(best_pos, polish_time)
        if polished_energy < best_energy:
            best_energy = polished_energy
            best_pos = polished_pos
    
    # Final validation and projection
    best_pos = _project_to_sphere(best_pos)
    
    if not np.all(np.isfinite(best_pos)):
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        best_pos = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        best_pos = _project_to_sphere(best_pos)
        
    # Check minimum distance
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, 1.0)
    min_dist = np.min(dist)
    
    if min_dist < 1e-9:
        rng = np.random.default_rng(123)
        perturbed = best_pos + rng.normal(0, 0.01, size=(N, 3))
        perturbed = _project_to_sphere(perturbed)
        pos, energy = _optimize_adam(perturbed, 5.0)
        if energy < best_energy:
            best_pos = pos
            best_energy = energy
            best_pos = _project_to_sphere(best_pos)

    return best_pos
>>>>>>> REPLACE

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_score(points):
    """Compute the packing score: min of pairwise dist/2 and boundary dist."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    radii = np.sqrt(np.sum(points ** 2, axis=1))
    min_boundary = np.min(1.0 - radii)
    return min(min_pair, min_boundary)

def _physics_energy_and_grad(centers, repulsion_power=4.0, wall_strength=50.0, margin=0.01):
    """
    Compute energy and gradient for physics-based relaxation.
    
    Energy = repulsion_term + wall_term
    - Repulsion: sum over i<j of 1/d_ij^repulsion_power (strong inverse power)
    - Wall: sum over i of max(0, |c_i| - (1-margin))^2 * wall_strength
    
    Gradient is computed analytically.
    """
    n = len(centers)
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # Pairwise repulsion
    for i in range(n):
        for j in range(i + 1, n):
            d_vec = centers[i] - centers[j]
            d = np.linalg.norm(d_vec)
            if d < 1e-12:
                # Avoid division by zero; add small random perturbation
                d_vec = np.random.normal(0, 1e-6, 2)
                d = np.linalg.norm(d_vec)
            
            # Energy contribution: 1/d^p
            inv_d_pow = 1.0 / (d ** repulsion_power)
            energy += inv_d_pow
            
            # Gradient: d/dx_i (1/d^p) = -p * (x_i - x_j) / d^(p+2)
            # Since d^2 = |x_i-x_j|^2, d/dx_i (d^2) = 2*(x_i-x_j)
            # d/dx_i (1/d^p) = -p * d^(-p-1) * d(d)/dx_i
            # d(d)/dx_i = (x_i - x_j)/d
            # So d/dx_i (1/d^p) = -p * (x_i - x_j) / d^(p+1)
            # Wait, let me redo: f = d^(-p), df/dd = -p * d^(-p-1)
            # dd/dx_i = (x_i - x_j)/d
            # df/dx_i = -p * d^(-p-1) * (x_i - x_j)/d = -p * (x_i - x_j) * d^(-p-2)
            
            coeff = -repulsion_power * inv_d_pow / (d * d)
            grad[i] += coeff * d_vec
            grad[j] -= coeff * d_vec
    
    # Wall potential
    radii = np.linalg.norm(centers, axis=1)
    for i in range(n):
        r = radii[i]
        if r > (1.0 - margin):
            excess = r - (1.0 - margin)
            energy += wall_strength * excess ** 2
            # Gradient: d/dx_i [wall_strength * (r - (1-margin))^2] 
            # = 2 * wall_strength * excess * dr/dx_i
            # dr/dx_i = x_i / r
            if r > 1e-12:
                grad[i] += 2.0 * wall_strength * excess * (centers[i] / r)
    
    return energy, grad

def _physics_relax(centers, max_iters=20000, lr_init=0.001, lr_decay=0.999, 
                   repulsion_power=4.0, wall_strength=50.0, margin=0.01,
                   time_limit=None, start_time=None):
    """
    Physics-based relaxation using gradient descent with adaptive learning rate.
    """
    if start_time is None:
        start_time = time.time()
    
    centers = centers.copy()
    current_energy, grad = _physics_energy_and_grad(centers, repulsion_power, wall_strength, margin)
    
    lr = lr_init
    best_energy = current_energy
    best_centers = centers.copy()
    
    for it in range(max_iters):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        # Adaptive learning rate: reduce if energy increases
        step = lr * grad
        new_centers = centers - step
        
        # Check if any point went outside unit circle significantly
        new_radii = np.linalg.norm(new_centers, axis=1)
        if np.any(new_radii > 1.05):
            # Scale back
            scale = 1.0 / np.max(new_radii) * 0.95
            new_centers = new_centers * scale
        
        new_energy, new_grad = _physics_energy_and_grad(new_centers, repulsion_power, wall_strength, margin)
        
        if new_energy < current_energy:
            centers = new_centers
            current_energy = new_energy
            grad = new_grad
            if current_energy < best_energy:
                best_energy = current_energy
                best_centers = centers.copy()
            # Slightly increase learning rate on success
            lr = min(lr * 1.01, lr_init * 2.0)
        else:
            # Reduce learning rate
            lr *= lr_decay
        
        if lr < 1e-10:
            break
    
    return best_centers

def _hexagonal_seed(n):
    """Generate a hexagonal lattice seed for n circles."""
    spacing = 2.0 / (np.sqrt(n) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        if len(points) >= n:
            idx = np.argsort(radius[mask], kind="stable")[:n]
            return points[idx]
        spacing *= 0.98

def _random_seed(n):
    """Generate random initial positions inside unit circle."""
    angles = np.random.uniform(0, 2 * np.pi, n)
    radii = np.sqrt(np.random.uniform(0, 1, n)) * 0.9
    points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return points

def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin before 150s kill
    
    best_centers = None
    best_r = 0.0
    
    # Multiple initializations with different strategies
    initializations = []
    
    # 1. Hexagonal lattice
    initializations.append(_hexagonal_seed(N))
    
    # 2. Random
    initializations.append(_random_seed(N))
    
    # 3. Concentric rings
    ring_counts = [1, 6, 12, 18, 13]  # 1+6+12+18+13 = 50
    ring_radii = [0.0, 0.15, 0.3, 0.45, 0.6]
    points = []
    for count, r in zip(ring_counts, ring_radii):
        if count == 1:
            points.append([0.0, 0.0])
        else:
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
            pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
            points.append(pts)
    initializations.append(np.vstack(points))
    
    # 4. Another random with different seed
    np.random.seed(42)
    initializations.append(_random_seed(N))
    
    # 5. Spiral
    t = np.linspace(0, 4 * np.pi, N)
    r_spiral = np.linspace(0.05, 0.85, N)
    initializations.append(np.column_stack([r_spiral * np.cos(t), r_spiral * np.sin(t)]))
    
    for init_idx, centers in enumerate(initializations):
        if time.time() - start_time > timeout - 30:
            break
        
        # Physics-based relaxation
        relaxed = _physics_relax(
            centers,
            max_iters=15000,
            lr_init=0.0005,
            lr_decay=0.998,
            repulsion_power=4.0,
            wall_strength=100.0,
            margin=0.005,
            time_limit=timeout - 30,
            start_time=start_time
        )
        
        # Compute score
        r = _compute_score(relaxed)
        
        if r > best_r:
            best_r = r
            best_centers = relaxed.copy()
    
    # Final polish: local hill climbing on best result
    if best_centers is not None and time.time() - start_time < timeout - 10:
        centers = best_centers.copy()
        current_r = best_r
        
        dist_mat = np.linalg.norm(centers[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=-1)
        np.fill_diagonal(dist_mat, np.inf)
        boundary_dists = 1.0 - np.linalg.norm(centers, axis=1)
        
        step_size = 0.005
        no_improve = 0
        
        while time.time() - start_time < timeout - 5:
            i = np.random.randint(N)
            delta = np.random.normal(0, step_size, size=2)
            new_c = centers[i] + delta
            if np.linalg.norm(new_c) > 1.0:
                continue
            
            old_row = dist_mat[i].copy()
            new_dists = np.linalg.norm(centers - new_c, axis=1)
            new_dists[i] = np.inf
            dist_mat[i] = new_dists
            dist_mat[:, i] = new_dists
            
            min_pair = np.min(dist_mat) / 2.0
            boundary_dists[i] = 1.0 - np.linalg.norm(new_c)
            min_bound = np.min(boundary_dists)
            
            candidate_r = min(min_pair, min_bound)
            
            if candidate_r > current_r + 1e-12:
                centers[i] = new_c
                current_r = candidate_r
                no_improve = 0
                if current_r > best_r:
                    best_r = current_r
                    best_centers = centers.copy()
                step_size = min(step_size * 1.02, 0.02)
            else:
                dist_mat[i] = old_row
                dist_mat[:, i] = old_row
                boundary_dists[i] = 1.0 - np.linalg.norm(centers[i])
                no_improve += 1
            
            if no_improve > 100:
                step_size *= 0.85
                no_improve = 0
                if step_size < 0.0001:
                    break
    
    # Final validation
    if best_centers is not None:
        radii = np.linalg.norm(best_centers, axis=1)
        if np.any(radii >= 1.0):
            best_centers = best_centers * (1.0 - 1e-8) / np.maximum(radii, 1e-12)
        # Ensure all values are finite
        if not np.all(np.isfinite(best_centers)):
            best_centers = _hexagonal_seed(N)
    
    return best_centers
# EVOLVE-BLOCK-END

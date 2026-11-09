import numpy as np
import time

N = 50
START_TIME = time.time()
TIME_LIMIT = 145.0  # Safety margin before 150s kill

def _score(points):
    """Compute the score: largest common radius."""
    if len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    # Check all points inside unit circle
    norms = np.hypot(points[:, 0], points[:, 1])
    if np.any(norms > 1.0):
        return 0.0
    # Min distance from any point to boundary
    boundary_min = 1.0 - np.max(norms)
    # Min pairwise distance / 2
    if N > 1:
        # Efficient pairwise distance calculation
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(d, np.inf)
        min_pair = np.min(d) / 2.0
    else:
        min_pair = boundary_min
    return min(min_pair, boundary_min)

def _hexagonal_initial(spacing):
    """Generate hexagonal lattice points inside unit disk."""
    points = []
    k = int(1.0 / spacing) + 2
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x * x + y * y <= (1.0 - spacing / 2.0) ** 2:
                points.append([x, y])
    points = np.array(points)
    if len(points) < N:
        return None
    # Select N closest to origin
    norms = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(norms, kind='stable')[:N]
    return points[idx]

def _molecular_dynamics(points, time_limit):
    """
    Molecular Dynamics simulation with cooling.
    Uses a repulsive potential to push circles apart.
    """
    current = points.copy()
    velocities = np.zeros_like(current)
    
    # Initial temperature
    T = 0.01
    T_min = 1e-6
    cooling_rate = 0.999
    
    dt = 0.001
    mass = 1.0
    
    # Potential strength
    repulsion_strength = 1.0
    boundary_strength = 1.0
    
    best = current.copy()
    best_score = _score(current)
    
    rng = np.random.default_rng(12345)
    
    # Precompute indices for pairs
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    while time.time() < time_limit:
        # Calculate forces
        forces = np.zeros_like(current)
        
        # Pairwise repulsion
        diff = current[i_idx] - current[j_idx]
        d = np.linalg.norm(diff, axis=1)
        
        # Avoid division by zero
        d_safe = np.where(d < 1e-8, 1e-8, d)
        
        # Force magnitude: k / d^2 (softened)
        # We want to maximize min distance, so repulsion should be strong when close
        # Using a potential like 1/d^2 gives force ~ 1/d^3
        # But we can use a simpler repulsion: F = (1 - d) / d^2 if d < 1, else 0?
        # Actually, for packing, we just want them to spread out.
        # Let's use a repulsive force that is always active but decays quickly.
        # F = repulsion_strength / (d^2 + eps)
        f_mag = repulsion_strength / (d_safe**2)
        
        # Direction
        dir_vec = diff / d_safe[:, np.newaxis]
        
        # Apply force
        # F_i += f * dir, F_j -= f * dir
        np.add.at(forces, i_idx, f_mag[:, np.newaxis] * dir_vec)
        np.add.at(forces, j_idx, -f_mag[:, np.newaxis] * dir_vec)
        
        # Boundary force: push back if outside
        norms = np.linalg.norm(current, axis=1)
        outside = norms > 1.0
        if np.any(outside):
            # Force proportional to violation
            f_boundary = boundary_strength * (norms[outside] - 1.0)
            dir_boundary = current[outside] / norms[outside, np.newaxis]
            forces[outside] -= f_boundary[:, np.newaxis] * dir_boundary
        
        # Update velocities (with damping)
        damping = 0.99
        velocities = velocities * damping + forces * dt / mass
        
        # Thermal noise
        velocities += rng.normal(0, np.sqrt(T * dt), (N, 2))
        
        # Update positions
        current = current + velocities * dt
        
        # Hard constraint: if outside, push back in
        norms = np.linalg.norm(current, axis=1)
        outside = norms > 1.0
        if np.any(outside):
            current[outside] = current[outside] / norms[outside, np.newaxis] * 0.999
            # Zero velocity for outside points to prevent oscillation
            velocities[outside] = 0.0
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
        
        # Evaluate score periodically
        if np.random.random() < 0.01: # ~1% of steps
            score = _score(current)
            if score > best_score:
                best_score = score
                best = current.copy()
        
    return best, best_score

def _local_optimize(points):
    """
    A simple iterative local optimization to polish the result.
    Uses a soft-min objective with gradient descent.
    """
    pts = points.copy()
    lr = 0.01
    k = 100.0 # Soft-min sharpness
    
    for _ in range(1000):
        # Compute distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(d, np.inf)
        
        # Soft min gradient
        y = -k * d
        y_flat = y.flatten()
        y_valid = y_flat[np.isfinite(y_flat)]
        
        if len(y_valid) == 0:
            break
            
        max_y = np.max(y_valid)
        exp_terms = np.exp(y - max_y)
        np.fill_diagonal(exp_terms, 0.0)
        
        S = np.sum(exp_terms)
        
        if S == 0:
            break
            
        grad = np.zeros_like(pts)
        
        # Pairwise contribution
        safe_d = np.where(d > 1e-8, d, 1.0)
        term = exp_terms[:, :, np.newaxis] * diff / safe_d[:, :, np.newaxis]
        grad_pair = np.sum(term, axis=1)
        grad_pair /= S
        
        # Boundary contribution
        norms = np.hypot(pts[:, 0], pts[:, 1])
        safe_norms = np.where(norms > 1e-8, norms, 1.0)
        mask = norms > 1.0
        grad_boundary = np.zeros_like(pts)
        if np.any(mask):
            grad_boundary[mask] = 100.0 * (norms[mask, np.newaxis] - 1.0) * pts[mask] / safe_norms[mask, np.newaxis]
        
        total_grad = grad_pair + grad_boundary
        
        # Update
        pts = pts - lr * total_grad
        
        # Project back to unit disk
        norms = np.hypot(pts[:, 0], pts[:, 1])
        mask = norms > 1.0
        if np.any(mask):
            pts[mask] = pts[mask] / norms[mask, np.newaxis] * 0.999
            
        # Adaptive learning rate
        if np.random.random() < 0.1:
            lr *= 0.95
            
    return pts

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_points = None
    best_score = 0.0
    
    # Initializations
    initializations = []
    
    # Hexagonal
    for spacing in [0.20, 0.22, 0.24, 0.26, 0.28]:
        pts = _hexagonal_initial(spacing)
        if pts is not None:
            initializations.append(pts)
            
    # Ring
    def _ring_initial():
        points = []
        points.append([0.0, 0.0])
        radii = [0.3, 0.55, 0.75, 0.9]
        counts = [6, 12, 18, 13]
        for r, n in zip(radii, counts):
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            for a in angles:
                points.append([r * np.cos(a), r * np.sin(a)])
        return np.array(points[:N])
    initializations.append(_ring_initial())
    
    # Random
    rng = np.random.default_rng(42)
    for _ in range(3):
        points = []
        for _ in range(N):
            while True:
                x = rng.uniform(-0.9, 0.9)
                y = rng.uniform(-0.9, 0.9)
                if x * x + y * y <= 0.81:
                    points.append([x, y])
                    break
        initializations.append(np.array(points))
        
    # Strategy:
    # 1. Run Molecular Dynamics for each initialization.
    # 2. Polish the best result with local optimization.
    
    # Time budget allocation
    # We have ~145s.
    # Let's spend 70% on MD and 30% on polishing.
    
    md_time_per_run = 15.0
    max_md_runs = int((TIME_LIMIT * 0.7) / md_time_per_run)
    
    for i, init_points in enumerate(initializations):
        if time.time() - START_TIME > TIME_LIMIT * 0.7:
            break
            
        # Run MD
        md_end_time = time.time() + md_time_per_run
        md_points, md_score = _molecular_dynamics(init_points, md_end_time)
        
        if md_score > best_score:
            best_score = md_score
            best_points = md_points.copy()
            
    # Final Polish Phase
    if best_points is not None:
        remaining_time = TIME_LIMIT - (time.time() - START_TIME)
        if remaining_time > 5:
            # Run local optimization on the best points
            polished = _local_optimize(best_points)
            score_p = _score(polished)
            if score_p > best_score:
                best_score = score_p
                best_points = polished.copy()
                
            # Try a few more polishes with slight perturbations
            for _ in range(3):
                if time.time() - START_TIME > TIME_LIMIT - 2:
                    break
                perturbed = best_points + np.random.normal(0, 0.005, (N, 2))
                norms = np.hypot(perturbed[:, 0], perturbed[:, 1])
                mask = norms > 1.0
                if np.any(mask):
                    perturbed[mask] = perturbed[mask] / norms[mask, np.newaxis] * 0.999
                    
                polished = _local_optimize(perturbed)
                score_p = _score(polished)
                if score_p > best_score:
                    best_score = score_p
                    best_points = polished.copy()

    # Fallback
    if best_points is None:
        best_points = _hexagonal_initial(0.25)
        if best_points is None:
            best_points = np.zeros((N, 2))

    return best_points

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_score(pts)}")
